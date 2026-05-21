"""
Live ICP-based SLAM for the Slamtec RPLIDAR A1M8.

Pure Python implementation using only numpy + scipy + matplotlib + rplidar.
No BreezySLAM or other SLAM libraries needed.

How it works:
    For each new scan, run Iterative Closest Point (ICP) to find the rigid
    transform (dx, dy, dtheta) that best aligns the new scan against a
    local "reference cloud" built from recent scans. The trajectory is the
    chain of these transforms; the map is the accumulated aligned points.

Tips for a good map:
    - Walk SLOWLY (~0.3 m/s). ICP relies on consecutive scans overlapping
      heavily; fast motion breaks that assumption.
    - Keep the LiDAR LEVEL.
    - Start in a feature-rich spot (corner, near furniture).
    - Bland hallways are the algorithm's weak point.
    - Close the plot window when done to save the map as PNG.

Usage:
    python slam_viewer.py --port /dev/cu.usbserial-0001
"""

import matplotlib
matplotlib.use("MacOSX")

import argparse
import sys
import threading
import time
from glob import glob
from queue import Queue, Empty

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d

from rplidar import RPLidar, RPLidarException


# ---------- LiDAR config ----------
DEFAULT_BAUDRATE = 115200
SCAN_TIMEOUT = 3
SCAN_QUEUE_MAX = 4

# ---------- SLAM config ----------
MIN_RANGE_MM = 150
MAX_RANGE_MM = 8000
DOWNSAMPLE_GRID_MM = 50

ICP_MAX_ITERATIONS = 25
ICP_TOLERANCE = 1e-4
ICP_MAX_PAIR_DIST_MM = 500

REF_CLOUD_MAX_POINTS = 8000
KEYFRAME_DIST_MM = 200
KEYFRAME_ROT_DEG = 10

# Rotation histogram pre-alignment
HIST_BINS = 360              # 1 deg resolution
HIST_BLUR_SIGMA = 2.0        # gaussian blur the histogram for smoother correlation
HIST_MIN_OVERLAP_PTS = 50    # need at least this many ref points to trust the estimate

# Visualization
VIEW_PADDING_M = 2.0
VIEW_MIN_SIZE_M = 6.0
VIEW_SMOOTH = 0.15


def auto_detect_port():
    if sys.platform.startswith("linux"):
        c = sorted(glob("/dev/ttyUSB*") + glob("/dev/ttyACM*"))
    elif sys.platform == "darwin":
        c = sorted(glob("/dev/cu.usbserial*") + glob("/dev/cu.SLAB_USBtoUART*"))
    elif sys.platform.startswith("win"):
        try:
            from serial.tools import list_ports
            c = [p.device for p in list_ports.comports()]
        except ImportError:
            c = []
    else:
        c = []
    return c[0] if c else None


def parse_args():
    p = argparse.ArgumentParser(description="ICP-SLAM viewer for RPLIDAR A1M8")
    p.add_argument("--port", default=None)
    p.add_argument("--baud", type=int, default=DEFAULT_BAUDRATE)
    p.add_argument("--output", default="room_map.png")
    return p.parse_args()


def scan_to_points(scan):
    """Convert rplidar scan (quality, angle_deg, dist_mm) to Nx2 sensor-frame mm.
    0 degrees -> +Y."""
    pts = []
    for (_q, ang_deg, dist) in scan:
        if MIN_RANGE_MM <= dist <= MAX_RANGE_MM:
            a = np.deg2rad(ang_deg)
            pts.append((dist * np.sin(a), dist * np.cos(a)))
    return np.array(pts, dtype=np.float64) if pts else np.zeros((0, 2))


def voxel_downsample(points, grid_mm):
    if len(points) == 0:
        return points
    keys = np.floor(points / grid_mm).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]


def apply_transform(points, dx, dy, dtheta):
    if len(points) == 0:
        return points
    c, s = np.cos(dtheta), np.sin(dtheta)
    R = np.array([[c, -s], [s, c]])
    return points @ R.T + np.array([dx, dy])


def angle_histogram(points, bins=HIST_BINS):
    """Build a 1D histogram of point angles (radians, 0..2pi).

    This is rotation-INVARIANT in the sense that rotating the cloud by theta
    just shifts the histogram by theta. So correlating two histograms reveals
    the rotation between two scans.

    We weight each angle bin by *something more informative than just count*
    -- we use the median distance in that bin. This makes the histogram
    encode the shape of the room (closer in some directions, farther in
    others), not just where the LiDAR happened to get returns. That makes
    correlation way more discriminative for rotation.
    """
    if len(points) == 0:
        return np.zeros(bins)

    # Angle of each point, in [0, 2*pi)
    ang = (np.arctan2(points[:, 1], points[:, 0]) + 2 * np.pi) % (2 * np.pi)
    dist = np.linalg.norm(points, axis=1)

    bin_idx = (ang / (2 * np.pi) * bins).astype(int) % bins

    hist = np.zeros(bins)
    counts = np.zeros(bins)
    np.add.at(hist, bin_idx, dist)
    np.add.at(counts, bin_idx, 1)

    # Mean distance per bin (0 where there are no points)
    with np.errstate(divide="ignore", invalid="ignore"):
        hist = np.where(counts > 0, hist / np.maximum(counts, 1), 0)

    # Smooth so single-point noise doesn't dominate
    if HIST_BLUR_SIGMA > 0:
        hist = gaussian_filter1d(hist, sigma=HIST_BLUR_SIGMA, mode="wrap")

    return hist


def estimate_rotation(source_pts, target_pts):
    """Estimate the rotation (radians) that aligns source angles to target angles
    using circular cross-correlation of angle histograms. Returns 0.0 if either
    cloud is too sparse to give a confident answer.
    """
    if len(source_pts) < HIST_MIN_OVERLAP_PTS or len(target_pts) < HIST_MIN_OVERLAP_PTS:
        return 0.0

    h_src = angle_histogram(source_pts)
    h_tgt = angle_histogram(target_pts)

    # Subtract means so DC component doesn't dominate correlation
    h_src = h_src - h_src.mean()
    h_tgt = h_tgt - h_tgt.mean()

    # Circular cross-correlation via FFT.
    # We want the shift k such that h_tgt[i] ≈ h_src[i - k] for all i,
    # i.e. source is rotated by +k bins to match target.
    corr = np.fft.irfft(np.fft.rfft(h_tgt) * np.conj(np.fft.rfft(h_src)),
                        n=HIST_BINS)
    best_shift = int(np.argmax(corr))

    # Convert bin shift to radians; wrap to [-pi, pi]
    dtheta = best_shift * (2 * np.pi / HIST_BINS)
    if dtheta > np.pi:
        dtheta -= 2 * np.pi
    return dtheta


def icp(source, target, init_dx=0.0, init_dy=0.0, init_dtheta=0.0,
        max_iter=ICP_MAX_ITERATIONS, tol=ICP_TOLERANCE,
        max_pair_dist=ICP_MAX_PAIR_DIST_MM):
    """Align source to target. Returns (dx, dy, dtheta, mean_error_mm).
    target ≈ apply_transform(source, dx, dy, dtheta)."""
    if len(source) < 5 or len(target) < 5:
        return init_dx, init_dy, init_dtheta, float("inf")

    tree = cKDTree(target)
    dx, dy, dtheta = init_dx, init_dy, init_dtheta
    prev_error = float("inf")

    for _ in range(max_iter):
        src_t = apply_transform(source, dx, dy, dtheta)
        dists, idxs = tree.query(src_t, k=1, workers=-1)

        mask = dists < max_pair_dist
        if mask.sum() < 5:
            break
        src_pairs = src_t[mask]
        tgt_pairs = target[idxs[mask]]

        sc = src_pairs.mean(axis=0)
        tc = tgt_pairs.mean(axis=0)
        src_c = src_pairs - sc
        tgt_c = tgt_pairs - tc

        # Kabsch: optimal rotation via SVD
        H = src_c.T @ tgt_c
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T

        t = tc - R @ sc
        ddtheta = np.arctan2(R[1, 0], R[0, 0])

        dx, dy = R @ np.array([dx, dy]) + t
        dtheta += ddtheta

        error = dists[mask].mean()
        if abs(prev_error - error) < tol * max_pair_dist:
            break
        prev_error = error

    return dx, dy, dtheta, prev_error


class LidarThread(threading.Thread):
    def __init__(self, lidar, scan_queue):
        super().__init__(daemon=True)
        self.lidar = lidar
        self.queue = scan_queue
        self.running = True

    def run(self):
        try:
            for scan in self.lidar.iter_scans(max_buf_meas=2000, min_len=5):
                if not self.running:
                    break
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except Empty:
                        pass
                self.queue.put(scan)
        except RPLidarException as e:
            print(f"LiDAR thread error: {e}")
        except Exception as e:
            print(f"LiDAR thread crashed: {e}")

    def stop(self):
        self.running = False


def main():
    args = parse_args()
    port = args.port or auto_detect_port()
    if not port:
        sys.exit("ERROR: no serial port specified or detected.")

    print(f"Connecting to LiDAR on {port}...")
    lidar = RPLidar(port, baudrate=args.baud, timeout=SCAN_TIMEOUT)
    try:
        print(f"Info: {lidar.get_info()}")
        print(f"Health: {lidar.get_health()}")
    except RPLidarException as e:
        lidar.stop(); lidar.disconnect()
        sys.exit(f"LiDAR error: {e}")

    scan_queue = Queue(maxsize=SCAN_QUEUE_MAX)
    lidar_thread = LidarThread(lidar, scan_queue)
    lidar_thread.start()

    # SLAM state
    pose = np.array([0.0, 0.0, 0.0])  # (x_mm, y_mm, theta_rad) in world frame
    trajectory_m = [(0.0, 0.0)]
    map_points = []           # list of (N,2) arrays in world mm
    ref_cloud = np.zeros((0, 2))
    last_kf_pose = pose.copy()

    # Matplotlib
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 9))
    map_scatter = ax.scatter([], [], s=1, c="white", alpha=0.7)
    traj_line, = ax.plot([], [], color="cyan", linewidth=1.5, label="Trajectory")
    pose_dot, = ax.plot([0], [0], marker="o", color="red", markersize=8, label="Current pose")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("ICP-SLAM — walk slowly, keep level")
    ax.legend(loc="upper right", facecolor="#222", edgecolor="#444")
    ax.set_aspect("equal")

    half = VIEW_MIN_SIZE_M / 2.0
    view = {"xmin": -half, "xmax": half, "ymin": -half, "ymax": half}
    ax.set_xlim(view["xmin"], view["xmax"])
    ax.set_ylim(view["ymin"], view["ymax"])

    scan_count = [0]
    keyframe_count = [0]

    def update(_frame):
        nonlocal pose, ref_cloud, last_kf_pose

        try:
            scan = scan_queue.get_nowait()
        except Empty:
            return []

        scan_count[0] += 1

        pts_sensor = scan_to_points(scan)
        pts_sensor = voxel_downsample(pts_sensor, DOWNSAMPLE_GRID_MM)
        if len(pts_sensor) < 10:
            return []

        if len(ref_cloud) > 50:
            # Step A: estimate rotation from angle histograms BEFORE ICP.
            # We compare the new scan (sensor frame) against the reference
            # cloud transformed into the previous sensor frame. The result
            # is the rotation change since the last pose, which we add to
            # the previous theta to get the world-frame initial guess.
            ref_in_prev_sensor = apply_transform(
                ref_cloud, -pose[0], -pose[1], 0.0
            )
            # Now rotate that by -pose[2] to fully bring into sensor frame
            ref_in_prev_sensor = apply_transform(
                ref_in_prev_sensor, 0.0, 0.0, -pose[2]
            )
            ddtheta = estimate_rotation(pts_sensor, ref_in_prev_sensor)
            init_theta = pose[2] + ddtheta

            dx, dy, dtheta, err = icp(
                pts_sensor, ref_cloud,
                init_dx=pose[0], init_dy=pose[1], init_dtheta=init_theta,
            )
            pose = np.array([dx, dy, dtheta])
        else:
            err = 0.0

        pts_world = apply_transform(pts_sensor, pose[0], pose[1], pose[2])

        # Keyframe decision
        d_trans = np.linalg.norm(pose[:2] - last_kf_pose[:2])
        d_rot = abs(np.rad2deg(
            (pose[2] - last_kf_pose[2] + np.pi) % (2 * np.pi) - np.pi))
        is_first = len(map_points) == 0
        if is_first or d_trans > KEYFRAME_DIST_MM or d_rot > KEYFRAME_ROT_DEG:
            map_points.append(pts_world)
            last_kf_pose = pose.copy()
            keyframe_count[0] += 1

            all_pts = np.vstack(map_points)
            if len(all_pts) > REF_CLOUD_MAX_POINTS:
                idx = np.random.choice(len(all_pts), REF_CLOUD_MAX_POINTS, replace=False)
                ref_cloud = all_pts[idx]
            else:
                ref_cloud = all_pts

        # Display
        x_m, y_m = pose[0] / 1000.0, pose[1] / 1000.0
        trajectory_m.append((x_m, y_m))

        if map_points:
            all_pts_m = np.vstack(map_points) / 1000.0
            map_scatter.set_offsets(all_pts_m)

        xs, ys = zip(*trajectory_m)
        traj_line.set_data(xs, ys)
        pose_dot.set_data([x_m], [y_m])

        # Auto-zoom
        SENSOR_RANGE_M = MAX_RANGE_MM / 1000.0
        tx_min = min(xs) - SENSOR_RANGE_M
        tx_max = max(xs) + SENSOR_RANGE_M
        ty_min = min(ys) - SENSOR_RANGE_M
        ty_max = max(ys) + SENSOR_RANGE_M
        target_size = max(tx_max - tx_min, ty_max - ty_min,
                          VIEW_MIN_SIZE_M) + 2 * VIEW_PADDING_M
        cx, cy = (tx_min + tx_max) / 2, (ty_min + ty_max) / 2
        target = {
            "xmin": cx - target_size / 2, "xmax": cx + target_size / 2,
            "ymin": cy - target_size / 2, "ymax": cy + target_size / 2,
        }
        for k in view:
            view[k] += VIEW_SMOOTH * (target[k] - view[k])
        ax.set_xlim(view["xmin"], view["xmax"])
        ax.set_ylim(view["ymin"], view["ymax"])

        ax.set_title(
            f"ICP-SLAM — {scan_count[0]} scans, {keyframe_count[0]} keyframes, "
            f"pose ({x_m:.2f}, {y_m:.2f}) m, err {err:.0f}mm"
        )
        return []

    ani = FuncAnimation(fig, update, interval=80, blit=False, cache_frame_data=False)

    print("\nReady. Walk slowly. Close the plot window to save the map.\n")
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping...")
        lidar_thread.stop()
        time.sleep(0.5)
        try:
            lidar.stop()
            lidar.stop_motor()
            lidar.disconnect()
        except Exception:
            pass

        if map_points:
            all_pts_m = np.vstack(map_points) / 1000.0
            xs, ys = zip(*trajectory_m)
            fig_out, ax_out = plt.subplots(figsize=(12, 12))
            ax_out.scatter(all_pts_m[:, 0], all_pts_m[:, 1], s=1, c="black")
            ax_out.plot(xs, ys, color="red", linewidth=1, alpha=0.6, label="Path")
            ax_out.set_aspect("equal")
            ax_out.set_xlabel("X (m)")
            ax_out.set_ylabel("Y (m)")
            ax_out.set_title(f"Final map — {scan_count[0]} scans, "
                             f"{keyframe_count[0]} keyframes")
            ax_out.legend()
            ax_out.grid(True, alpha=0.3)
            fig_out.savefig(args.output, dpi=150, bbox_inches="tight")
            print(f"Saved map → {args.output}")
        print(f"Total scans: {scan_count[0]}, keyframes: {keyframe_count[0]}")


if __name__ == "__main__":
    main()