"""
Real-time 2D visualization for the Slamtec RPLIDAR A1M8.

Reads 360 degree scan data over USB serial and plots it on a live Cartesian chart.
Points within 1 foot of the sensor are highlighted in red.

Tested with the rplidar-roboticia Python library on macOS (Apple Silicon).

Usage:
    python lidar_viewer.py                                     # auto-detect port
    python lidar_viewer.py --port /dev/cu.usbserial-0001       # macOS
    python lidar_viewer.py --port /dev/ttyUSB0                 # Linux
    python lidar_viewer.py --port COM3                         # Windows

Press Ctrl+C in the terminal or close the plot window to exit cleanly.
"""

import matplotlib
matplotlib.use("MacOSX")  # native backend on macOS; works without extra deps

import argparse
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from rplidar import RPLidar, RPLidarException

# ---------- Configuration ----------
DEFAULT_BAUDRATE = 115200          # A1M8 uses 115200; A2/A3 use 256000
DEFAULT_MAX_DISTANCE_MM = 6000     # plot half-width in millimeters (A1 reaches ~12000)
POINT_SIZE = 4
SCAN_TIMEOUT = 3                   # seconds
DANGER_MM = 304.8                  # 1 foot in millimeters


def auto_detect_port():
    """Try to find the LiDAR's serial port across OSes. Returns None if none found."""
    candidates = []
    if sys.platform.startswith("linux"):
        candidates = sorted(glob("/dev/ttyUSB*") + glob("/dev/ttyACM*"))
    elif sys.platform == "darwin":
        candidates = sorted(
            glob("/dev/cu.usbserial*")
            + glob("/dev/cu.SLAB_USBtoUART*")
            + glob("/dev/cu.usbmodem*")
        )
    elif sys.platform.startswith("win"):
        try:
            from serial.tools import list_ports
            candidates = [p.device for p in list_ports.comports()]
        except ImportError:
            candidates = []
    return candidates[0] if candidates else None


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time RPLIDAR A1M8 viewer")
    parser.add_argument("--port", default=None,
                        help="Serial port (auto-detected if omitted)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUDRATE,
                        help=f"Baud rate (default {DEFAULT_BAUDRATE})")
    parser.add_argument("--max-distance", type=int, default=DEFAULT_MAX_DISTANCE_MM,
                        help="Max plot extent in mm (default 6000)")
    return parser.parse_args()


def main():
    args = parse_args()
    port = args.port or auto_detect_port()
    if not port:
        sys.exit("ERROR: no serial port specified and none auto-detected. "
                 "Pass one with --port (e.g. /dev/cu.usbserial-0001 or COM3).")

    print(f"Connecting to RPLIDAR on {port} @ {args.baud} baud...")
    lidar = RPLidar(port, baudrate=args.baud, timeout=SCAN_TIMEOUT)

    try:
        info = lidar.get_info()
        health = lidar.get_health()
        print(f"Device info: {info}")
        print(f"Health: {health}")
        if health[0] != "Good":
            print("WARNING: device health is not 'Good'. Continuing anyway.")
    except RPLidarException as e:
        lidar.stop()
        lidar.disconnect()
        sys.exit(f"ERROR talking to LiDAR: {e}")

    # ---------- Matplotlib setup ----------
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)  # Cartesian (square) plot
    lim = args.max_distance
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)", color="#888888")
    ax.set_ylabel("Y (mm)", color="#888888")
    ax.tick_params(colors="#888888")
    ax.grid(color="#333333")
    ax.set_title("RPLIDAR A1M8 — live 2D scan", color="white", pad=18)

    # Dashed red ring at 1 foot — the danger threshold
    danger_ring = plt.Circle((0, 0), DANGER_MM, fill=False,
                             edgecolor="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.add_patch(danger_ring)

    # Sensor marker at origin
    ax.plot(0, 0, marker="o", color="cyan", markersize=8)

    # Generator yields one full revolution at a time as a list of (quality, angle, dist)
    scan_iter = lidar.iter_scans(max_buf_meas=2000, min_len=5)

    def update(_frame):
        try:
            scan = next(scan_iter)
        except StopIteration:
            return []
        except RPLidarException as e:
            print(f"Scan error (continuing): {e}")
            return []

        angles_deg = np.array([m[1] for m in scan], dtype=float)
        distances = np.array([m[2] for m in scan], dtype=float)

        mask = distances > 0
        ang = np.deg2rad(angles_deg[mask])
        dist = distances[mask]

        # Polar -> Cartesian. 0 deg on the LiDAR points "forward" (+Y on screen).
        x = dist * np.sin(ang)
        y = dist * np.cos(ang)

        # Red if within 1 foot, white otherwise.
        danger = dist < DANGER_MM
        colors = np.where(danger, "red", "white")

        if danger.any():
            print(f"Frame: {mask.sum()} pts | WARNING {danger.sum()} within 1 ft")
        else:
            print(f"Frame: {mask.sum()} pts | nearest: {dist.min():.0f}mm")

        # Clear previous scan points (the danger ring is a Patch, not a Collection,
        # so it survives this).
        for coll in ax.collections:
            coll.remove()

        ax.scatter(x, y, s=POINT_SIZE * 4, c=colors)
        return []

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down LiDAR...")
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()