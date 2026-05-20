"""Quick diagnostic: print raw scan data to terminal to see if the LiDAR is sending anything."""
import sys
from rplidar import RPLidar

PORT = "/dev/cu.usbserial-0001"

lidar = RPLidar(PORT, baudrate=115200, timeout=3)
print("Info:", lidar.get_info())
print("Health:", lidar.get_health())

print("\nStarting scan... (Ctrl+C to stop)\n")
try:
    for i, scan in enumerate(lidar.iter_scans(max_buf_meas=2000, min_len=5)):
        print(f"Scan {i}: got {len(scan)} points. First point: {scan[0] if scan else 'empty'}")
        if i >= 10:
            break
except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
finally:
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
    print("Done.")