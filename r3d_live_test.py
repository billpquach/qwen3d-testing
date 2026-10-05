"""Record3D live USB test: does the stream give a real camera pose with every frame?

No window, no OpenCV needed. Run it, then slowly move the phone ~0.5 m while it records.
    pip install record3d numpy
    python r3d_live_test.py            # 10 seconds
    python r3d_live_test.py 20         # 20 seconds
"""
import sys
import time
from threading import Event

import numpy as np
from record3d import Record3DStream

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0

devs = Record3DStream.get_connected_devices()
print(f"{len(devs)} device(s) found")
for d in devs:
    print(f"  product_id={d.product_id} udid={d.udid}")
if not devs:
    sys.exit("No device. Check the cable, 'Trust This Computer', usbmuxd, and that USB streaming is started in the app.")

new_frame, stopped = Event(), Event()
session = Record3DStream()
session.on_new_frame = new_frame.set
session.on_stream_stopped = stopped.set
session.connect(devs[0])

poses, t0, n = [], time.time(), 0
print(f"recording {SECONDS:.0f} s, move the phone slowly...")
while time.time() - t0 < SECONDS and not stopped.is_set():
    if not new_frame.wait(timeout=5):
        sys.exit("No frames for 5 s. Is the stream running on the phone (red record button)?")
    new_frame.clear()
    depth = session.get_depth_frame()
    rgb = session.get_rgb_frame()
    conf = session.get_confidence_frame()
    k = session.get_intrinsic_mat()
    p = session.get_camera_pose()
    poses.append([p.qx, p.qy, p.qz, p.qw, p.tx, p.ty, p.tz])
    if n == 0:
        print(f"device type: {session.get_device_type()} (1 = LiDAR)")
        print(f"rgb {rgb.shape} {rgb.dtype} | depth {depth.shape} {depth.dtype} "
              f"range {np.nanmin(depth):.2f}-{np.nanmax(depth):.2f} m | conf {conf.shape}")
        print(f"intrinsics fx={k.fx:.1f} fy={k.fy:.1f} cx={k.tx:.1f} cy={k.ty:.1f}")
    if n % 15 == 0:
        print(f"frame {n:4d}  q=({p.qx:+.3f},{p.qy:+.3f},{p.qz:+.3f},{p.qw:+.3f})  "
              f"t=({p.tx:+.3f},{p.ty:+.3f},{p.tz:+.3f}) m")
    n += 1

session.disconnect()
P = np.array(poses)
elapsed = time.time() - t0
moved = np.linalg.norm(P[:, 4:] - P[0, 4:], axis=1).max() if len(P) else 0.0
path = np.linalg.norm(np.diff(P[:, 4:], axis=0), axis=1).sum() if len(P) > 1 else 0.0
qnorm = np.linalg.norm(P[:, :4], axis=1) if len(P) else np.array([0.0])
print("\n=== SUMMARY (send me this) ===")
print(f"frames: {n} in {elapsed:.1f} s ({n / elapsed:.1f} fps)")
print(f"max distance from start: {moved:.3f} m, path length: {path:.3f} m")
print(f"quaternion norms: {qnorm.min():.4f}-{qnorm.max():.4f}")
print(f"all-zero poses: {int((np.abs(P).sum(1) == 0).sum())}, identity poses: "
      f"{int(np.all(np.isclose(P, [0, 0, 0, 1, 0, 0, 0]), axis=1).sum())}")
print("VERDICT:", "POSE OK" if moved > 0.05 and abs(qnorm.mean() - 1) < 0.01
      else "POSE MISSING OR STATIC (did you move the phone?)")
