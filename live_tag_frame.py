"""Live test of the shared world frame: phone pose (ARKit) + AprilTag -> tag-anchored coordinates.

Every part of the demo (phone, Jetson, dashboard, rover) will measure positions relative to the
tag. This script checks that this is stable: it detects the tag ~10 times a second, computes
where the tag is in ARKit's world, and reports how much that estimate moves while you walk around.
A fixed tag should stay put; movement = ARKit drift + tag-pose noise.

    pip install record3d pupil-apriltags numpy
    python live_tag_frame.py                 # 30 s, tag size 0.1728 m
    python live_tag_frame.py 60 --tag-size 0.1728

While it runs: start ~1 m from the tag, walk slowly in an arc around the table, keep the tag in
view most of the time. It prints your headset position in tag coordinates as you move.
"""
import argparse
import json
import os
import sys
import time
from threading import Event

import numpy as np
from pupil_apriltags import Detector
from record3d import Record3DStream

ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])  # ARKit camera axes -> OpenCV camera axes


def pose_to_T(p):
    x, y, z, w = p.qx, p.qy, p.qz, p.qw
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [p.tx, p.ty, p.tz]
    return T


def rot_angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seconds", nargs="?", type=float, default=30)
    ap.add_argument("--tag-size", type=float, default=0.1728, help="black square edge in metres")
    ap.add_argument("--tag-id", type=int, default=0)
    ap.add_argument("--every", type=int, default=6, help="run detection on every Nth frame (60 fps / 6 = 10 Hz)")
    args = ap.parse_args()

    devs = Record3DStream.get_connected_devices()
    if not devs:
        sys.exit("No device: check cable, Trust, usbmuxd, and that streaming is started in the app.")
    new_frame, stopped = Event(), Event()
    s = Record3DStream()
    s.on_new_frame = new_frame.set
    s.on_stream_stopped = stopped.set
    s.connect(devs[0])

    det = Detector(families="tag36h11", quad_decimate=2.0, nthreads=4)
    world_tag, cam_in_tag, dists, n, t0, last_print = [], [], [], 0, time.time(), 0
    print(f"running {args.seconds:.0f} s, looking for tag36h11 id {args.tag_id} ({args.tag_size} m)...")
    while time.time() - t0 < args.seconds and not stopped.is_set():
        if not new_frame.wait(timeout=5):
            sys.exit("No frames for 5 s.")
        new_frame.clear()
        n += 1
        if n % args.every:
            continue
        rgb = s.get_rgb_frame()
        k = s.get_intrinsic_mat()
        T_w_cam = pose_to_T(s.get_camera_pose()) @ ARKIT_TO_CV          # camera -> ARKit world
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
        hits = [d for d in det.detect(gray, estimate_tag_pose=True, camera_params=(k.fx, k.fy, k.tx, k.ty),
                                      tag_size=args.tag_size) if d.tag_id == args.tag_id]
        if not hits:
            continue
        d = hits[0]
        T_cam_tag = np.eye(4)
        T_cam_tag[:3, :3], T_cam_tag[:3, 3] = d.pose_R, d.pose_t[:, 0]
        T_w_tag = T_w_cam @ T_cam_tag                                     # tag -> ARKit world
        world_tag.append(T_w_tag)
        dists.append(float(np.linalg.norm(d.pose_t)))
        c = np.linalg.inv(T_w_tag) @ T_w_cam[:, 3]                        # camera position in tag frame
        cam_in_tag.append(c[:3])
        if time.time() - last_print > 1.0:
            last_print = time.time()
            print(f"t={time.time() - t0:5.1f}s  tag seen {len(world_tag):3d}x  dist {dists[-1]:.2f} m  "
                  f"headset in tag frame: x={c[0]:+.2f} y={c[1]:+.2f} z={c[2]:+.2f} m  "
                  f"tag in world: ({T_w_tag[0, 3]:+.3f},{T_w_tag[1, 3]:+.3f},{T_w_tag[2, 3]:+.3f})")

    print("\n=== SUMMARY (send me this) ===")
    if len(world_tag) < 5:
        print(f"tag detected only {len(world_tag)} times: check tag id/size, lighting, distance")
    else:
        P = np.array([T[:3, 3] for T in world_tag])
        med = np.median(P, axis=0)
        dev = np.linalg.norm(P - med, axis=1)
        R0 = world_tag[int(np.argmin(dev))][:3, :3]
        ang = np.array([rot_angle_deg(R0.T @ T[:3, :3]) for T in world_tag])
        path = np.array(cam_in_tag)
        walked = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
        print(f"detections: {len(world_tag)} over {time.time() - t0:.0f} s, tag distance {min(dists):.2f}-{max(dists):.2f} m, "
              f"headset moved ~{walked:.1f} m")
        print(f"tag position spread: median {np.median(dev) * 100:.1f} cm, 95th pct {np.percentile(dev, 95) * 100:.1f} cm, "
              f"max {dev.max() * 100:.1f} cm")
        print(f"tag orientation spread: median {np.median(ang):.1f} deg, 95th pct {np.percentile(ang, 95):.1f} deg")
        p95 = np.percentile(dev, 95)
        print("VERDICT:", "STABLE (< 2 cm): one anchor is enough" if p95 < 0.02 else
              "OK (2-5 cm): re-anchor whenever the tag is in view" if p95 < 0.05 else
              "UNSTABLE (> 5 cm): send me this output (drift, bad tag size, or pose convention)")
        T_med = world_tag[int(np.argmin(dev))]
        json.dump({"tag_id": args.tag_id, "tag_size_m": args.tag_size, "world_from_tag": T_med.tolist(),
                   "note": "ARKit world <- tag frame. Tag frame (pupil-apriltags): origin at tag centre, x right, y down, z into the tag (points down for a tag lying on a table)"},
                  open("tag_anchor.json", "w"), indent=2)
        print("saved tag_anchor.json")
    sys.stdout.flush()
    os._exit(0)  # record3d's teardown segfaults; skip it


if __name__ == "__main__":
    main()
