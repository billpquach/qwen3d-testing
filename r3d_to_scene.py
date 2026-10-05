"""Convert a Record3D .r3d capture into a ScanNet-style scene folder for Qwen-3D.

Qwen-3D was trained on ScanNet / ScanNet++ frames prepared like this (see
data_preparation/scannetpp/process_snpp_rgbd_to_scannet.py): 640x480 color, depth resized
to the same size and stored as uint16 millimetres, one 4x4 camera-to-world pose per frame
in the OpenCV camera convention, and a z-up world.

Output layout:
    <out>/color/<i>.jpg        RGB, 480x640 (portrait captures stay portrait)
    <out>/depth/<i>.png        uint16 depth in mm, same size as color (nearest-neighbour upsampled)
    <out>/conf/<i>.png         uint8 LiDAR confidence 0/1/2, same size
    <out>/pose/<i>.txt         4x4 camera-to-world, OpenCV camera (x right, y down, z forward)
    <out>/intrinsic/<i>.txt    3x3 intrinsics for the 480x640 images
    <out>/meta.json            world transform (ARKit world -> scene), frame list, source info
    <out>/preview.png          sanity-check plots

Usage:
    pip install lzfse            # if that fails: pip install pyliblzfse
    python r3d_to_scene.py capture.r3d scenes/kitchen1
    python r3d_to_scene.py capture.r3d scenes/kitchen1 --step 5     # keep every 5th frame
    python r3d_to_scene.py capture.r3d scenes/kitchen1_kf --keyframes  # keep a frame only after the
                     # camera moved 10 cm or turned 15 deg, skipping blurry ones (auto-scan prototype)
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import lzfse

    decompress = lzfse.decompress
except ImportError:
    import liblzfse

    decompress = liblzfse.decompress

# ARKit camera (x right, y up, z backward) -> OpenCV camera (x right, y down, z forward)
ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
# ARKit world is y-up; ScanNet scenes are z-up. (x, y, z) -> (x, -z, y)
Y_UP_TO_Z_UP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def quat_to_R(qx, qy, qz, qw):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def backproject(depth, K, T_c2w, mask, stride=1):
    v, u = np.nonzero(mask[::stride, ::stride])
    v, u = v * stride, u * stride
    z = depth[v, u]
    pts = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)], 1)
    return (T_c2w @ pts.T).T[:, :3]


def sharpness(jpeg_bytes):
    """Variance of the Laplacian on a small grayscale copy: low = motion blur."""
    g = np.asarray(Image.open(io.BytesIO(jpeg_bytes)).convert("L").resize((240, 320)), np.float32)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def select_keyframes(zf, poses, move_m, turn_deg, blur_ratio=0.6):
    """Keep a frame when the camera has moved/turned enough since the last keyframe and the
    frame is not blurry relative to the keyframes so far. This is the logic the live auto-scan
    will run on the Jetson, applied here to a recording."""
    keep, sharp, last = [], [], None
    skipped_blur = 0
    for i, (qx, qy, qz, qw, tx, ty, tz) in enumerate(poses):
        R = quat_to_R(qx, qy, qz, qw)
        t = np.array([tx, ty, tz])
        if last is not None:
            dt = np.linalg.norm(t - last[1])
            ang = np.degrees(np.arccos(np.clip((np.trace(last[0].T @ R) - 1) / 2, -1, 1)))
            if dt < move_m and ang < turn_deg:
                continue
        s = sharpness(zf.read(f"rgbd/{i}.jpg"))
        if len(sharp) >= 3 and s < blur_ratio * np.median(sharp):
            skipped_blur += 1
            continue  # try the next frame instead; the motion test stays satisfied
        keep.append(i)
        sharp.append(s)
        last = (R, t)
    print(f"keyframes: {len(keep)} of {len(poses)} frames (moved >= {move_m * 100:.0f} cm or turned "
          f">= {turn_deg:.0f} deg; {skipped_blur} blurry candidates skipped)")
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("r3d")
    ap.add_argument("out")
    ap.add_argument("--step", type=int, default=10, help="keep every Nth frame (60 fps capture -> 6 fps at 10)")
    ap.add_argument("--max-depth", type=float, default=4.0, help="ignore depth beyond this when fitting the floor (m)")
    ap.add_argument("--keyframes", action="store_true", help="select frames by camera motion + sharpness instead of --step")
    ap.add_argument("--kf-move", type=float, default=0.10, help="keyframe after this much camera movement (m)")
    ap.add_argument("--kf-turn", type=float, default=15.0, help="keyframe after this much camera rotation (deg)")
    args = ap.parse_args()

    out = Path(args.out)
    for d in ("color", "depth", "conf", "pose", "intrinsic"):
        (out / d).mkdir(parents=True, exist_ok=True)

    zf = zipfile.ZipFile(args.r3d)
    meta = json.loads(zf.read("metadata"))
    W, H, dw, dh = meta["w"], meta["h"], meta["dw"], meta["dh"]
    n_total = len(meta["poses"])
    coeffs = meta.get("perFrameIntrinsicCoeffs") or [None] * n_total
    K0 = np.array(meta["K"], dtype=np.float64).reshape(3, 3).T  # stored column-major
    out_w, out_h = (480, 640) if W < H else (640, 480)
    sx, sy = out_w / W, out_h / H
    print(f"capture: {n_total} frames at {meta.get('fps')} fps, RGB {W}x{H}, depth {dw}x{dh} "
          f"-> saving {'keyframes' if args.keyframes else f'every {args.step}th frame'} at {out_w}x{out_h}")

    if args.keyframes:
        frame_ids = select_keyframes(zf, meta["poses"], args.kf_move, args.kf_turn)
    else:
        frame_ids = list(range(0, n_total, args.step))
    raw = []  # (i, rgb, depth, conf, K, T_c2w_arkit_cv)
    for i in frame_ids:
        rgb = Image.open(io.BytesIO(zf.read(f"rgbd/{i}.jpg"))).convert("RGB")
        depth = np.frombuffer(decompress(zf.read(f"rgbd/{i}.depth")), np.float32).reshape(dh, dw).copy()
        conf = np.frombuffer(decompress(zf.read(f"rgbd/{i}.conf")), np.uint8).reshape(dh, dw)
        depth[~np.isfinite(depth)] = 0
        if coeffs[i] is not None:
            fx, fy, cx, cy = coeffs[i]
        else:
            fx, fy, cx, cy = K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]
        K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])
        qx, qy, qz, qw, tx, ty, tz = meta["poses"][i]
        T = np.eye(4)
        T[:3, :3] = quat_to_R(qx, qy, qz, qw)
        T[:3, 3] = [tx, ty, tz]
        raw.append((i, rgb, depth, conf, K, T @ ARKIT_TO_CV))

    # Scene frame: z-up, xy centred on the capture, floor (2nd-percentile height) at z = 0.
    # ScanNet's axis-aligned scenes look like this, and the mask decoder sees absolute xyz.
    pts = []
    for i, rgb, depth, conf, K, T in raw:
        d = np.array(Image.fromarray(depth).resize((out_w, out_h), Image.NEAREST))
        c = np.array(Image.fromarray(conf).resize((out_w, out_h), Image.NEAREST))
        pts.append(backproject(d, K, Y_UP_TO_Z_UP @ T, (c == 2) & (d > 0) & (d < args.max_depth), stride=8))
    pts = np.concatenate(pts)
    lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
    offset = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    world = np.eye(4)
    world[:3, 3] = offset
    world = world @ Y_UP_TO_Z_UP  # ARKit world -> scene

    for k, (i, rgb, depth, conf, K, T) in enumerate(raw):
        rgb.resize((out_w, out_h), Image.BILINEAR).save(out / "color" / f"{k}.jpg", quality=95)
        d = np.array(Image.fromarray(depth).resize((out_w, out_h), Image.NEAREST))
        Image.fromarray(np.clip(d * 1000, 0, 65535).astype(np.uint16)).save(out / "depth" / f"{k}.png")
        Image.fromarray(conf).resize((out_w, out_h), Image.NEAREST).save(out / "conf" / f"{k}.png")
        np.savetxt(out / "pose" / f"{k}.txt", world @ T, fmt="%.8f")
        np.savetxt(out / "intrinsic" / f"{k}.txt", K, fmt="%.6f")

    json.dump({
        "source": str(Path(args.r3d).name),
        "num_frames": len(raw),
        "source_frame_ids": frame_ids,
        "image_size_wh": [out_w, out_h],
        "fps": meta.get("fps"),
        "step": None if args.keyframes else args.step,
        "keyframes": bool(args.keyframes),
        "world_from_arkit": world.tolist(),
        "note": "scene point = world_from_arkit @ ARKit world point; poses are OpenCV camera-to-scene",
    }, open(out / "meta.json", "w"), indent=2)

    # Sanity check 1: frames agree with each other (same check we ran on the sample: ~0.5 cm).
    def load(k):
        d = np.array(Image.open(out / "depth" / f"{k}.png")).astype(np.float32) / 1000
        c = np.array(Image.open(out / "conf" / f"{k}.png"))
        return d, c, np.loadtxt(out / "intrinsic" / f"{k}.txt"), np.loadtxt(out / "pose" / f"{k}.txt")

    errs = []
    n = len(raw)
    for a, b in [(0, n // 4), (n // 4, n // 2), (n // 2, 3 * n // 4)]:
        if a == b:
            continue
        da, ca, Ka, Ta = load(a)
        db, cb, Kb, Tb = load(b)
        p = backproject(da, Ka, Ta, (ca == 2) & (da > 0), stride=4)
        pc = (np.linalg.inv(Tb) @ np.c_[p, np.ones(len(p))].T).T
        z = pc[:, 2]
        ok = z > 0.1
        u = np.round(pc[ok, 0] * Kb[0, 0] / z[ok] + Kb[0, 2]).astype(int)
        v = np.round(pc[ok, 1] * Kb[1, 1] / z[ok] + Kb[1, 2]).astype(int)
        inb = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
        e = np.abs(db[v[inb], u[inb]] - z[ok][inb])
        e = e[cb[v[inb], u[inb]] == 2]
        if len(e):
            errs.append(np.median(e))
            print(f"  consistency frame {a} -> {b}: overlap {inb.mean():.0%}, median depth error {np.median(e) * 100:.1f} cm")
    verdict = ("skipped (need at least 2 frames)" if n < 2 else
               "OK" if errs and max(errs) < 0.03 else "CHECK (poses or depth may be off)")
    print(f"frame consistency: {verdict}")

    # Sanity check 2: preview of the fused cloud in the scene frame.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cloud, cols = [], []
    for k in range(0, n, max(1, n // 12)):
        d, c, K, T = load(k)
        img = np.asarray(Image.open(out / "color" / f"{k}.jpg"))
        m = (c == 2) & (d > 0) & (d < args.max_depth)
        v_, u_ = np.nonzero(m[::4, ::4])
        cloud.append(backproject(d, K, T, m, stride=4))
        cols.append(img[v_ * 4, u_ * 4] / 255)
    cloud, cols = np.concatenate(cloud), np.concatenate(cols)
    cam = np.array([np.loadtxt(out / "pose" / f"{k}.txt")[:3, 3] for k in range(n)])
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(Image.open(out / "color" / f"{n // 2}.jpg"))
    ax[0].set_title(f"frame {n // 2} of {n}")
    ax[0].axis("off")
    ax[1].scatter(cloud[:, 0], cloud[:, 1], c=cols, s=0.3)
    ax[1].plot(cam[:, 0], cam[:, 1], "k-", lw=1)
    ax[1].plot(cam[0, 0], cam[0, 1], "go")
    ax[1].set_aspect("equal")
    ax[1].set_title("top-down (x, y), camera path in black")
    ax[2].scatter(cloud[:, 0], cloud[:, 2], c=cols, s=0.3)
    ax[2].set_aspect("equal")
    ax[2].set_title("side (x, z): floor/table should be flat and horizontal")
    plt.tight_layout()
    plt.savefig(out / "preview.png", dpi=80)
    print(f"saved {n} frames to {out}/ and preview to {out}/preview.png")


if __name__ == "__main__":
    main()
