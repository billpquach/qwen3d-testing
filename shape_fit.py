"""Complete a one-sided object scan with a simple shape: sphere, upright cylinder or box.

Qwen-3D's mask only contains surfaces the camera saw (the volleyball is a half shell), plus a
"puddle" of table around the base. This script:
  1. removes the table layer under the object,
  2. fits a sphere, an upright cylinder and an upright box to what is left (robustly, so a mug
     handle doesn't drag the fit), and keeps the best fit (or the one the label suggests),
  3. samples the full shape and keeps only the parts far from observed points: the "inferred"
     geometry, to be drawn faint in the X-ray view. Observed points stay as measured.
Flat objects (scissors, paper) have nothing above the table, so they get a footprint only.

Input is a result from qwen3d_ground.py (scene frame: metres, z up, table horizontal).
    python shape_fit.py scenes/kitchen1/results/the_volleyball.ply
    python shape_fit.py scenes/kitchen1/results/*.ply              # all results
    python shape_fit.py results/the_mug.ply --shape cylinder       # force a shape
Writes <name>_observed.ply, <name>_inferred.ply, <name>_shape.json and <name>_shape.png.
"""
import argparse
import json
from pathlib import Path

import numpy as np

TABLE_GAP_M = 0.015      # points within this of the table surface belong to the table
FLAT_HEIGHT_M = 0.03     # objects shorter than this are treated as flat
NEAR_OBSERVED_M = 0.015  # sampled shape points closer than this to a measured point are "observed"
LABEL_HINTS = {"sphere": ("ball", "orange", "apple", "globe"),
               "cylinder": ("can", "cup", "mug", "bottle", "jar", "glass", "tube", "candle"),
               "box": ("box", "book", "carton", "case", "phone", "laptop", "remote")}


def read_ply(path):
    lines = Path(path).read_text().splitlines()
    end = next(i for i, l in enumerate(lines) if l.strip() == "end_header")
    return np.loadtxt(lines[end + 1:], ndmin=2)[:, :3]


def write_ply(path, pts):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\n"
                f"property float z\nend_header\n")
        np.savetxt(f, pts, fmt="%.4f")


# ----------------------------------------------------------------------------- fits
def fit_sphere(p):
    A = np.c_[2 * p, np.ones(len(p))]
    sol, *_ = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)
    c = sol[:3]
    r = np.sqrt(max(sol[3] + c @ c, 1e-8))
    return {"center": c, "radius": r}, np.abs(np.linalg.norm(p - c, axis=1) - r)


def fit_cylinder(p, z0):
    xy = p[:, :2]
    A = np.c_[2 * xy, np.ones(len(xy))]
    sol, *_ = np.linalg.lstsq(A, (xy ** 2).sum(1), rcond=None)
    c = sol[:2]
    r = np.sqrt(max(sol[2] + c @ c, 1e-8))
    return {"center_xy": c, "radius": r, "z0": z0, "z1": np.percentile(p[:, 2], 98)}, np.abs(np.linalg.norm(xy - c, axis=1) - r)


def fit_box(p, z0):
    xy = p[:, :2]
    m = xy.mean(0)
    _, _, vt = np.linalg.svd(xy - m, full_matrices=False)
    local = (xy - m) @ vt.T
    lo, hi = local.min(0), local.max(0)
    # distance of each point to the nearest side face (top face handled by z)
    d_side = np.minimum(np.abs(local - lo), np.abs(local - hi)).min(1)
    d_top = np.abs(p[:, 2] - p[:, 2].max())
    return {"center_xy": m + ((lo + hi) / 2) @ vt, "axes": vt, "half": (hi - lo) / 2, "z0": z0,
            "z1": np.percentile(p[:, 2], 98)}, np.minimum(d_side, d_top)


def robust(fit, p, *extra, iters=2, keep_m=0.02):
    """Fit, drop points far from the shape (handles, stray table), refit."""
    params, _ = fit(p, *extra)
    for _ in range(iters):
        res = residuals(params, p)
        inl = res < max(keep_m, 2.5 * np.median(res))
        if inl.sum() < 20:
            break
        params, _ = fit(p[inl], *extra)
    return params, residuals(params, p)


def residuals(params, p):
    if "radius" in params and "center" in params:
        return np.abs(np.linalg.norm(p - params["center"], axis=1) - params["radius"])
    if "radius" in params:
        return np.abs(np.linalg.norm(p[:, :2] - params["center_xy"], axis=1) - params["radius"])
    local = (p[:, :2] - params["center_xy"]) @ params["axes"].T
    d_side = np.abs(np.abs(local) - params["half"]).min(1)
    return np.minimum(d_side, np.abs(p[:, 2] - params["z1"]))


# ----------------------------------------------------------------------------- sampling
def sample_shape(kind, q, z0, step=0.008):
    if kind == "sphere":
        c, r = q["center"], q["radius"]
        n = max(200, int(4 * np.pi * r * r / step ** 2))
        k = np.arange(n) + 0.5
        phi, th = np.arccos(1 - 2 * k / n), np.pi * (1 + 5 ** 0.5) * k
        pts = c + r * np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]
        return pts[pts[:, 2] >= z0]
    if kind == "cylinder":
        (cx, cy), r, z1 = q["center_xy"], q["radius"], q["z1"]
        a = np.arange(0, 2 * np.pi, step / r)
        zs = np.arange(z0, z1 + 1e-6, step)
        side = np.array([[cx + r * np.cos(t), cy + r * np.sin(t), z] for z in zs for t in a])
        rr = np.arange(step, r, step)
        top = np.array([[cx + s * np.cos(t), cy + s * np.sin(t), z1] for s in rr for t in np.arange(0, 2 * np.pi, step / s)])
        return np.vstack([side, top]) if len(top) else side
    # box
    hx, hy = q["half"]
    u = np.arange(-hx, hx + 1e-6, step)
    v = np.arange(-hy, hy + 1e-6, step)
    zs = np.arange(z0, q["z1"] + 1e-6, step)
    faces = [np.array([[x, s * hy, z] for x in u for z in zs]) for s in (-1, 1)]
    faces += [np.array([[s * hx, y, z] for y in v for z in zs]) for s in (-1, 1)]
    faces.append(np.array([[x, y, q["z1"]] for x in u for y in v]))
    loc = np.vstack(faces)
    xy = loc[:, :2] @ q["axes"] + q["center_xy"]
    return np.c_[xy, loc[:, 2]]


def describe(kind, q):
    if kind == "sphere":
        return {"center_m": q["center"].round(3).tolist(), "diameter_m": round(float(2 * q["radius"]), 3)}
    if kind == "cylinder":
        return {"center_xy_m": q["center_xy"].round(3).tolist(), "diameter_m": round(float(2 * q["radius"]), 3),
                "height_m": round(float(q["z1"] - q["z0"]), 3)}
    return {"center_xy_m": q["center_xy"].round(3).tolist(), "size_m": (2 * q["half"]).round(3).tolist(),
            "height_m": round(float(q["z1"] - q["z0"]), 3)}


# ----------------------------------------------------------------------------- main
def process(path, force=None):
    from scipy.spatial import cKDTree

    path = Path(path)
    pts = read_ply(path)
    label = ""
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        label = (meta.get("target") or meta.get("query") or "").lower()

    # 1. table layer: the densest thin slab at the bottom of the mask
    z = pts[:, 2]
    lo = np.percentile(z, 3)
    hist, edges = np.histogram(z[z < lo + 0.05], bins=20)
    table_z = edges[np.argmax(hist)] + (edges[1] - edges[0]) / 2
    obj = pts[z > table_z + TABLE_GAP_M]
    height = obj[:, 2].max() - table_z if len(obj) else 0.0
    out = {"source": path.name, "label": label, "n_mask_points": int(len(pts)), "table_z_m": round(float(table_z), 3)}

    if len(obj) < 30 or height < FLAT_HEIGHT_M:
        xy = pts[:, :2]
        out.update(shape="flat", note="nothing measurable above the table; footprint only",
                   footprint_center_m=xy.mean(0).round(3).tolist(),
                   footprint_size_m=(np.percentile(xy, 95, 0) - np.percentile(xy, 5, 0)).round(3).tolist())
        observed, inferred = pts, np.zeros((0, 3))
    else:
        fits = {"sphere": robust(fit_sphere, obj), "cylinder": robust(fit_cylinder, obj, table_z),
                "box": robust(fit_box, obj, table_z)}
        scale = max(height, 0.05)
        scores = {k: float(np.median(r) / scale) for k, (q, r) in fits.items()}
        hint = next((k for k, words in LABEL_HINTS.items() if any(w in label for w in words)), None)
        kind = force or (hint if hint and scores[hint] < 2 * min(scores.values()) else min(scores, key=scores.get))
        q, res = fits[kind]
        shape_pts = sample_shape(kind, q, table_z)
        d, _ = cKDTree(obj).query(shape_pts, k=1)
        inferred = shape_pts[d > NEAR_OBSERVED_M]
        observed = obj
        out.update(shape=kind, chosen_by="forced" if force else ("label" if kind == hint else "best fit"),
                   fit_error_median_m=round(float(np.median(res)), 4),
                   fit_scores={k: round(v, 3) for k, v in scores.items()},
                   object_height_m=round(float(height), 3), n_observed=int(len(observed)), n_inferred=int(len(inferred)),
                   **describe(kind, q))

    stem = path.with_suffix("")
    write_ply(f"{stem}_observed.ply", observed)
    write_ply(f"{stem}_inferred.ply", inferred)
    Path(f"{stem}_shape.json").write_text(json.dumps(out, indent=2))
    plot(stem, observed, inferred, pts, out)
    print(json.dumps(out))
    return out


def plot(stem, observed, inferred, raw, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    views = [(0, 1, "top (x, y)"), (0, 2, "side (x, z)"), (1, 2, "side (y, z)")]
    for k, (a, b, name) in enumerate(views):
        ax[k].scatter(raw[:, a], raw[:, b], s=1, c="0.85", label="Qwen-3D mask")
        if len(inferred):
            ax[k].scatter(inferred[:, a], inferred[:, b], s=1, c="tab:orange", alpha=0.25, label="inferred")
        ax[k].scatter(observed[:, a], observed[:, b], s=2, c="tab:blue", label="observed")
        ax[k].set_aspect("equal")
        ax[k].set_title(name)
    ax[0].legend(markerscale=6, fontsize=8)
    size = {k: v for k, v in out.items() if k.endswith("_m") and k not in ("table_z_m",)}
    fig.suptitle(f"{out['source']}: {out['shape']}  {size}", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{stem}_shape.png", dpi=70)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", nargs="+")
    ap.add_argument("--shape", choices=["sphere", "cylinder", "box"])
    args = ap.parse_args()
    for p in args.ply:
        if p.endswith(("_observed.ply", "_inferred.ply")):
            continue
        process(p, args.shape)
