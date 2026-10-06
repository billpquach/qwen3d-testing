"""Qwen-3D grounding on a converted Record3D scene: text query -> 3D object points.

Runs the same steps as Qwen3D.forward() in eval mode, minus the dataset/evaluator plumbing:
  1. Qwen ViT on N frames -> one feature per 28x28 patch, lifted to 3D with depth + pose,
     averaged into 5 cm voxels (the "point cloud tokens").
  2. LLM (int8) reads [point tokens + your sentence] with Qwen-3D's xyz RoPE.
  3. Point features are interpolated onto a dense LiDAR cloud (2 cm voxels; on ScanNet this
     is the mesh, here it is our fused high-confidence depth), and the mask decoder predicts
     100 candidate masks. Each candidate is scored against the target words of the sentence.
The ViT/3D step is computed once per scene and reused for every query.

Run from the qwen3d repo root, in the qwen3d env:
    python qwen3d_ground.py scenes/kitchen1 "the black mug"
    python qwen3d_ground.py scenes/kitchen1 "the mug next to the ball" --target "mug"
    python qwen3d_ground.py scenes/kitchen1            # interactive: type queries, Enter to quit
Outputs go to <scene>/results/: a PNG per query, the object's points (.ply) and a JSON summary.
"""
import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")
warnings.filterwarnings("ignore", message=".*inputs will be cast.*")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

GB = 1024 ** 3
FOUND_SCORE = 0.5   # below this the best candidate is "not found" (absent laptop scored 0.26, real objects 0.79-0.95)
RELATION_WORDS = r"\b(next to|beside|near|on top of|on|under|below|above|behind|in front of|left of|right of|" \
                 r"to the left|to the right|between|closest to|farthest from|inside|in|at|by|with|that|which)\b"


# ----------------------------------------------------------------------------- model
def load_model(quant):
    import logging

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.projects.deeplab import add_deeplab_config

    import train  # noqa: F401  registers the Qwen3D meta-architecture
    from qwen3d.config import add_maskformer2_config, add_maskformer2_video_config
    from qwen3d.data_video.data_utils import resolve_feature_dir_for_backbone

    logging.getLogger("fvcore").setLevel(logging.ERROR)
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file("qwen3d/configs/qwen_3d.yaml")
    cfg.merge_from_list([
        "QWEN_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct",
        "MODEL.WEIGHTS", "ckpts/qwen3d_3b.pth",
        "USE_WANDB", "False",
        "OUTPUT_DIR", "/tmp/qwen3d_ground",
    ])
    resolve_feature_dir_for_backbone(cfg)

    t0 = time.time()
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    if quant != "none":
        import bitsandbytes as bnb

        model.qwen_model = model.qwen_model.merge_and_unload()
        cfg.defrost()
        cfg.USE_LORA = False
        model.cfg = cfg

        def q(lin):
            w = lin.weight.data.cpu()
            b = lin.bias.data.cpu() if lin.bias is not None else None
            if quant == "int8":
                m = bnb.nn.Linear8bitLt(lin.in_features, lin.out_features, bias=b is not None,
                                        has_fp16_weights=False, threshold=6.0)
                m.weight = bnb.nn.Int8Params(w.half(), requires_grad=False, has_fp16_weights=False)
            else:
                m = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=b is not None,
                                      compute_dtype=torch.bfloat16, quant_type="nf4")
                m.weight = bnb.nn.Params4bit(w, requires_grad=False, quant_type="nf4")
            if b is not None:
                m.bias = nn.Parameter(b, requires_grad=False)
            return m.cuda()

        for layer in model.qwen_model.model.layers:
            for parent in (layer.self_attn, layer.mlp):
                for name, child in list(parent.named_children()):
                    if isinstance(child, nn.Linear):
                        setattr(parent, name, q(child))
            torch.cuda.empty_cache()
    cfg.freeze()
    torch.cuda.empty_cache()  # hand back the cache left over from the bf16 copy
    print(f"model ready ({quant}) in {time.time() - t0:.0f} s: {torch.cuda.memory_allocated() / GB:.2f} GB in tensors, "
          f"{torch.cuda.memory_reserved() / GB:.2f} GB reserved by PyTorch (nvidia-smi adds ~0.4 GB CUDA context)")
    return model, cfg


# ----------------------------------------------------------------------------- scene
def load_frame(scene, k):
    rgb = np.array(Image.open(scene / "color" / f"{k}.jpg").convert("RGB"))
    depth = np.array(Image.open(scene / "depth" / f"{k}.png")).astype(np.float32) / 1000.0
    conf = np.array(Image.open(scene / "conf" / f"{k}.png"))
    K = np.loadtxt(scene / "intrinsic" / f"{k}.txt")
    T = np.loadtxt(scene / "pose" / f"{k}.txt")
    return rgb, depth, conf, K, T


def backproject(depth, K, T, mask, stride=1):
    v, u = np.nonzero(mask[::stride, ::stride])
    v, u = v * stride, u * stride
    z = depth[v, u]
    p = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)], 1)
    return (T @ p.T).T[:, :3]


@torch.no_grad()
def prepare_scene(model, cfg, scene, n_frames, max_depth):
    """Everything that does not depend on the query: ViT features, 3D tokens, dense target cloud."""
    from scipy.spatial import cKDTree
    from torch_scatter import scatter_mean
    from transformers import AutoProcessor

    from qwen3d.data_video.data_utils import get_multiview_xyz, qwen_preprocess_frames
    from qwen3d.modeling.backproject.backproject import multiscsale_voxelize, voxelization
    from qwen3d.utils.util_3d import sample_2D_indices

    dev = model.device
    t0 = time.time()
    all_ids = sorted(int(p.stem) for p in (scene / "color").glob("*.jpg"))
    pick = sorted(set(np.linspace(0, len(all_ids) - 1, min(n_frames, len(all_ids))).round().astype(int)))
    frames = [load_frame(scene, all_ids[i]) for i in pick]
    timing = {"load": time.time() - t0}

    images = [torch.as_tensor(f[0].transpose(2, 0, 1).copy()) for f in frames]
    depths = [torch.from_numpy(f[1]) for f in frames]
    poses = [torch.from_numpy(f[4]).float() for f in frames]
    intr = [torch.from_numpy(f[3]).float() for f in frames]
    v, h, w = len(images), images[0].shape[1], images[0].shape[2]

    multi_scale_xyz, _, _, new_h, new_w = get_multiview_xyz(
        shape=(v, h, w), size_divisibility=cfg.INPUT.SIZE_DIVISIBILITY, depths=depths, poses=poses,
        intrinsics=intr, is_train=False, augment_3d=False,
        interpolation_method=cfg.MODEL.INTERPOLATION_METHOD, mask_valid=cfg.MASK_VALID,
        mean_center=cfg.MEAN_CENTER, do_rot_scale=cfg.DO_ROT_SCALE, scannet_pc=None,
        align_matrix=None, vil3d=cfg.VIL3D, scales=cfg.MULTIVIEW_XYZ_SCALES,
        min_pixel=cfg.INPUT.MIN_PIXEL, max_pixel=cfg.INPUT.MAX_PIXEL)

    timing["3d"] = time.time() - t0 - sum(timing.values())
    processor = AutoProcessor.from_pretrained(cfg.QWEN_MODEL, min_pixels=cfg.INPUT.MIN_PIXEL,
                                              max_pixels=cfg.INPUT.MAX_PIXEL).image_processor
    pv_list, grid_list = qwen_preprocess_frames(processor, images)
    timing["preprocess"] = time.time() - t0 - sum(timing.values())

    # ViT one frame at a time: identical output (attention is per image), much less memory.
    feats = []
    for pv, grid in zip(pv_list, grid_list):
        feats.append(model.visual(pv.to(dev).to(model.qwen_model.dtype), grid_thw=grid.view(1, 3).to(dev)))
    featurecloud = torch.cat(feats, 0)
    torch.cuda.synchronize()
    timing["vit"] = time.time() - t0 - sum(timing.values())

    xyz = [torch.stack([s]).to(dev) for s in multi_scale_xyz]  # [1, V, h, w, 3] per scale
    p2v = multiscsale_voxelize(xyz, cfg.INPUT.VOXEL_SIZE[::-1])
    pointcloud = xyz[1].reshape(-1, 3)
    point2voxel = p2v[1].squeeze(0)
    assert featurecloud.shape[0] == pointcloud.shape[0] == point2voxel.shape[0], (
        f"ViT tokens {featurecloud.shape[0]} != 3D points {pointcloud.shape[0]} "
        f"(image {w}x{h} -> feature grid {new_w}x{new_h})")
    featurecloud = scatter_mean(featurecloud, point2voxel, dim=0).unsqueeze(0)
    pointcloud = scatter_mean(pointcloud, point2voxel, dim=0).unsqueeze(0)
    pixel_indices = sample_2D_indices(new_h, new_w, [p2v[1][0]], [v])

    # Dense target cloud ("ghost points"): high-confidence LiDAR from every converted frame,
    # 1 cm grid, kept only where Qwen has tokens nearby.
    timing["tokens"] = time.time() - t0 - sum(timing.values())
    dense = []
    for k in all_ids:
        _, d, c, K, T = load_frame(scene, k)
        dense.append(backproject(d, K, T, (c == 2) & (d > 0) & (d < max_depth), stride=3))
    dense = np.concatenate(dense)
    g = np.floor(dense / 0.01).astype(np.int64)
    g -= g.min(0)
    span = g.max(0) + 1
    _, keep = np.unique((g[:, 0] * span[1] + g[:, 1]) * span[2] + g[:, 2], return_index=True)
    dense = dense[keep]
    tok = pointcloud[0].float().cpu().numpy()
    dist, _ = cKDTree(tok).query(dense, k=1)
    dense = dense[dist < 0.08].astype(np.float32)

    tpc = torch.from_numpy(dense).to(dev)
    t_p2v = voxelization(tpc[None].clone(), 0.02)[0]
    tpc_vox = scatter_mean(tpc[None], t_p2v[None], dim=1)
    segments = torch.arange(tpc_vox.shape[1], device=dev)[None]

    torch.cuda.synchronize()
    timing["dense"] = time.time() - t0 - sum(timing.values())
    print("prep breakdown: " + ", ".join(f"{k} {x:.1f}s" for k, x in timing.items()))
    print(f"scene: {v} frames ({w}x{h}) -> {featurecloud.shape[1]} point tokens (5 cm), "
          f"{len(dense)} dense points / {tpc_vox.shape[1]} 2 cm voxels, prepared in {time.time() - t0:.1f} s")
    return dict(featurecloud=featurecloud, pointcloud=pointcloud, pixel_indices=pixel_indices, v=v,
                dense=dense, t_p2v=t_p2v, tpc_vox=tpc_vox, segments=segments,
                frames=frames, pick=pick)


# ----------------------------------------------------------------------------- query
def default_target(query):
    """Words before the first relation word, minus a leading article: 'the mug next to the ball' -> 'mug'."""
    target = re.split(RELATION_WORDS, query, maxsplit=1, flags=re.I)[0].strip() or query
    return re.sub(r"^(the|a|an)\s+", "", target, flags=re.I) or target


def target_token_ids(tokenizer, query, target):
    """Token positions (within the sentence) of the target phrase; the decoder scores against these.
    Returns (positions, target_used, note). Scoring against the whole sentence would also reward the
    anchor ("ball" in "the mug next to the ball"), so a --target that isn't in the query falls back
    to the default target instead."""
    note = None
    if target and target.lower() not in query.lower():
        note = f"--target '{target}' is not in the query, used '{default_target(query)}' instead"
        print(f"  WARNING: {note}")
        target = None
    if not target:
        target = default_target(query)
    start = query.lower().find(target.lower())
    end = start + len(target)
    enc = tokenizer(query, return_offsets_mapping=True, add_special_tokens=False)
    ids = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < end and b > start]
    return ids, target, note


@torch.no_grad()
def ground(model, cfg, S, query, target=None):
    from qwen3d.conversation_template import TEXT_INPUT_GEN_3D
    from qwen3d.modeling_qwen2_5_vl_modified import build_full_mask

    dev = model.device
    tok = model.qwen_processor.tokenizer
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    inputs = tok([TEXT_INPUT_GEN_3D.format(target_sentence=query)], return_tensors="pt").to(dev)
    input_ids, attn = inputs["input_ids"], inputs["attention_mask"]
    embeds = model.qwen_model.get_input_embeddings()(input_ids)
    ph = model.qwen_model.config.pointcloud_token_id
    assert (input_ids == ph).sum() == 1
    idx = (input_ids == ph).int().argmax(1)[0].item()
    fc = S["featurecloud"][0].to(embeds.dtype)
    n = fc.shape[0]
    ids = torch.cat([input_ids[0, :idx], input_ids.new_full((n,), ph), input_ids[0, idx + 1:]])[None]
    pad = torch.cat([attn[0, :idx], attn.new_ones(n), attn[0, idx + 1:]])[None]
    emb = torch.cat([embeds[0, :idx], fc, embeds[0, idx + 1:]])[None]

    # Bidirectional attention over the prompt, exactly as Qwen3D.forward builds it.
    L = ids.shape[1]
    answer_start = (L - 1 - (ids.flip(1) == model.im_start_token_id).float().argmax(1)) + 3
    full = build_full_mask(pad, dev)
    causal = torch.tril(torch.ones((L, L), dtype=torch.bool, device=dev))
    a = answer_start[0].item()
    full[0, 0, :a, a:] = False
    full[0, 0, a:, a:] = causal[a:, a:]
    attn4 = torch.where(full, 0.0, torch.finfo(model.qwen_model.dtype).min)

    out = model.qwen_model(
        input_ids=ids, attention_mask=attn4, padding_mask=pad, inputs_embeds=emb,
        output_hidden_states=True, return_dict=True, rope_type=cfg.ROPE_TYPE,
        pointcloud_pixel_pos=S["pixel_indices"], pointcloud_xyz=S["pointcloud"],
        use_causal_mask=cfg.CAUSAL_MASK, generate_mode=False, answer_start=answer_start)
    hidden = out.hidden_states[-1]
    del out, full, attn4

    starts, ends = model.get_target_sentence_spans(ids)
    point_feats = model.connector(hidden[:, idx: idx + n])
    text_feats = model.text_connector(hidden[:, starts[0]: ends[0]])

    mask_feats = model.upsample_point_features(S["pointcloud"], point_feats, S["tpc_vox"], k_neighbors=8)[..., None]
    outputs = model.mask_decoder(
        mask_feats, shape=[1, S["v"]], mask_features_xyz=S["tpc_vox"], segments=S["segments"],
        decoder_3d=True, actual_decoder_3d=True, scannet_all_masks_batched=None,
        max_valid_points=[S["tpc_vox"].shape[1]], qwen_3d_text_features=text_feats)
    logits = outputs["pred_logits"][0].float().sigmoid()   # [100 queries, sentence tokens]
    masks = outputs["pred_masks"][0].float()               # [100 queries, 2 cm voxels]

    pos, target_used, target_note = target_token_ids(tok, query, target)
    n_text = ends[0].item() - starts[0].item()
    if not pos or max(pos) >= n_text:
        target_note = (target_note or '') + ' (target tokens not found, scored the whole sentence)'
        pos = list(range(n_text))
    scores = logits[:, pos].mean(-1)
    order = scores.argsort(descending=True)
    torch.cuda.synchronize()
    dt = time.time() - t0

    t_p2v = S["t_p2v"]
    cands = []
    for q in order[:5].tolist():
        vm = masks[q].sigmoid() > 0.5
        cands.append(dict(score=scores[q].item(), point_mask=vm[t_p2v].cpu().numpy()))
    return dict(query=query, target=target_used, target_note=target_note, target_tokens=[tok.decode([ids[0, starts[0] + p].item()]) for p in pos],
                cands=cands, seconds=dt, peak_gb=torch.cuda.max_memory_allocated() / GB, n_tokens=L)


# ----------------------------------------------------------------------------- output
def save_result(scene, S, R, world_from_arkit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = scene / "results"
    outdir.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", R["query"].lower()).strip("_")[:40]
    dense = S["dense"]
    best = R["cands"][0]
    pts = dense[best["point_mask"]]

    summary = dict(query=R["query"], target=R["target"], target_tokens=R["target_tokens"],
                   **({"target_note": R["target_note"]} if R.get("target_note") else {}),
                   found=bool(best["score"] >= FOUND_SCORE), score=round(best["score"], 4), top5_scores=[round(c["score"], 4) for c in R["cands"]],
                   n_points=int(len(pts)), seconds=round(R["seconds"], 2), peak_gpu_gb=round(R["peak_gb"], 2),
                   llm_tokens=R["n_tokens"])
    if len(pts):
        lo, hi = pts.min(0), pts.max(0)
        c = pts.mean(0)
        c_arkit = np.linalg.inv(world_from_arkit) @ np.r_[c, 1.0]
        summary.update(center_scene_m=np.round(c, 3).tolist(), center_arkit_m=np.round(c_arkit[:3], 3).tolist(),
                       size_xyz_m=np.round(hi - lo, 3).tolist())
        with open(outdir / f"{slug}.ply", "w") as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\n"
                    f"property float z\nend_header\n")
            np.savetxt(f, pts, fmt="%.4f")
    json.dump(summary, open(outdir / f"{slug}.json", "w"), indent=2)

    # Figure: best mask projected into 3 frames, plus top-down and side views of the scene.
    fig, ax = plt.subplots(1, 5, figsize=(26, 6))
    frames = S["frames"]
    for j, fi in enumerate(np.linspace(0, len(frames) - 1, 3).round().astype(int)):
        rgb, d, _, K, T = frames[fi]
        ax[j].imshow(rgb)
        if len(pts):
            pc = (np.linalg.inv(T) @ np.c_[pts, np.ones(len(pts))].T).T
            ok = pc[:, 2] > 0.05
            u = pc[ok, 0] * K[0, 0] / pc[ok, 2] + K[0, 2]
            v = pc[ok, 1] * K[1, 1] / pc[ok, 2] + K[1, 2]
            ax[j].scatter(u, v, s=1, c="red", alpha=0.5)
        ax[j].set_xlim(0, rgb.shape[1])
        ax[j].set_ylim(rgb.shape[0], 0)
        ax[j].axis("off")
        ax[j].set_title(f"frame {S['pick'][fi]}")
    sub = dense[:: max(1, len(dense) // 60000)]
    colors = ["red", "royalblue", "orange"]
    for k, (a, b, name) in enumerate([(0, 1, "top-down (x, y)"), (0, 2, "side (x, z)")]):
        axk = ax[3 + k]
        axk.scatter(sub[:, a], sub[:, b], s=0.2, c="0.75")
        for ci in range(min(3, len(R["cands"])) - 1, -1, -1):
            cp = dense[R["cands"][ci]["point_mask"]]
            axk.scatter(cp[:, a], cp[:, b], s=1 if ci else 2, c=colors[ci],
                        label=f"#{ci + 1} score {R['cands'][ci]['score']:.3f}")
        axk.set_aspect("equal")
        axk.set_title(name)
        axk.legend(loc="upper right", markerscale=6, fontsize=8)
    fig.suptitle(f'"{R["query"]}"  target: "{R["target"]}"  best score {best["score"]:.3f}, '
                 f'{len(pts)} points, {R["seconds"]:.1f} s', fontsize=13)
    plt.tight_layout()
    plt.savefig(outdir / f"{slug}.png", dpi=70)
    plt.close(fig)
    return summary, outdir / f"{slug}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("query", nargs="?")
    ap.add_argument("--target", help="the words naming the object you want, e.g. 'mug' in 'the mug next to the ball'")
    ap.add_argument("--frames", type=int, default=15, help="frames given to the model (more = better coverage, slower)")
    ap.add_argument("--max-depth", type=float, default=3.0, help="ignore LiDAR points farther than this (m)")
    ap.add_argument("--quant", choices=["int8", "nf4", "none"], default="int8")
    args = ap.parse_args()

    scene = Path(args.scene)
    world_from_arkit = np.array(json.load(open(scene / "meta.json"))["world_from_arkit"])
    model, cfg = load_model(args.quant)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        S = prepare_scene(model, cfg, scene, args.frames, args.max_depth)

        def run(query, target):
            R = ground(model, cfg, S, query, target)
            summary, png = save_result(scene, S, R, world_from_arkit)
            torch.cuda.empty_cache()
            print(json.dumps(summary, indent=2))
            print(f"saved {png}")

        if args.query:
            run(args.query, args.target)
            return
        print("\nType a query, optionally 'query | target words'. Empty line quits.")
        while True:
            try:
                line = input("query> ").strip()
            except EOFError:
                break
            if not line:
                break
            query, _, target = line.partition("|")
            try:
                run(query.strip(), target.strip() or None)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("out of GPU memory: try --frames 8 or --max-depth 2.0")


if __name__ == "__main__":
    main()
