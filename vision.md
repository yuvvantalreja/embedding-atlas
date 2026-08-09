# Vision features for the anchor atlas (deferred — design notes)

Status: **not implemented**. The observation embedding in `3lego_sorting.py` currently uses
proprioception + binned action history only. This document captures everything needed to add
the vision block later, matching the method of `initial_report_v6.html`.

## What the report did

The report's observation features are: **per-camera SigLIP tokens from the *base* pi0.5
checkpoint, mean+std pooled**, concatenated with proprioception and the binned 120-step
action history (per-modality z-score → weighted concat → PCA → t-SNE).

Terminology, precisely:

- **pi0.5 is a VLA** (vision-language-action model, Physical Intelligence). It is built on
  **PaliGemma** — a VLM composed of a **SigLIP-So400m/14 vision tower** (~430M params) and a
  **Gemma 2B** language model — plus a ~300M flow-matching **action expert** that generates
  continuous action chunks.
- **SigLIP is not the VLA** — it is only the vision encoder inside it (a CLIP-style,
  contrastively trained ViT from Google).
- "**base** pi0.5 SigLIP" = the *pretrained, non-fine-tuned* checkpoint's vision tower used as
  a feature extractor. Two reasons this choice is right:
  1. **Policy-independent**: the observation embedding stays fixed while different fine-tuned
     policies are compared against it.
  2. **It is the representation the policies actually see**: every fine-tune shares that tower.
- Pooling: for each camera image, take the SigLIP patch-token sequence and compute the
  **mean and std over tokens** (→ 2×1152 dims per camera for So400m), per camera, then concat
  across the 3 cameras (top, left_wrist, right_wrist). The report notes this "per-camera
  mean+std" beats naive joint mean-pooling (effective dim 46 vs 15 after their PCA).

## Encoder options (no VLA download needed)

| Option | Params | Fidelity | Notes |
|---|---|---|---|
| `google/siglip-so400m-patch14-224` (HF) | ~430M | matches pi0's tower | recommended when faithful |
| `google/siglip-base-patch16-224` (HF) | ~90M | proxy, ~6× faster | fine for atlas clustering |
| Extract tower from openpi base pi0.5 ckpt | ~430M | exact | multi-GB download, JAX weights — not worth it |

Load via `transformers` `SiglipVisionModel` (or `AutoModel` + take `vision_model`), grab
`last_hidden_state` tokens, pool mean+std. fp16 on MPS.

## Workload and M2 (8GB) feasibility

- Images needed: **anchor frames only** — `n_anchors × 3 cameras`. At `ANCHOR_STRIDE=60`
  (1 Hz): baseline ≈ 14.4k anchors → **~43k images per dataset**; double when the adversarial
  dataset lands. You never push "GBs of images" through at once — stream-decode and batch.
- Decoding: videos are AV1 mp4s at 224×224/60fps. Do **one sequential decode pass per video**
  (PyAV), collecting the anchor frame indices for that episode in one go (see
  `_decode_episode_frames` in `packages/backend/embedding_atlas/lerobot_utils.py` — it already
  implements exactly this pattern). ~30–60 min per dataset.
- Inference (use **MPS**, fp16, batch ≤16 — CPU-only is 4–6× slower, don't):
  - SigLIP-So400m: ~210 GFLOPs/img → ~5–10 img/s → **~1.5–2.5 h per dataset**
  - SigLIP-Base: ~35 GFLOPs/img → ~30–60 img/s → **~15–25 min per dataset**
- RAM: So400m fp16 ≈ 0.9GB weights + PyTorch/decode overhead — fits in 8GB if batch ≤16 and
  nothing heavy runs alongside. If memory pressure bites, use SigLIP-Base or run on a GPU box.

## Architecture: cached artifact, not a notebook step

The notebook must **never run vision inference**. Instead:

1. A standalone one-time batch script (to be written) walks the anchor manifest, decodes anchor
   frames, runs the encoder, pools per camera, and writes a parquet:
   - columns: `anchor_key` (string, `"{regime}:{episode_index}:{frame_index}"` — matches the
     column produced by `build_anchor_vectors`) + one vector column (e.g. `vision_vector`,
     float32 list, per-camera mean+std concat).
   - Output size ≈ 43k × 6912 dims ≈ 600MB fp16 (or PCA-reduce to ~256 dims first → ~50MB).
   - The script can run on this M2 (overnight-safe) or on the GPU training box — the parquet
     is the only thing that moves.
2. The notebook consumes it via the already-implemented hook:
   ```python
   anchors = build_anchor_vectors(
       DATASET_ROOTS, ...,
       extra_blocks={"vision": "features/vision_siglip.parquet"},
       block_weights={"proprio": 1.0, "history": 1.0, "vision": 1.0},
   )
   ```
   `extra_blocks` tables are joined on `anchor_key`, z-scored, energy-normalized, and
   concatenated as additional blocks of the obs vector. Anchors missing from the table raise a
   clear error (recompute the parquet after changing `ANCHOR_STRIDE` or adding a dataset).
3. Changing the obs vector changes the projection cache key — the obs UMAP legitimately
   recomputes; the trajectory side is untouched.

## Existing repo pieces to reuse when implementing

- `lerobot_utils.has_videos(root)` / `video_keys(info)` — gate on videos being downloaded
  (they are **not** present locally today; only parquet + meta).
- `lerobot_utils._decode_episode_frames(video_path, frame_indices, size)` — sequential AV1
  decode grabbing exact frame indices (needs `pip install av`).
- `projection.py` custom-embedder hook (`embedder=<async callable>`) — only relevant if you
  ever want projection-time embedding instead of the cached-parquet route; prefer the parquet.
