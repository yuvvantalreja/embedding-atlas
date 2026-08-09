"""
Utilities for loading and enriching LeRobot datasets for Embedding Atlas.

LeRobot (https://github.com/huggingface/lerobot) is the de-facto standard
format for robot-learning demonstration data. A v2.x dataset looks like::

    my_dataset/
        meta/
            info.json            # features, shapes, fps, path templates
            episodes.jsonl       # per-episode task list + length
            tasks.jsonl          # task_index -> task string
            episodes_stats.jsonl # per-episode feature statistics
            <custom>.jsonl       # optional extra per-episode metadata
        data/chunk-000/
            episode_000000.parquet   # per-frame state/action/index columns
        videos/chunk-000/<camera>/
            episode_000000.mp4       # optional camera streams

This module turns that layout into DataFrames ready for
:func:`embedding_atlas.projection.compute_projection` and
:class:`embedding_atlas.widget.EmbeddingAtlasWidget`:

- :func:`load_lerobot_frames` — frame-level table with task strings and
  episode metadata joined in, with optional temporal subsampling (``stride``).
- :func:`add_episode_progress` / :func:`add_kinematic_features` — derived
  columns (progress, phase, joint speeds, active arm, gripper state,
  action-tracking error) that make the atlas filterable in interesting ways.
- :func:`build_frame_vectors` — per-frame vectors for UMAP, with optional
  temporal windowing so the embedding captures motion, not just pose.
- :func:`build_episode_vectors` — one fixed-length trajectory signature per
  episode for an episode-level atlas.
- :func:`add_video_frame_column` — decode per-frame images from the dataset's
  videos (when present) for tooltip display.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RESERVED_META_FILES = {
    "info.json",
    "stats.json",
    "tasks.jsonl",
    "episodes.jsonl",
    "episodes_stats.jsonl",
}


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------


def load_lerobot_info(root: str | Path) -> dict:
    """Load ``meta/info.json`` (features, shapes, fps, path templates)."""
    root = Path(root)
    with open(root / "meta" / "info.json") as f:
        info = json.load(f)
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v2"):
        logger.warning(
            "Dataset codebase_version is %r; these helpers were written for "
            "v2.x layouts. Loading may fail if the path templates differ.",
            version,
        )
    return info


def load_lerobot_tasks(root: str | Path) -> pd.DataFrame:
    """Load ``meta/tasks.jsonl`` as a DataFrame with task_index / task columns."""
    return pd.read_json(Path(root) / "meta" / "tasks.jsonl", lines=True)


def feature_names(info: dict, feature: str) -> list[str] | None:
    """
    Return the flat list of dimension names for *feature* from info.json.

    LeRobot stores ``names`` in several shapes (flat list, nested list,
    axis dict); this normalizes all of them to a flat list of strings.
    """
    spec = info.get("features", {}).get(feature)
    if spec is None:
        return None
    names = spec.get("names")
    if names is None:
        return None
    if isinstance(names, dict):
        # e.g. {"motors": [...]} or {"axes": [...]}
        names = next(iter(names.values()))
    flat: list[str] = []
    stack = list(names)
    while stack:
        item = stack.pop(0)
        if isinstance(item, (list, tuple)):
            stack = list(item) + stack
        else:
            flat.append(str(item))
    return flat


def load_lerobot_episodes(root: str | Path) -> pd.DataFrame:
    """
    Build a one-row-per-episode metadata table.

    Combines ``meta/episodes.jsonl`` (tasks + length) with every other
    per-episode jsonl file in ``meta/`` that has an ``episode_index`` column
    (LeRobot datasets often ship custom metadata like task hierarchies this
    way). List-valued task fields are flattened to strings so they work as
    categorical columns in the atlas.

    Adds ``episode_length`` (frames) and ``duration_s`` (from the dataset fps).
    """
    root = Path(root)
    info = load_lerobot_info(root)

    episodes = pd.read_json(root / "meta" / "episodes.jsonl", lines=True)
    if "tasks" in episodes.columns:
        episodes["episode_tasks"] = episodes["tasks"].apply(
            lambda t: " | ".join(t) if isinstance(t, (list, tuple)) else str(t)
        )
        episodes = episodes.drop(columns=["tasks"])
    if "length" in episodes.columns:
        episodes = episodes.rename(columns={"length": "episode_length"})
        fps = info.get("fps")
        if fps:
            episodes["duration_s"] = episodes["episode_length"] / float(fps)

    # Merge any custom per-episode metadata files (e.g. high_level.jsonl)
    for meta_file in sorted((root / "meta").glob("*.jsonl")):
        if meta_file.name in RESERVED_META_FILES:
            continue
        try:
            extra = pd.read_json(meta_file, lines=True)
        except ValueError as e:
            logger.warning("Skipping unreadable meta file %s: %s", meta_file.name, e)
            continue
        if "episode_index" not in extra.columns:
            logger.info(
                "Skipping %s: no episode_index column to merge on.", meta_file.name
            )
            continue
        overlap = (set(extra.columns) & set(episodes.columns)) - {"episode_index"}
        if overlap:
            extra = extra.drop(columns=list(overlap))
        episodes = episodes.merge(extra, on="episode_index", how="left")
        logger.info(
            "Merged episode metadata from %s: %s",
            meta_file.name,
            [c for c in extra.columns if c != "episode_index"],
        )

    return episodes


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------


def _episode_parquet_path(root: Path, info: dict, episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    template = info["data_path"]
    return root / template.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def load_lerobot_frames(
    root: str | Path,
    *,
    episodes: Iterable[int] | None = None,
    stride: int = 1,
    columns: list[str] | None = None,
    with_episode_metadata: bool = True,
) -> pd.DataFrame:
    """
    Load frame-level data from a LeRobot dataset into one DataFrame.

    Parameters
    ----------
    root:
        Dataset root directory (the one containing ``meta/`` and ``data/``).
    episodes:
        Episode indices to load. ``None`` loads every episode.
    stride:
        Keep every ``stride``-th frame of each episode. High-frequency
        datasets (30–60 fps) have heavily redundant consecutive frames;
        a stride of 15–30 keeps the structure while making UMAP tractable.
    columns:
        Parquet columns to read (``None`` reads all non-video columns).
    with_episode_metadata:
        Join the task string (via ``task_index``) and the per-episode
        metadata table from :func:`load_lerobot_episodes` onto each frame.

    Returns a DataFrame ordered by (episode_index, frame_index).
    """
    root = Path(root)
    info = load_lerobot_info(root)

    if episodes is None:
        episodes = range(int(info["total_episodes"]))
    episodes = sorted(set(int(e) for e in episodes))

    parts: list[pd.DataFrame] = []
    missing = 0
    for ep in episodes:
        path = _episode_parquet_path(root, info, ep)
        if not path.exists():
            missing += 1
            continue
        ep_df = pd.read_parquet(path, columns=columns)
        if stride > 1:
            ep_df = ep_df.iloc[::stride]
        parts.append(ep_df)
    if missing:
        logger.warning("%d episode parquet files were missing; skipped.", missing)
    if not parts:
        raise FileNotFoundError(f"No episode data found under {root / 'data'}")

    df = pd.concat(parts, ignore_index=True)

    if with_episode_metadata:
        if "task_index" in df.columns:
            tasks = load_lerobot_tasks(root)
            task_map = dict(zip(tasks["task_index"], tasks["task"]))
            df["task"] = df["task_index"].map(task_map)
        episode_meta = load_lerobot_episodes(root)
        overlap = (set(episode_meta.columns) & set(df.columns)) - {"episode_index"}
        if overlap:
            episode_meta = episode_meta.drop(columns=list(overlap))
        df = df.merge(episode_meta, on="episode_index", how="left")

    logger.info(
        "Loaded %d frames from %d episodes (stride=%d, fps=%s)",
        len(df),
        len(parts),
        stride,
        info.get("fps"),
    )
    return df


def load_lerobot_regimes(
    roots: dict[str, str | Path],
    *,
    episodes: Iterable[int] | None = None,
    stride: int = 1,
    columns: list[str] | None = None,
    with_episode_metadata: bool = True,
    regime_col: str = "regime",
) -> pd.DataFrame:
    """
    Load frames from multiple LeRobot dataset roots into one DataFrame.

    Each entry of *roots* maps a regime name (e.g. ``"adversarial"``,
    ``"baseline"``) to a dataset root. A *regime_col* column tags every frame,
    and ``episode_id`` (``"{regime}:{episode_index}"``) is globally unique
    across datasets. Roots that don't exist yet are skipped with a warning so
    a config can list datasets that arrive later.
    """
    parts: list[pd.DataFrame] = []
    for regime, root in roots.items():
        root = Path(root)
        if not (root / "meta" / "info.json").exists():
            logger.warning("Regime %r: no dataset at %s — skipping.", regime, root)
            continue
        df = load_lerobot_frames(
            root,
            episodes=episodes,
            stride=stride,
            columns=columns,
            with_episode_metadata=with_episode_metadata,
        )
        df[regime_col] = regime
        df["episode_id"] = f"{regime}:" + df["episode_index"].astype(str)
        parts.append(df)
    if not parts:
        raise FileNotFoundError(
            f"No LeRobot dataset found under any of: {[str(r) for r in roots.values()]}"
        )
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------


def unpack_vector_column(
    df: pd.DataFrame,
    column: str,
    names: list[str] | None = None,
    prefix: str | None = None,
) -> pd.DataFrame:
    """
    Expand a vector-valued column into one scalar column per dimension.

    Vectorized version of :func:`embedding_atlas.rl_utils.unpack_state_components`
    suitable for large frame tables. Column names come from *names* (e.g. from
    :func:`feature_names`) or default to ``{column}_{i}``.
    """
    M = np.vstack(df[column].to_numpy())
    if names is None:
        names = [f"{column}_{i}" for i in range(M.shape[1])]
    if len(names) != M.shape[1]:
        raise ValueError(
            f"Got {len(names)} names for {M.shape[1]}-dimensional column {column!r}"
        )
    for i, name in enumerate(names):
        out = f"{prefix}{name}" if prefix else name
        df[out] = M[:, i]
    return df


def add_episode_progress(
    df: pd.DataFrame,
    *,
    episode_col: str = "episode_index",
    frame_col: str = "frame_index",
) -> pd.DataFrame:
    """
    Add ``progress`` (0–1 within episode) and ``episode_phase``
    (start / middle / end) columns.

    Uses the ``episode_length`` column when present (merged from
    ``episodes.jsonl``, so it is correct even for strided frames);
    otherwise falls back to the max frame index seen per episode.
    """
    if "episode_length" in df.columns:
        length = df["episode_length"].astype(float)
    else:
        length = df.groupby(episode_col)[frame_col].transform("max").astype(float) + 1
        df["episode_length"] = length.astype(int)

    df["progress"] = (df[frame_col] / (length - 1).clip(lower=1)).clip(0.0, 1.0)
    df["episode_phase"] = pd.cut(
        df["progress"],
        bins=[-0.001, 0.15, 0.85, 1.001],
        labels=["start", "middle", "end"],
    ).astype(str)
    return df


def _kinematic_groups(names: list[str]) -> dict[str, list[int]]:
    """Split state dimensions into arm-joint / gripper / left / right groups."""
    lower = [n.lower() for n in names]
    grippers = [i for i, n in enumerate(lower) if "gripper" in n]
    joints = [i for i in range(len(names)) if i not in grippers]
    left = [i for i in joints if lower[i].startswith("left")]
    right = [i for i in joints if lower[i].startswith("right")]
    return {"joints": joints, "grippers": grippers, "left": left, "right": right}


def add_kinematic_features(
    df: pd.DataFrame,
    *,
    info: dict | None = None,
    state_names: list[str] | None = None,
    state_col: str = "observation.state",
    action_col: str | None = "action",
    time_col: str = "timestamp",
    episode_col: str = "episode_index",
) -> pd.DataFrame:
    """
    Add motion-derived columns computed from the proprioceptive state.

    Columns added (subject to what the dataset provides):

    - ``joint_speed`` — L2 norm of joint velocities (rad/s), finite-differenced
      within each episode using the timestamp column, so it stays correct
      under frame striding.
    - ``left_arm_speed`` / ``right_arm_speed`` / ``active_arm`` — for bimanual
      robots whose dimension names carry ``left_``/``right_`` prefixes.
      ``active_arm`` is one of ``both`` / ``left`` / ``right`` / ``idle``.
    - ``<gripper>_pos`` — each gripper dimension normalized to [0, 1] by its
      observed range, plus a ``<gripper>_state`` category (below 0.3 / above
      0.7 of range → the two extremes; note that which end means "open"
      depends on the robot's convention).
    - ``tracking_error`` — mean |action − state| over arm joints. For
      position-controlled arms this is the servo lag: spikes mark contact
      events, aggressive corrections, or teleop jerks.

    Dimension names are taken from *state_names*, or from *info* (the parsed
    info.json) when given.
    """
    if state_names is None and info is not None:
        state_names = feature_names(info, state_col)

    X = np.vstack(df[state_col].to_numpy()).astype(np.float32)
    n, dims = X.shape
    if state_names is None:
        state_names = [f"dim_{i}" for i in range(dims)]
    groups = _kinematic_groups(state_names)

    # Per-episode velocities via finite differences on the timestamp axis.
    t = df[time_col].to_numpy(dtype=np.float64)
    vel = np.zeros_like(X)
    positions = df.reset_index(drop=True).groupby(episode_col, sort=False).indices
    for ep_positions in positions.values():
        pos = np.asarray(ep_positions)
        if len(pos) < 2:
            continue
        tt = t[pos]
        if np.any(np.diff(tt) <= 0):
            tt = np.arange(len(pos), dtype=np.float64)
        vel[pos] = np.gradient(X[pos], tt, axis=0)

    joints = groups["joints"] or list(range(dims))
    df["joint_speed"] = np.linalg.norm(vel[:, joints], axis=1)

    if groups["left"] and groups["right"]:
        left_speed = np.linalg.norm(vel[:, groups["left"]], axis=1)
        right_speed = np.linalg.norm(vel[:, groups["right"]], axis=1)
        df["left_arm_speed"] = left_speed
        df["right_arm_speed"] = right_speed

        combined = np.concatenate([left_speed, right_speed])
        moving = combined[combined > 0]
        thresh = 0.1 * np.percentile(moving, 95) if len(moving) else 1e-6
        l_active = left_speed > thresh
        r_active = right_speed > thresh
        active = np.where(
            l_active & r_active,
            "both",
            np.where(l_active, "left", np.where(r_active, "right", "idle")),
        )
        df["active_arm"] = active

    for gi in groups["grippers"]:
        name = state_names[gi]
        g = X[:, gi]
        lo, hi = float(g.min()), float(g.max())
        span = (hi - lo) or 1.0
        g_pos = (g - lo) / span
        df[f"{name}_pos"] = g_pos
        df[f"{name}_state"] = pd.cut(
            g_pos,
            bins=[-0.001, 0.3, 0.7, 1.001],
            labels=["low", "mid", "high"],
        ).astype(str)

    if action_col and action_col in df.columns:
        A = np.vstack(df[action_col].to_numpy()).astype(np.float32)
        if A.shape == X.shape:
            df["tracking_error"] = np.abs(A - X)[:, joints].mean(axis=1)
        else:
            logger.info(
                "Skipping tracking_error: action shape %s != state shape %s",
                A.shape,
                X.shape,
            )

    return df


def format_frame_description(
    df: pd.DataFrame,
    *,
    task_col: str = "task",
    episode_col: str = "episode_index",
) -> pd.Series:
    """
    Build a concise per-frame text string for the widget ``text`` column,
    e.g. ``"place red block in red tray | ep 12 @ 34% | arm: left"``.

    Uses whichever of the enrichment columns are present.
    """

    def _fmt(row) -> str:
        parts = []
        if task_col in row.index and isinstance(row[task_col], str):
            parts.append(row[task_col])
        loc = f"ep {row[episode_col]}"
        if "progress" in row.index:
            loc += f" @ {row['progress']:.0%}"
        parts.append(loc)
        if "active_arm" in row.index:
            parts.append(f"arm: {row['active_arm']}")
        if "joint_speed" in row.index:
            parts.append(f"speed: {row['joint_speed']:.2f}")
        return " | ".join(parts)

    return df.apply(_fmt, axis=1)


# ---------------------------------------------------------------------------
# Vector building for projection
# ---------------------------------------------------------------------------


def _stack_columns(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    mats = [np.vstack(df[c].to_numpy()).astype(np.float32) for c in columns]
    return np.concatenate(mats, axis=1)


def _zscore(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    return (X - mean) / std


def build_frame_vectors(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] = ("observation.state", "action"),
    window: int = 0,
    normalize: bool = True,
    episode_col: str = "episode_index",
    output_column: str = "__lerobot_vector__",
) -> tuple[pd.DataFrame, str]:
    """
    Build one projection vector per frame from vector-valued columns.

    Parameters
    ----------
    columns:
        Vector columns to concatenate per frame. Typical choices:
        ``("observation.state",)`` for pure pose space, or
        ``("observation.state", "action")`` to also separate frames by
        commanded intent.
    window:
        Temporal context: stack each frame with its ``window`` predecessors
        and successors (clamped at episode boundaries), so the embedding
        clusters *motion snippets* rather than static poses. A window of
        2–3 on ~3 Hz strided data works well for manipulation primitives
        (reach / grasp / transport / release).
    normalize:
        Z-score each dimension before windowing. Strongly recommended:
        joint ranges differ wildly and grippers would otherwise dominate
        or vanish.

    Returns ``(df, output_column)`` with the vector column added, matching
    the convention of :func:`embedding_atlas.rl_utils.prepare_rl_data_for_projection`.
    """
    X = _stack_columns(df, columns)
    if normalize:
        X = _zscore(X)

    if window > 0:
        d = X.shape[1]
        offsets = list(range(-window, window + 1))
        out = np.empty((len(df), d * len(offsets)), dtype=np.float32)
        positions = df.reset_index(drop=True).groupby(episode_col, sort=False).indices
        for ep_positions in positions.values():
            pos = np.asarray(ep_positions)
            local = np.arange(len(pos))
            for k, off in enumerate(offsets):
                src = pos[np.clip(local + off, 0, len(pos) - 1)]
                out[pos, k * d : (k + 1) * d] = X[src]
        X = out

    df[output_column] = list(X)
    logger.info(
        "Built frame vectors: %d frames x %d dims (columns=%s, window=%d)",
        X.shape[0],
        X.shape[1],
        list(columns),
        window,
    )
    return df, output_column


def build_episode_vectors(
    frames: pd.DataFrame,
    *,
    columns: Sequence[str] = ("observation.state",),
    n_samples: int = 32,
    normalize: bool = True,
    episode_col: str = "episode_index",
    output_column: str = "__lerobot_episode_vector__",
) -> tuple[pd.DataFrame, str]:
    """
    Build one fixed-length trajectory signature per episode.

    Each episode's state trajectory is linearly resampled to *n_samples*
    timesteps and flattened, giving a ``(n_samples * dims)`` vector that
    captures the whole motion regardless of episode length. Projecting these
    gives an episode-level atlas where demonstrations of the same behavior
    cluster together and outlier demonstrations stand out.

    Returns ``(episode_df, output_column)`` where *episode_df* has one row
    per episode (just ``episode_col`` + the vector; merge your episode
    metadata onto it).
    """
    per_dim = _stack_columns(frames, columns)
    if normalize:
        per_dim = _zscore(per_dim)
    d = per_dim.shape[1]

    target = np.linspace(0.0, 1.0, n_samples)
    ep_indices: list[int] = []
    signatures: list[np.ndarray] = []
    positions = frames.reset_index(drop=True).groupby(episode_col, sort=True).indices
    for ep, ep_positions in positions.items():
        pos = np.asarray(ep_positions)
        M = per_dim[pos]
        if len(pos) < 2:
            sig = np.repeat(M, n_samples, axis=0)
        else:
            src = np.linspace(0.0, 1.0, len(pos))
            sig = np.empty((n_samples, d), dtype=np.float32)
            for j in range(d):
                sig[:, j] = np.interp(target, src, M[:, j])
        ep_indices.append(ep)
        signatures.append(sig.flatten())

    episode_df = pd.DataFrame({episode_col: ep_indices})
    episode_df[output_column] = signatures
    logger.info(
        "Built episode vectors: %d episodes x %d dims (%d samples x %d state dims)",
        len(episode_df),
        n_samples * d,
        n_samples,
        d,
    )
    return episode_df, output_column


# ---------------------------------------------------------------------------
# Anchor pipeline: linked observation <-> future-trajectory atlas
# ---------------------------------------------------------------------------


def _bin_windows(gathered: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean-pool (n_anchors, window, dims) time windows into (n_anchors, n_bins * dims)."""
    n_a, w, d = gathered.shape
    return gathered.reshape(n_a, n_bins, w // n_bins, d).mean(axis=2).reshape(n_a, n_bins * d)


def _as_matrix(vectors) -> np.ndarray:
    if isinstance(vectors, np.ndarray) and vectors.ndim == 2:
        return vectors.astype(np.float32, copy=False)
    return np.vstack([np.asarray(v, dtype=np.float32) for v in vectors])


def build_anchor_vectors(
    roots: dict[str, str | Path],
    *,
    anchor_stride: int = 60,
    history_steps: int = 120,
    horizon: int = 180,
    history_bins: int = 12,
    horizon_bins: int = 12,
    block_weights: dict[str, float] | None = None,
    extra_blocks: dict[str, Any] | None = None,
    state_col: str = "observation.state",
    action_col: str = "action",
    episodes: Iterable[int] | None = None,
    regime_col: str = "regime",
    obs_output: str = "obs_vector",
    traj_output: str = "traj_vector",
) -> pd.DataFrame:
    """
    Build the canonical anchor-frame table for a linked obs <-> trajectory atlas.

    Every ``anchor_stride``-th frame of every episode becomes an anchor. Each
    anchor yields two vectors (the method of the YAM report):

    - *obs_output*: what a policy conditions on — proprioception at the anchor
      plus the past ``history_steps`` actions mean-pooled into ``history_bins``
      time bins (repeat-padded before episode start). Blocks are z-scored
      per dimension (globally, pooled across regimes so regimes stay
      comparable), scaled by ``1/sqrt(dim)`` so no block dominates by width,
      and weighted by *block_weights* (name -> weight, default 1.0).
    - *traj_output*: the action taken — the next ``horizon`` actions
      mean-pooled into ``horizon_bins`` bins (clamp-padded with the final
      action at episode end; ``future_padded_frac`` records how much of the
      window was padded so heavily-padded anchors can be filtered).

    *extra_blocks* merges precomputed per-anchor features (e.g. vision
    embeddings, see vision.md) as additional obs blocks: each value is a
    DataFrame or parquet path with an ``anchor_key`` column plus exactly one
    vector column.

    Episodes are processed one at a time at full frame rate, so memory stays
    flat regardless of dataset size. Missing roots are skipped with a warning.

    Returns one row per anchor with metadata (regime, episode_id, task,
    subtask & other per-episode meta, progress, anchor_key) and a positional
    ``anchor_id`` int column suitable for the widget's ``row_id``.
    """
    if history_steps % history_bins != 0:
        raise ValueError(
            f"history_steps ({history_steps}) must be divisible by history_bins ({history_bins})"
        )
    if horizon % horizon_bins != 0:
        raise ValueError(
            f"horizon ({horizon}) must be divisible by horizon_bins ({horizon_bins})"
        )

    weights = {"proprio": 1.0, "history": 1.0}
    if block_weights:
        weights.update(block_weights)

    proprio_parts: list[np.ndarray] = []
    hist_parts: list[np.ndarray] = []
    fut_parts: list[np.ndarray] = []
    meta_parts: list[pd.DataFrame] = []
    episode_meta_parts: list[pd.DataFrame] = []

    for regime, root in roots.items():
        root = Path(root)
        if not (root / "meta" / "info.json").exists():
            logger.warning("Regime %r: no dataset at %s — skipping.", regime, root)
            continue
        info = load_lerobot_info(root)
        tasks = load_lerobot_tasks(root)
        task_map = dict(zip(tasks["task_index"], tasks["task"]))
        ep_meta = load_lerobot_episodes(root)
        ep_meta[regime_col] = regime
        episode_meta_parts.append(ep_meta)

        if episodes is None:
            ep_indices: Iterable[int] = range(int(info["total_episodes"]))
        else:
            ep_indices = sorted(set(int(e) for e in episodes))

        n_eps = 0
        for ep in ep_indices:
            path = _episode_parquet_path(root, info, ep)
            if not path.exists():
                continue
            ep_df = pd.read_parquet(
                path,
                columns=[state_col, action_col, "timestamp", "frame_index", "task_index"],
            )
            length = len(ep_df)
            if length == 0:
                continue
            S = np.vstack(ep_df[state_col].to_numpy()).astype(np.float32)
            A = np.vstack(ep_df[action_col].to_numpy()).astype(np.float32)
            a = np.arange(0, length, anchor_stride)
            idx_h = np.clip(a[:, None] + np.arange(-history_steps + 1, 1)[None, :], 0, length - 1)
            idx_f = np.clip(a[:, None] + np.arange(1, horizon + 1)[None, :], 0, length - 1)
            proprio_parts.append(S[a])
            hist_parts.append(_bin_windows(A[idx_h], history_bins))
            fut_parts.append(_bin_windows(A[idx_f], horizon_bins))
            task_idx = ep_df["task_index"].to_numpy()[a]
            meta_parts.append(
                pd.DataFrame(
                    {
                        regime_col: regime,
                        "episode_index": ep,
                        "frame_index": ep_df["frame_index"].to_numpy()[a],
                        "timestamp": ep_df["timestamp"].to_numpy()[a],
                        "task_index": task_idx,
                        "task": [task_map.get(int(t)) for t in task_idx],
                        "future_padded_frac": np.clip(a + horizon - (length - 1), 0, None)
                        / horizon,
                    }
                )
            )
            n_eps += 1
        logger.info("Regime %r: anchored %d episodes.", regime, n_eps)

    if not meta_parts:
        raise FileNotFoundError(
            f"No episode data found under any of: {[str(r) for r in roots.values()]}"
        )

    anchors = pd.concat(meta_parts, ignore_index=True)
    anchors["episode_id"] = anchors[regime_col] + ":" + anchors["episode_index"].astype(str)
    anchors["anchor_key"] = anchors["episode_id"] + ":" + anchors["frame_index"].astype(str)

    episode_meta = pd.concat(episode_meta_parts, ignore_index=True)
    overlap = (set(episode_meta.columns) & set(anchors.columns)) - {
        regime_col,
        "episode_index",
    }
    if overlap:
        episode_meta = episode_meta.drop(columns=list(overlap))
    anchors = anchors.merge(episode_meta, on=[regime_col, "episode_index"], how="left")

    if "episode_length" in anchors.columns:
        anchors["progress"] = (
            anchors["frame_index"] / (anchors["episode_length"] - 1).clip(lower=1)
        ).clip(0.0, 1.0)

    # Z-score each block globally (pooled across regimes), then weighted
    # energy-normalized concat for the observation vector.
    blocks: dict[str, np.ndarray] = {
        "proprio": _zscore(np.concatenate(proprio_parts)),
        "history": _zscore(np.concatenate(hist_parts)),
    }

    if extra_blocks:
        for name, table in extra_blocks.items():
            if isinstance(table, (str, Path)):
                table = pd.read_parquet(table)
            if "anchor_key" not in table.columns:
                raise ValueError(f"extra block {name!r} needs an 'anchor_key' column")
            vec_cols = [c for c in table.columns if c != "anchor_key"]
            if len(vec_cols) != 1:
                raise ValueError(
                    f"extra block {name!r} must have exactly one vector column, got {vec_cols}"
                )
            series = table.set_index("anchor_key")[vec_cols[0]]
            missing = ~anchors["anchor_key"].isin(series.index)
            if missing.any():
                raise ValueError(
                    f"extra block {name!r} is missing {int(missing.sum())} of "
                    f"{len(anchors)} anchors (e.g. "
                    f"{anchors.loc[missing, 'anchor_key'].iloc[0]!r}); recompute it "
                    f"for the current anchor set"
                )
            blocks[name] = _zscore(_as_matrix(series.loc[anchors["anchor_key"]].to_numpy()))

    obs = np.concatenate(
        [
            Z * (float(weights.get(name, 1.0)) / np.sqrt(Z.shape[1]))
            for name, Z in blocks.items()
        ],
        axis=1,
    ).astype(np.float32)
    traj = _zscore(np.concatenate(fut_parts)).astype(np.float32)

    anchors[obs_output] = list(obs)
    anchors[traj_output] = list(traj)
    anchors["anchor_id"] = np.arange(len(anchors), dtype=np.int32)

    logger.info(
        "Built %d anchors: %s %d dims (blocks: %s), %s %d dims.",
        len(anchors),
        obs_output,
        obs.shape[1],
        {n: z.shape[1] for n, z in blocks.items()},
        traj_output,
        traj.shape[1],
    )
    return anchors


# ---------------------------------------------------------------------------
# Cluster / multimodality analysis
# ---------------------------------------------------------------------------

SUBTASK_FAMILY_KEYWORDS = [("apart", "apart"), ("remove", "remove"), ("place", "place")]


def add_subtask_family(
    df: pd.DataFrame,
    *,
    task_col: str | None = None,
    output_col: str = "subtask_family",
) -> pd.DataFrame:
    """
    Merge raw task strings into coarse behavior families by keyword
    (``apart`` / ``remove`` / ``place`` / ``other``), like the YAM report.

    *task_col* defaults to ``"subtask"`` when present, else ``"task"``.
    """
    if task_col is None:
        task_col = "subtask" if "subtask" in df.columns else "task"

    def _family(value) -> str:
        text = str(value).lower()
        for keyword, family in SUBTASK_FAMILY_KEYWORDS:
            if keyword in text:
                return family
        return "other"

    df[output_col] = df[task_col].map(_family)
    return df


def add_cluster_columns(
    df: pd.DataFrame,
    *,
    x: str | None = None,
    y: str | None = None,
    vector_col: str | None = None,
    ks: Sequence[int] = (50, 150, 300),
    prefix: str = "blob",
    random_state: int = 0,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Add k-means "blob" columns at several granularities (the report's K slider
    as categorical columns).

    Clusters on the 2D projection (*x*, *y*) by default so blobs are spatially
    coherent in the widget; pass *vector_col* to cluster in the original
    high-dimensional space instead (robustness check).
    """
    from sklearn.cluster import KMeans

    if vector_col is not None:
        X = _as_matrix(df[vector_col].to_numpy())
    elif x is not None and y is not None:
        X = df[[x, y]].to_numpy(dtype=np.float32)
    else:
        raise ValueError("Pass either x/y (2D projection columns) or vector_col")

    added: list[str] = []
    for k in ks:
        k_eff = min(int(k), len(df))
        labels = KMeans(n_clusters=k_eff, n_init=4, random_state=random_state).fit_predict(X)
        col = f"{prefix}_k{k}"
        width = max(2, len(str(k_eff - 1)))
        df[col] = [f"b{lab:0{width}d}" for lab in labels]
        added.append(col)
    logger.info("Added cluster columns %s on %d points.", added, len(df))
    return df, added


def _blob_label_stats(
    df: pd.DataFrame,
    group_cols: list[str],
    label_col: str,
    min_count: int | None,
) -> pd.DataFrame:
    """Per-group label statistics with a noise threshold on label counts."""
    rows: list[dict] = []
    for key, group in df.groupby(group_cols, sort=False, observed=True):
        counts = group[label_col].value_counts()
        n = int(counts.sum())
        threshold = int(min_count) if min_count is not None else max(2, int(np.ceil(0.02 * n)))
        passing = counts[counts >= threshold]
        if passing.empty:
            passing = counts.iloc[:1]
        p = passing / passing.sum()
        if not isinstance(key, tuple):
            key = (key,)
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "n": n,
                "n_behaviors": int(len(passing)),
                "multimodal": bool(len(passing) >= 2),
                "purity": float(counts.iloc[0] / n),
                "entropy": float(-(p * np.log(p)).sum()),
                "place_remove": bool(
                    "place" in passing.index and "remove" in passing.index
                ),
            }
        )
    return pd.DataFrame(rows)


def add_neighborhood_stats(
    df: pd.DataFrame,
    *,
    blob_col: str,
    label_col: str = "subtask_family",
    regime_col: str | None = None,
    min_count: int | None = None,
    prefix: str | None = None,
) -> pd.DataFrame:
    """
    Compute per-neighborhood (blob) label statistics and merge them back as
    frame columns, so the atlas can be colored by local multimodality.

    Columns added: ``{prefix}_n_behaviors``, ``_multimodal`` (>= 2 labels pass
    the noise threshold ``max(2, 2% of blob)``), ``_purity``, ``_entropy``,
    ``_place_remove`` (both behaviors present — the report's collision flag).

    With *regime_col*, stats are computed within (blob, regime) so each
    regime's frames reflect their own local distribution. *label_col* may
    also be the other side's blob column, giving a cross-side fan-out count
    (the static analog of the report's hover linking).

    Returns a new DataFrame.
    """
    if prefix is None:
        prefix = f"{blob_col}_nbhd"
    group_cols = [blob_col] + ([regime_col] if regime_col else [])
    stats = _blob_label_stats(df, group_cols, label_col, min_count)
    rename = {
        c: f"{prefix}_{c}"
        for c in ["n_behaviors", "multimodal", "purity", "entropy", "place_remove"]
    }
    stats = stats.drop(columns=["n"]).rename(columns=rename)
    existing = [c for c in rename.values() if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
    return df.merge(stats, on=group_cols, how="left")


def multimodality_summary(
    df: pd.DataFrame,
    *,
    blob_cols: Sequence[str],
    label_col: str = "subtask_family",
    regime_col: str | None = "regime",
    min_count: int | None = None,
) -> pd.DataFrame:
    """
    Summarize how often observation neighborhoods map to multiple behaviors,
    per regime and blob granularity — the report's "60% adv vs 29% baseline
    multimodal" table and its ~7% place/remove collision figure.

    ``frac_blobs_*`` counts neighborhoods; ``frac_mass_*`` weights them by the
    number of anchors they contain.
    """
    if regime_col is not None and regime_col in df.columns:
        regimes = list(df[regime_col].unique())
    else:
        regimes, regime_col = [None], None

    rows: list[dict] = []
    for blob_col in blob_cols:
        for regime in regimes:
            sub = df if regime is None else df[df[regime_col] == regime]
            stats = _blob_label_stats(sub, [blob_col], label_col, min_count)
            total = int(stats["n"].sum())
            rows.append(
                {
                    "regime": "all" if regime is None else regime,
                    "neighborhoods": blob_col,
                    "n_blobs": len(stats),
                    "frac_blobs_multimodal": float(stats["multimodal"].mean()),
                    "frac_mass_multimodal": float(
                        stats.loc[stats["multimodal"], "n"].sum() / total
                    ),
                    "frac_blobs_place_remove": float(stats["place_remove"].mean()),
                    "frac_mass_place_remove": float(
                        stats.loc[stats["place_remove"], "n"].sum() / total
                    ),
                    "mean_local_behaviors": float(stats["n_behaviors"].mean()),
                }
            )
    return pd.DataFrame(rows)


def co_embed_vectors(
    train_vectors,
    new_vectors=None,
    *,
    umap_args: dict | None = None,
):
    """
    Fit UMAP on *train_vectors* and place *new_vectors* into the SAME 2D space
    via out-of-sample transform.

    This is the hook for policy-prediction analysis: fit on the dataset's
    action-chunk vectors, transform each policy's sampled chunks, and compare
    coverage in a shared map. Unlike ``compute_projection`` (which discards
    the fitted model), the reducer is returned for further transforms.

    Returns ``(train_xy, new_xy_or_None, reducer)``.
    """
    import umap

    args = {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "random_state": 42}
    if umap_args:
        args.update(umap_args)

    reducer = umap.UMAP(**args)
    train_xy = reducer.fit_transform(_as_matrix(train_vectors))
    new_xy = None
    if new_vectors is not None:
        new_xy = reducer.transform(_as_matrix(new_vectors))
    return train_xy, new_xy, reducer


# ---------------------------------------------------------------------------
# Video frames for tooltips (optional; requires videos/ + PyAV)
# ---------------------------------------------------------------------------


def video_keys(info: dict) -> list[str]:
    """Return the feature names with dtype 'video' (camera streams)."""
    return [
        name
        for name, spec in info.get("features", {}).items()
        if spec.get("dtype") == "video"
    ]


def has_videos(root: str | Path, video_key: str | None = None) -> bool:
    """Check whether the dataset's video files are present on disk."""
    root = Path(root)
    info = load_lerobot_info(root)
    keys = [video_key] if video_key else video_keys(info)
    if not keys or "video_path" not in info:
        return False
    probe = root / info["video_path"].format(
        episode_chunk=0, video_key=keys[0], episode_index=0
    )
    return probe.exists()


def _decode_episode_frames(
    video_path: Path,
    frame_indices: Sequence[int],
    size: int | None,
) -> dict[int, dict]:
    """Decode the requested frame indices from one video into PNG-bytes dicts."""
    try:
        import av
    except ImportError as e:
        raise ImportError(
            "Decoding LeRobot videos requires PyAV. Install with: pip install av"
        ) from e

    wanted = sorted(set(int(i) for i in frame_indices))
    out: dict[int, dict] = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        next_wanted = 0
        for i, frame in enumerate(container.decode(stream)):
            if next_wanted >= len(wanted):
                break
            if i == wanted[next_wanted]:
                img = frame.to_image()
                if size:
                    img.thumbnail((size, size))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                out[i] = {"bytes": buf.getvalue()}
                next_wanted += 1
    return out


def add_video_frame_column(
    df: pd.DataFrame,
    root: str | Path,
    *,
    video_key: str | None = None,
    output_column: str = "frame_image",
    max_images: int | None = 5000,
    size: int | None = 224,
    episode_col: str = "episode_index",
    frame_col: str = "frame_index",
) -> pd.DataFrame:
    """
    Decode camera frames from the dataset's videos into an image column
    (Embedding Atlas ``{"bytes": ...}`` format) for tooltip display.

    Requires the ``videos/`` directory to be present (LeRobot datasets are
    often downloaded without it) and PyAV. When *max_images* is smaller than
    the frame count, frames are sampled uniformly; other rows get ``None``.
    """
    root = Path(root)
    info = load_lerobot_info(root)
    if video_key is None:
        keys = video_keys(info)
        if not keys:
            raise ValueError("Dataset has no video features in info.json")
        video_key = keys[0]
    if not has_videos(root, video_key):
        raise FileNotFoundError(
            f"Video files for {video_key!r} not found under {root}. "
            "Download the dataset's videos/ directory to enable image tooltips."
        )

    chunks_size = int(info.get("chunks_size", 1000))
    df[output_column] = None

    if max_images is None or max_images >= len(df):
        selected = df.index
    else:
        step = max(1, len(df) // max_images)
        selected = df.index[::step][:max_images]

    sub = df.loc[selected, [episode_col, frame_col]]
    decoded = 0
    for ep, group in sub.groupby(episode_col):
        path = root / info["video_path"].format(
            episode_chunk=int(ep) // chunks_size,
            video_key=video_key,
            episode_index=int(ep),
        )
        if not path.exists():
            logger.warning("Missing video for episode %s: %s", ep, path)
            continue
        frames = _decode_episode_frames(path, group[frame_col].tolist(), size)
        for row_idx, f_idx in zip(group.index, group[frame_col]):
            img = frames.get(int(f_idx))
            if img is not None:
                df.at[row_idx, output_column] = img
                decoded += 1

    logger.info("Decoded %d video frames into column %r", decoded, output_column)
    return df
