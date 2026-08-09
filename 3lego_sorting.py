"""
3-Lego Sorting — LeRobot Dataset Atlas
======================================
Explore a bimanual "yam" robot demonstration dataset (525 episodes, 862K
frames at 60 fps) with Embedding Atlas:

  1. Frame-level atlas  – every ~20th frame embedded by (state, action) with
     temporal context; episode trajectories overlaid.
  2. Episode-level atlas – one trajectory signature per demonstration;
     behaviors cluster, outlier demos stand out.

Run with:
    marimo edit 3lego_sorting.py

Uses helpers from embedding_atlas.lerobot_utils.
"""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full", app_title="3-Lego Sorting — LeRobot Atlas")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    return mo, np


@app.cell
def _():
    from embedding_atlas.lerobot_utils import (
        add_episode_progress,
        add_kinematic_features,
        build_episode_vectors,
        build_frame_vectors,
        format_frame_description,
        load_lerobot_episodes,
        load_lerobot_frames,
        load_lerobot_info,
    )
    from embedding_atlas.projection import async_compute_projection
    from embedding_atlas.widget import EmbeddingAtlasWidget

    return (
        EmbeddingAtlasWidget,
        add_episode_progress,
        add_kinematic_features,
        async_compute_projection,
        build_episode_vectors,
        build_frame_vectors,
        format_frame_description,
        load_lerobot_episodes,
        load_lerobot_frames,
        load_lerobot_info,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3-Lego Sorting
    """)
    return


@app.cell
def _():
    DATASET_ROOT = "datasets/3lego_round1_baseline"
    # 60 fps -> keep every 20th frame (3 Hz). 862K frames -> ~43K points.
    # sampling 
    STRIDE = 5
    return DATASET_ROOT, STRIDE


@app.cell
def _(
    DATASET_ROOT,
    STRIDE,
    add_episode_progress,
    add_kinematic_features,
    format_frame_description,
    load_lerobot_frames,
    load_lerobot_info,
):
    info = load_lerobot_info(DATASET_ROOT)
    frames = load_lerobot_frames(DATASET_ROOT, stride=STRIDE)
    frames = add_episode_progress(frames)
    frames = add_kinematic_features(frames, info=info)
    frames["description"] = format_frame_description(frames)
    print(f"{len(frames)} frames, {frames['episode_index'].nunique()} episodes")
    return (frames,)


@app.cell(hide_code=True)
def _(frames, mo):
    _counts = (
        frames.drop_duplicates("episode_index")
        .groupby("subtask")
        .size()
        .sort_values(ascending=False)
        .rename("episodes")
        .reset_index()
    )
    mo.vstack([
        mo.md("### Episodes per subtask"),
        mo.ui.table(_counts, selection=None),
        mo.md(
            "Note the long tail: *remove X from wrong tray* recovery "
            "behaviors have only 1–3 demonstrations each."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Frame-level atlas

    Each point is one (strided) frame, embedded from the z-scored
    `(observation.state, action)` vector stacked over a ±2-frame temporal
    window — so the map clusters *motion snippets* (reach, grasp, transport,
    release), not just static poses.

    Things to try:

    - Color by `task` to see which motions are shared across subtasks.
    - Color by `active_arm` — bimanual vs single-arm regions.
    - Filter `tracking_error` high: contact events and teleop corrections.
    - Click a point to focus its episode's trajectory through the map.
    """)
    return


@app.cell
async def _(async_compute_projection, build_frame_vectors, frames):
    frame_atlas_df = frames.copy()
    frame_atlas_df, _vec = build_frame_vectors(
        frame_atlas_df,
        columns=("observation.state", "action"),
        window=2,
    )
    frame_atlas_df = await async_compute_projection(
        frame_atlas_df,
        inputs=_vec,
        modality="vector",
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
        umap_args={"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"},
    )
    frame_atlas_df = frame_atlas_df.drop(
        columns=[
            c
            for c in [_vec, "observation.state", "action"]
            if c in frame_atlas_df.columns
        ]
    )
    print(f"Frame atlas ready: {len(frame_atlas_df)} points")
    return (frame_atlas_df,)


@app.cell
def _(EmbeddingAtlasWidget, frame_atlas_df):
    frame_widget = EmbeddingAtlasWidget(
        frame_atlas_df,
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
        text="description",
        labels="automatic",
        # color="task",
        show_charts=True,
        show_table=True,
        point_size=1.0
    )
    return (frame_widget,)


@app.cell
def _(frame_widget):
    frame_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Episode-level atlas

    One point per demonstration: each episode's 14-D state trajectory is
    resampled to 32 timesteps and flattened into a fixed-length signature,
    so whole behaviors — not frames — are compared.

    Things to try:

    - Color by `subtask`: same-behavior demos should form tight clusters.
      Demos far from their cluster are candidates for review (weird
      teleop, mislabeled task, unusual scene layout).
    - Color by `duration_s` to spot abnormally slow demonstrations.
    - The rare *remove-from-wrong-tray* recovery demos sit in their own
      corners — a direct view of dataset imbalance.
    """)
    return


@app.cell
async def _(
    DATASET_ROOT,
    async_compute_projection,
    build_episode_vectors,
    frames,
    load_lerobot_episodes,
):
    _ep_vecs, _evec = build_episode_vectors(
        frames,
        columns=("observation.state",),
        n_samples=32,
    )
    episode_atlas_df = load_lerobot_episodes(DATASET_ROOT).merge(
        _ep_vecs, on="episode_index"
    )
    episode_atlas_df = await async_compute_projection(
        episode_atlas_df,
        inputs=_evec,
        modality="vector",
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
        umap_args={"n_neighbors": 10, "min_dist": 0.05, "metric": "cosine"},
    )
    episode_atlas_df = episode_atlas_df.drop(columns=[_evec])
    print(f"Episode atlas ready: {len(episode_atlas_df)} episodes")
    return (episode_atlas_df,)


@app.cell
def _(EmbeddingAtlasWidget, episode_atlas_df):
    episode_widget = EmbeddingAtlasWidget(
        episode_atlas_df,
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
        text="subtask",
        labels="automatic",
        show_charts=True,
        show_table=True,
        point_size=5.0,
    )
    return (episode_widget,)


@app.cell
def _(episode_widget):
    episode_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Observation ↔ Trajectory linked atlas

    Rebuild of the data-side analysis from `initial_report_v6.html`: every
    anchor frame (1 per second) yields an **observation** vector
    (proprioception + binned 120-step action history — what a policy
    conditions on) and a **future-trajectory** vector (binned next-180-step
    action chunk — what the demonstrator did). Both are embedded separately;
    the two maps below share the same rows.

    **Brush the observation map, then click "Update trajectory view" — the
    trajectory map lights up the chunks those observations produce.** An
    observation region that fans out to
    *multiple* trajectory regions is multimodal p(a|obs) — the report's
    central finding (adv ≈ 60% vs baseline ≈ 29% of obs neighborhoods
    multimodal). Caveats: this uses UMAP (report: t-SNE) and no vision
    features yet (see `vision.md`), so expect the adv ≫ baseline *gap*
    to reproduce, not the exact numbers.
    """)
    return


@app.cell
def _():
    from embedding_atlas.lerobot_utils import (
        add_cluster_columns,
        add_neighborhood_stats,
        add_subtask_family,
        build_anchor_vectors,
        multimodality_summary,
    )

    return (
        add_cluster_columns,
        add_neighborhood_stats,
        add_subtask_family,
        build_anchor_vectors,
        multimodality_summary,
    )


@app.cell
def _():
    DATASET_ROOTS = {
        "baseline": "datasets/3lego_round1_baseline",
        "adv_round1": "datasets/adversarial/3lego_round1",
        "adv_round2": "datasets/adversarial/3lego_round2",
        "adv_round3": "datasets/adversarial/3lego_round3",
        "adv_round4": "datasets/adversarial/3lego_round4",
    }
    ANCHOR_STRIDE = 60  # 60 fps -> one anchor per second
    HISTORY_STEPS = 120  # 2 s of action history (report: 120-step history)
    HORIZON = 180  # 3 s future action chunk (report: H=180)
    HISTORY_BINS = 12
    HORIZON_BINS = 12
    BLOB_KS = (50, 150, 300)
    PRIMARY_K = 150
    return (
        ANCHOR_STRIDE,
        BLOB_KS,
        DATASET_ROOTS,
        HISTORY_BINS,
        HISTORY_STEPS,
        HORIZON,
        HORIZON_BINS,
        PRIMARY_K,
    )


@app.cell
def _(
    ANCHOR_STRIDE,
    DATASET_ROOTS,
    HISTORY_BINS,
    HISTORY_STEPS,
    HORIZON,
    HORIZON_BINS,
    add_subtask_family,
    build_anchor_vectors,
    np,
):
    anchors = build_anchor_vectors(
        DATASET_ROOTS,
        anchor_stride=ANCHOR_STRIDE,
        history_steps=HISTORY_STEPS,
        horizon=HORIZON,
        history_bins=HISTORY_BINS,
        horizon_bins=HORIZON_BINS,
    )
    anchors = add_subtask_family(anchors)
    # Coarse regime for the report-style comparison; `regime` keeps the
    # per-round detail for charts and filtering.
    anchors["collection"] = np.where(
        anchors["regime"] == "baseline", "baseline", "adversarial"
    )
    anchors["anchor_text"] = (
        anchors["subtask_family"]
        + ": "
        + anchors["task"].fillna("")
        + " | "
        + anchors["episode_id"]
        + " @ "
        + (anchors["progress"].fillna(0.0) * 100).round().astype(int).astype(str)
        + "%"
    )
    print(anchors.groupby("regime").size().rename("anchors").to_string())
    return (anchors,)


@app.cell
async def _(anchors, async_compute_projection):
    anchors_proj = anchors.copy()
    anchors_proj = await async_compute_projection(
        anchors_proj,
        inputs="obs_vector",
        modality="vector",
        x="obs_x",
        y="obs_y",
        neighbors="obs_neighbors",
        umap_args={
            "metric": "euclidean",
            "n_neighbors": 15,
            "min_dist": 0.1,
            "random_state": 42,
        },
    )
    anchors_proj = await async_compute_projection(
        anchors_proj,
        inputs="traj_vector",
        modality="vector",
        x="traj_x",
        y="traj_y",
        neighbors="traj_neighbors",
        umap_args={
            "metric": "euclidean",
            "n_neighbors": 15,
            "min_dist": 0.1,
            "random_state": 42,
        },
    )
    print(f"Anchor projections ready: {len(anchors_proj)} points")
    return (anchors_proj,)


@app.cell
def _(
    BLOB_KS,
    PRIMARY_K,
    add_cluster_columns,
    add_neighborhood_stats,
    anchors_proj,
):
    anchor_atlas_df = anchors_proj.copy()
    anchor_atlas_df, blob_cols_obs = add_cluster_columns(
        anchor_atlas_df, x="obs_x", y="obs_y", ks=BLOB_KS, prefix="obs_blob"
    )
    anchor_atlas_df, blob_cols_traj = add_cluster_columns(
        anchor_atlas_df, x="traj_x", y="traj_y", ks=BLOB_KS, prefix="traj_blob"
    )
    # Local behavior stats per obs / traj neighborhood, and the cross-side
    # fan-out (how many trajectory regions an obs neighborhood reaches).
    anchor_atlas_df = add_neighborhood_stats(
        anchor_atlas_df,
        blob_col=f"obs_blob_k{PRIMARY_K}",
        regime_col="collection",
        prefix="obs_nbhd",
    )
    anchor_atlas_df = add_neighborhood_stats(
        anchor_atlas_df,
        blob_col=f"traj_blob_k{PRIMARY_K}",
        regime_col="collection",
        prefix="traj_nbhd",
    )
    anchor_atlas_df = add_neighborhood_stats(
        anchor_atlas_df,
        blob_col=f"obs_blob_k{PRIMARY_K}",
        label_col=f"traj_blob_k{PRIMARY_K}",
        regime_col="collection",
        prefix="obs_fanout",
    )
    anchor_atlas_df = anchor_atlas_df.drop(columns=["obs_vector", "traj_vector"])
    return anchor_atlas_df, blob_cols_obs


@app.cell
def _(EmbeddingAtlasWidget, anchor_atlas_df, mo):
    obs_widget = EmbeddingAtlasWidget(
        anchor_atlas_df,
        x="obs_x",
        y="obs_y",
        neighbors="obs_neighbors",
        row_id="anchor_id",
        text="anchor_text",
        labels="automatic",
        show_charts=True,
        show_table=False,
        point_size=2.0,
        default_charts_include=[
            "collection",
            "regime",
            "subtask_family",
            "subtask",
            "progress",
            "obs_nbhd_n_behaviors",
            "obs_nbhd_multimodal",
            "obs_nbhd_place_remove",
            "obs_fanout_n_behaviors",
            "future_padded_frac",
        ],
    )
    obs_ui = mo.ui.anywidget(obs_widget)
    return obs_ui, obs_widget


@app.cell
def _(obs_ui):
    obs_ui
    return


@app.cell
def _(mo):
    link_btn = mo.ui.run_button(label="🔗 Update trajectory view from selection")
    link_btn
    return (link_btn,)


@app.cell
def _(EmbeddingAtlasWidget, anchor_atlas_df, link_btn, np, obs_widget):
    # Re-runs on every button press (link_btn.value). The predicate is read
    # from the raw obs_widget — deliberately NOT via mo.ui.anywidget(...)
    # .value, which would re-trigger this cell on every pan/zoom state sync.
    # A fresh widget is created on each run: marimo invalidates UI elements
    # from previous runs of a cell, so caching them renders blank.
    link_btn.value
    _pred = obs_widget.selection(format="predicate")
    if _pred is None:
        _mask = np.zeros(len(anchor_atlas_df), dtype=bool)
    else:
        _ids = obs_widget._connection.sql(
            f"SELECT anchor_id FROM embedding_atlas WHERE {_pred}"
        ).df()["anchor_id"]
        _mask = anchor_atlas_df["anchor_id"].isin(_ids).to_numpy()
    _traj_df = anchor_atlas_df.assign(
        obs_selection=np.where(_mask, "selected", "context")
    )
    traj_widget = EmbeddingAtlasWidget(
        _traj_df,
        x="traj_x",
        y="traj_y",
        neighbors="traj_neighbors",
        row_id="anchor_id",
        text="anchor_text",
        color="obs_selection",
        labels="disabled",
        show_charts=False,
        show_table=False,
        point_size=2.0,
    )
    print(
        f"Trajectory view: {int(_mask.sum())} selected / {len(_traj_df)} anchors"
        + ("" if _pred else " (no selection — brush the obs map, then click the button)")
    )
    return (traj_widget,)


@app.cell
def _(traj_widget):
    traj_widget
    return


@app.cell(hide_code=True)
def _(anchor_atlas_df, blob_cols_obs, mo, multimodality_summary):
    mm_summary = multimodality_summary(
        anchor_atlas_df, blob_cols=blob_cols_obs, regime_col="collection"
    )
    mm_by_round = multimodality_summary(
        anchor_atlas_df, blob_cols=blob_cols_obs, regime_col="regime"
    )
    _note = (
        "Both collections present — report comparison: adv ≈ 60% vs baseline "
        "≈ 29% multimodal obs neighborhoods, ~7% place/remove collisions. "
        "Absolute numbers read higher here (no vision features yet — coarser "
        "obs neighborhoods); the adversarial−baseline *gap* is the readout."
        if anchor_atlas_df["collection"].nunique() >= 2
        else "Only one collection loaded. Add the adversarial datasets to "
        "`DATASET_ROOTS` to reproduce the report's comparison."
    )
    mo.vstack(
        [
            mo.md("### Multimodality: obs neighborhoods mapping to ≥2 behaviors"),
            mo.ui.table(mm_summary.round(3), selection=None),
            mo.md(_note),
            mo.md("Per adversarial round:"),
            mo.ui.table(mm_by_round.round(3), selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Policy predictions (deferred)

    The report's §3–11 compare each policy's sampled action chunks against
    the dataset's action manifold. To activate that here, provide per-anchor
    sampled chunks (parquet: `anchor_key`, `policy`, `sample_idx`, and an
    H-step action-chunk vector binned like `traj_vector`), then use
    `embedding_atlas.lerobot_utils.co_embed_vectors(dataset_chunks,
    predicted_chunks)` to place predictions into the dataset's trajectory
    map — one atlas, colored by source, with under-coverage visible as
    dataset regions no prediction reaches. Vision features for the obs side
    are likewise deferred — see `vision.md`.
    """)
    return


if __name__ == "__main__":
    app.run()
