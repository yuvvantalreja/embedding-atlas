"""
Real-World RL Experiment — SAC on BipedalWalker-v3
===================================================
A research-grade replay-buffer visualization powered by Embedding Atlas.

We train **Soft Actor-Critic (SAC)** on **BipedalWalker-v3**, a 24-dimensional
continuous-control locomotion task that is widely used as a stand-in for
real-world legged-robotics research. The replay buffer captures every
transition `(state, action, reward, next_state, done)`. We enrich each
transition with domain-specific metadata (gait phase, forward speed, critic
Q-value, policy entropy, episode outcome) and project the (state, action)
vectors to 2D with UMAP.

Run with:
    marimo edit experiment.py

Prerequisites:
    pip install "stable-baselines3[extra]" "gymnasium[box2d]" umap-learn matplotlib

Training time:
    ~25-30 min on a modern CPU for 200k SAC training steps, plus another
    ~15-20 min to collect 200k post-training rollout transitions with rendered
    frames. Every artifact is cached next to the notebook (SAC checkpoint,
    replay buffer, and the rollout dataset as parquet) so re-running picks up
    cached state instead of regenerating.

What you should see:
    - Gait-phase clusters (double_stance, left_stance, right_stance, flight).
    - A smooth critic-value gradient across the embedding.
    - Surviving episodes that trace coherent paths through high-value regions.
    - Failed episodes that terminate in a distinct "fall" region.
    - Hovering on any point shows the rendered BipedalWalker frame for that
      transition.
"""

import marimo

__generated_with = "0.23.8"
app = marimo.App(
    width="full",
    app_title="Real-World RL Experiment — SAC + BipedalWalker",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    return mo, np, pd


@app.cell
def _():
    from embedding_atlas.widget import EmbeddingAtlasWidget
    from embedding_atlas.projection import async_compute_projection
    from embedding_atlas.rl_activations import add_activation_column
    import gymnasium as gym
    from stable_baselines3 import SAC
    import matplotlib
    matplotlib.use("Agg")
    return (
        EmbeddingAtlasWidget,
        SAC,
        add_activation_column,
        async_compute_projection,
        gym,
    )


@app.cell
def _():
    from embedding_atlas.rl_utils import (
        prepare_rl_data_for_projection,
        unpack_state_components,
        format_state_description,
        compute_episode_metadata,
    )

    return (
        compute_episode_metadata,
        format_state_description,
        prepare_rl_data_for_projection,
        unpack_state_components,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## BipedalWalker-v3
    """)
    return


@app.cell(hide_code=True)
def _(SAC, gym):
    """Train SAC on BipedalWalker-v3, or load a cached checkpoint if available."""
    import os

    CHECKPOINT = "sac_bipedalwalker.zip"
    REPLAY = "sac_bipedalwalker_buffer.pkl"
    TOTAL_STEPS = 200_000
    BUFFER_SIZE = 300_000

    # RL-Zoo-tuned hyperparameters for BipedalWalker-v3.
    sac_kwargs = dict(
        learning_rate=7.3e-4,
        buffer_size=BUFFER_SIZE,
        batch_size=256,
        gamma=0.98,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10_000,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[400, 300]),
        # Must be False so the replay buffer keeps `next_observations` explicitly.
        optimize_memory_usage=False,
        verbose=0,
        seed=42,
    )

    walker_env = gym.make("BipedalWalker-v3")

    if os.path.exists(CHECKPOINT) and os.path.exists(REPLAY):
        print(f"Loading cached SAC model + replay buffer from {CHECKPOINT}, {REPLAY}")
        sac_model = SAC.load(CHECKPOINT, env=walker_env)
        # NOTE: SAC.load() does NOT restore the replay buffer — load it separately.
        sac_model.load_replay_buffer(REPLAY)
    else:
        print(f"Training SAC for {TOTAL_STEPS:,} steps (~25-30 min on CPU)...")
        sac_model = SAC("MlpPolicy", walker_env, **sac_kwargs)
        sac_model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)
        sac_model.save(CHECKPOINT)
        sac_model.save_replay_buffer(REPLAY)
        print(f"Saved checkpoint to {CHECKPOINT} and replay buffer to {REPLAY}")

    walker_env.close()
    return os, sac_model


@app.cell(hide_code=True)
def _(gym, np, os, pd, sac_model):
    """
    Collect ~200k transitions by rolling out the trained policy with rendering
    on, capturing a downscaled JPEG of each step for the embedding-atlas tooltip.

    We can't render replay-buffer states retroactively (the simulator's full
    state isn't stored), so we generate a fresh on-policy dataset where every
    transition has an aligned image. The first run is slow (~15-20 min for
    rendering 200k frames); subsequent runs load the cached parquet instead.
    """
    import io

    from PIL import Image

    DATASET_PARQUET = "sac_bipedalwalker_rollouts.parquet"
    TARGET_TRANSITIONS = 200_000
    IMG_WIDTH = 240
    IMG_HEIGHT = 160
    JPEG_QUALITY = 65  # JPEG keeps the column small enough to fit in memory.

    def _load_cached():
        print(f"Loading cached rollout dataset from {DATASET_PARQUET}")
        df = pd.read_parquet(DATASET_PARQUET)
        # Restore the image-column dict wrapping that parquet flattens.
        if "environment_image" in df.columns:
            df["environment_image"] = df["environment_image"].apply(
                lambda b: {"bytes": b} if isinstance(b, (bytes, bytearray)) else b
            )
        print(f"Loaded {len(df):,} transitions with images.")
        return df

    def _build_fresh():
        print(
            f"Rolling out trained policy to collect {TARGET_TRANSITIONS:,} "
            f"transitions with rendered frames (~15-20 min)..."
        )
        render_env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
        rows: list[dict] = []
        episode_id = 0
        obs, _ = render_env.reset(seed=42)

        while len(rows) < TARGET_TRANSITIONS:
            # deterministic=False so the dataset includes some policy noise;
            # this gives a richer mix of survived/fell trajectories.
            action, _ = sac_model.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _info = render_env.step(action)
            done = bool(terminated or truncated)

            frame = render_env.render()
            # Downscale + JPEG-encode to keep the image column small.
            pil = Image.fromarray(frame.astype(np.uint8), "RGB").resize(
                (IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR
            )
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=JPEG_QUALITY)
            img_bytes = buf.getvalue()

            rows.append({
                "state": obs.flatten().tolist(),
                "action": np.asarray(action, dtype=np.float32).flatten().tolist(),
                "reward": float(reward),
                "next_state": next_obs.flatten().tolist(),
                "done": done,
                "timestep": len(rows),
                "rollout_episode": episode_id,
                "environment_image": img_bytes,
            })

            if done:
                episode_id += 1
                obs, _ = render_env.reset()
                if episode_id % 25 == 0:
                    print(
                        f"  episode {episode_id}, "
                        f"{len(rows):,} / {TARGET_TRANSITIONS:,} transitions"
                    )
            else:
                obs = next_obs

        render_env.close()

        df = pd.DataFrame(rows)
        print(
            f"Collected {len(df):,} transitions across {episode_id + 1} episodes."
        )

        # Cache raw bytes to parquet; we re-wrap into {"bytes": ...} on load.
        print(f"Caching dataset to {DATASET_PARQUET} (this can take a minute)...")
        df.to_parquet(DATASET_PARQUET, compression="snappy")

        # Wrap image bytes for the embedding-atlas format.
        df["environment_image"] = df["environment_image"].apply(
            lambda b: {"bytes": b}
        )
        return df

    walker_df = _load_cached() if os.path.exists(DATASET_PARQUET) else _build_fresh()
    return (walker_df,)


@app.cell(hide_code=True)
def _(
    compute_episode_metadata,
    format_state_description,
    np,
    pd,
    sac_model,
    unpack_state_components,
    walker_df,
):
    """Enrich the replay-buffer DataFrame with locomotion-specific metadata."""
    import torch

    w_df = walker_df.copy()

    # ----- Unpack the 24D observation into named scalars -----
    STATE_COMPONENTS = [
        "hull_angle", "hull_angular_vel",
        "vel_x", "vel_y",
        "hip_1_angle", "knee_1_angle", "hip_1_speed", "knee_1_speed",
        "hip_2_angle", "knee_2_angle", "hip_2_speed", "knee_2_speed",
        "leg_1_contact", "leg_2_contact",
        *[f"lidar_{i}" for i in range(10)],
    ]
    w_df = unpack_state_components(w_df, "state", STATE_COMPONENTS)

    # ----- Unpack the 4D continuous action into joint torques -----
    ACTION_COMPONENTS = ["hip_1_torque", "knee_1_torque", "hip_2_torque", "knee_2_torque"]
    w_df = unpack_state_components(w_df, "action", ACTION_COMPONENTS)

    # ----- Episode metadata (episode id, step, length, return, phase) -----
    w_df = compute_episode_metadata(w_df, done_col="done", reward_col="reward")

    # ----- Derived locomotion features -----
    w_df["forward_speed"] = w_df["vel_x"]
    w_df["action_magnitude"] = np.sqrt(
        w_df[ACTION_COMPONENTS].pow(2).sum(axis=1)
    )

    # Body orientation from hull angle.
    w_df["body_orientation"] = "upright"
    w_df.loc[w_df["hull_angle"] > 0.25, "body_orientation"] = "tilted_fwd"
    w_df.loc[w_df["hull_angle"] < -0.25, "body_orientation"] = "tilted_back"

    # Gait phase from foot-contact pattern.
    _c1 = w_df["leg_1_contact"] > 0.5
    _c2 = w_df["leg_2_contact"] > 0.5
    w_df["gait_phase"] = np.select(
        [_c1 & _c2, _c1 & ~_c2, ~_c1 & _c2],
        ["double_stance", "left_stance", "right_stance"],
        default="flight",
    )

    # Episode outcome from episode return.
    w_df["episode_outcome"] = "partial"
    w_df.loc[w_df["episode_return"] > 100, "episode_outcome"] = "survived"
    w_df.loc[w_df["episode_return"] < -50, "episode_outcome"] = "fell"

    # ----- SAC critic Q-value (min of twin critics) + policy entropy -----
    # Batched torch passes; row-by-row would take ~5 min at 200k rows.
    device = sac_model.device
    states_arr = np.vstack(
        w_df["state"].apply(np.asarray).to_numpy()
    ).astype(np.float32)
    actions_arr = np.vstack(
        w_df["action"].apply(np.asarray).to_numpy()
    ).astype(np.float32)

    BATCH = 4096
    q_values: list[float] = []
    entropies: list[float] = []
    with torch.no_grad():
        for i in range(0, len(states_arr), BATCH):
            s_t = torch.as_tensor(states_arr[i : i + BATCH], device=device)
            a_t = torch.as_tensor(actions_arr[i : i + BATCH], device=device)
            q1, q2 = sac_model.critic(s_t, a_t)
            q_values.extend(torch.min(q1, q2).flatten().cpu().tolist())
            # Policy entropy per state: -log_pi(a*|s) where a* ~ current policy.
            _, log_prob = sac_model.actor.action_log_prob(s_t)
            entropies.extend((-log_prob).flatten().cpu().tolist())

    w_df["q_value"] = q_values
    w_df["policy_entropy"] = entropies

    # Tercile bins for the entropy chart.
    _e33, _e67 = np.percentile(entropies, [33, 67])
    w_df["entropy_category"] = pd.cut(
        w_df["policy_entropy"],
        bins=[-np.inf, _e33, _e67, np.inf],
        labels=["low_entropy", "mid_entropy", "high_entropy"],
    ).astype(str)

    # Q-value tercile bins for filtering.
    _q33, _q67 = np.percentile(q_values, [33, 67])
    w_df["q_category"] = pd.cut(
        w_df["q_value"],
        bins=[-np.inf, _q33, _q67, np.inf],
        labels=["low_q", "mid_q", "high_q"],
    ).astype(str)

    # Hover text for the widget.
    w_df["state_description"] = format_state_description(
        w_df,
        component_cols=["hull_angle", "forward_speed", "action_magnitude"],
        action_col="gait_phase",
        q_col="q_value",
        reward_col="reward",
    )

    walker_enriched = w_df
    print(f"Enriched {len(walker_enriched):,} transitions with locomotion metadata.")
    return (walker_enriched,)


@app.cell(hide_code=True)
async def _(
    async_compute_projection,
    np,
    prepare_rl_data_for_projection,
    walker_enriched,
):
    """UMAP projection of (state, action) vectors with feature standardization."""
    walker_proj = walker_enriched.copy()
    walker_proj, _vec_col = prepare_rl_data_for_projection(
        walker_proj,
        observation_columns=["state"],
        action_columns=["action"],
    )

    # Z-score-normalize the concatenated vector. BipedalWalker mixes features on
    # very different scales (lidar in [0, 1], joint velocities unbounded), so a
    # raw cosine/euclidean metric is dominated by the noisier dimensions.
    _matrix = np.vstack(walker_proj[_vec_col].apply(np.asarray).to_numpy()).astype(np.float32)
    _mu = _matrix.mean(axis=0)
    _sigma = _matrix.std(axis=0) + 1e-6
    _matrix = (_matrix - _mu) / _sigma
    walker_proj[_vec_col] = list(_matrix)

    walker_proj = await async_compute_projection(
        walker_proj,
        inputs=_vec_col,
        modality="vector",
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
        umap_args={"n_neighbors": 25, "min_dist": 0.1, "metric": "euclidean"},
    )

    walker_proj = walker_proj.drop(
        columns=[c for c in [_vec_col, "state", "next_state"] if c in walker_proj.columns]
    )
    print(f"BipedalWalker projection ready: {len(walker_proj):,} points")
    return (walker_proj,)


@app.cell
def _(EmbeddingAtlasWidget, walker_proj):
    walker_widget = EmbeddingAtlasWidget(
        walker_proj,
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
        text="state_description",
        image="environment_image",
        labels="automatic",
        show_charts=True,
        show_table=True,
        point_size=2.0,
        trajectory_id_field="episode",
        trajectories={
            "group_by": "episode",
            "order_by": "step_in_episode",
            "color_by": "episode_outcome",
            "colors": {
                "survived": "#16a34a",
                "fell": "#dc2626",
                "partial": "#64748b",
            },
            "max_groups": 40,
            "width": 1,
            "opacity": 0.001,
        },
    )
    return (walker_widget,)


@app.cell
def _(walker_widget):
    walker_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Activation View

    The projection above uses the raw `(state, action)` vector. clusters reflect
    physical similarity (hull angle, joint positions, lidar readings).

    Below we re-project using the trained penultimate-layer
    activations . Each point is now
    placed by how the policy network internally represents that state.

    States the policy treats as equivalent end up adjacent, even if their raw
    data differs.
    """)
    return


@app.cell(hide_code=True)
async def _(
    add_activation_column,
    async_compute_projection,
    np,
    prepare_rl_data_for_projection,
    sac_model,
    walker_enriched,
):
    """Re-project BipedalWalker using SAC actor penultimate activations."""
    walker_act = walker_enriched.copy()
    walker_act = add_activation_column(walker_act, sac_model, layer="auto")
    walker_act, _vec_col = prepare_rl_data_for_projection(
        walker_act,
        observation_columns=["activations"],
        action_columns=None,
    )

    # Activation values can have very different per-dim scales; standardize before UMAP
    # so the metric isn't dominated by the noisier dimensions (same rationale as the
    # raw-state projection above).
    _matrix = np.vstack(walker_act[_vec_col].apply(np.asarray).to_numpy()).astype(np.float32)
    _mu = _matrix.mean(axis=0)
    _sigma = _matrix.std(axis=0) + 1e-6
    _matrix = (_matrix - _mu) / _sigma
    walker_act[_vec_col] = list(_matrix)

    walker_act = await async_compute_projection(
        walker_act,
        inputs=_vec_col,
        modality="vector",
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
        umap_args={"n_neighbors": 25, "min_dist": 0.1, "metric": "euclidean"},
    )
    walker_act = walker_act.drop(
        columns=[c for c in [_vec_col, "state", "next_state", "activations"] if c in walker_act.columns]
    )
    print(f"BipedalWalker activation projection ready: {len(walker_act):,} points")
    return (walker_act,)


@app.cell
def _(EmbeddingAtlasWidget, walker_act):
    walker_act_widget = EmbeddingAtlasWidget(
        walker_act,
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
        text="state_description",
        image="environment_image",
        labels="automatic",
        show_charts=True,
        show_table=True,
        point_size=2.0,
        trajectory_id_field="episode",
        trajectories={
            "group_by": "episode",
            "order_by": "step_in_episode",
            "color_by": "episode_outcome",
            "colors": {
                "survived": "#16a34a",
                "fell": "#dc2626",
                "partial": "#64748b",
            },
            "max_groups": 40,
            "width": 1,
            "opacity": 0.001,
        },
    )
    return (walker_act_widget,)


@app.cell
def _(walker_act_widget):
    walker_act_widget
    return


if __name__ == "__main__":
    app.run()
