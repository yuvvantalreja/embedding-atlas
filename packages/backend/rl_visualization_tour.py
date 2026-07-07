"""
RL Visualization — Embedding Atlas
====================================
Three interactive sections powered by real RL training data:

  1. CartPole DQN  – Policy evolution with rich metadata and guided exploration
  2. Trajectories  – Follow individual episodes through the embedding
  3. LunarLander   – Dense rewards, 8D state, flight-phase clustering

Run with:
    marimo edit rl_visualization_tour.py

Prerequisites:
    pip install stable-baselines3 gymnasium umap-learn matplotlib
"""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full", app_title="RL Visualization — Embedding Atlas")


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
    from stable_baselines3 import DQN
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return (
        DQN,
        EmbeddingAtlasWidget,
        add_activation_column,
        async_compute_projection,
        gym,
    )


@app.cell
def _():
    from embedding_atlas.rl_utils import (
        prepare_rl_data_for_projection,
        unpack_state_components,
        compute_td_error,
        format_state_description,
        compute_episode_metadata,
        plot_episode_timeline,
    )

    return (
        compute_episode_metadata,
        compute_td_error,
        format_state_description,
        prepare_rl_data_for_projection,
        unpack_state_components,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Replay Buffer Visualization
    """)
    return


@app.cell(hide_code=True)
def _(DQN, gym, pd):
    """Train CartPole DQN for 50k steps and extract replay buffer."""
    TOTAL_STEPS = 50_000
    BUFFER_SIZE = TOTAL_STEPS + 1000

    env = gym.make("CartPole-v1")
    cartpole_model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=BUFFER_SIZE,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=0,
        seed=42,
    )

    print("model:", cartpole_model)

    cartpole_model.learn(total_timesteps=TOTAL_STEPS)
    env.close()

    # Extract replay buffer
    _buf = cartpole_model.replay_buffer
    _n = _buf.pos


    _rows = [
        {
            "state": _buf.observations[i].flatten().tolist(),
            "action": int(_buf.actions[i].flat[0]),
            "reward": float(_buf.rewards[i].flat[0]),
            "next_state": _buf.next_observations[i].flatten().tolist(),
            "done": bool(_buf.dones[i].flat[0]),
            "timestep": i,
        }
        for i in range(_n)
    ]
    cartpole_df = pd.DataFrame(_rows)

    print(cartpole_df.head)

    print(f"Extracted {len(cartpole_df)} replay buffer transitions.")
    return cartpole_df, cartpole_model


@app.cell(hide_code=True)
def _(
    cartpole_df,
    cartpole_model,
    compute_episode_metadata,
    compute_td_error,
    format_state_description,
    np,
    pd,
    unpack_state_components,
):
    """Build rich metadata for CartPole replay buffer."""
    if cartpole_df is not None:
        cp_df = cartpole_df.copy()

        # Unpack 4D state into named columns
        cp_df = unpack_state_components(
            cp_df,
            "state",
            [
                "cart_position",
                "cart_velocity",
                "pole_angle",
                "pole_angular_velocity",
            ],
        )

        # Human-readable action names
        cp_df["action_name"] = cp_df["action"].map({0: "push_left", 1: "push_right"})

        # Episode metadata
        cp_df = compute_episode_metadata(cp_df, done_col="done", reward_col="reward")

        print(cp_df.head)

        # Training stage
        _n = len(cp_df)
        cp_df["training_stage"] = pd.cut(
            cp_df["timestep"],
            bins=[-1, _n // 3, 2 * _n // 3, _n],
            labels=["early", "mid", "late"],
        ).astype(str)

        # Pole danger categories
        cp_df["pole_danger"] = "safe"
        cp_df.loc[cp_df["pole_angle"].abs() > 0.10, "pole_danger"] = "warning"
        cp_df.loc[cp_df["pole_angle"].abs() > 0.15, "pole_danger"] = "critical"

        # Episode outcome
        cp_df["episode_outcome"] = np.where(
            cp_df["episode_length"] >= 200, "survived", "fell"
        )

        # Q-values from trained model
        import torch as _torch

        _q_vals, _max_qs = [], []
        with _torch.no_grad():
            for _, _r in cp_df.iterrows():
                _s = np.array(_r["state"], dtype=np.float32).reshape(1, -1)
                _ot, _ = cartpole_model.policy.obs_to_tensor(_s)
                _qa = cartpole_model.q_net(_ot).cpu().numpy()[0]
                _q_vals.append(float(_qa[int(_r["action"])]))
                _max_qs.append(float(np.max(_qa)))

        cp_df["q_value"] = _q_vals
        cp_df["max_q_value"] = _max_qs
        cp_df["advantage"] = cp_df["q_value"] - cp_df["max_q_value"]

        # Q-value categories
        _t33, _t67 = np.percentile(_q_vals, [33, 67])
        cp_df["q_category"] = pd.cut(
            cp_df["q_value"],
            bins=[-np.inf, _t33, _t67, np.inf],
            labels=["low_q", "mid_q", "high_q"],
        ).astype(str)

        # TD error
        cp_df = compute_td_error(cp_df, cartpole_model, gamma=0.99)
        _td_66, _td_90 = np.percentile(cp_df["td_error"], [66, 90])
        cp_df["td_category"] = "expected"
        cp_df.loc[cp_df["td_error"] > _td_66, "td_category"] = "surprising"
        cp_df.loc[cp_df["td_error"] > _td_90, "td_category"] = "very_surprising"

        # Skill level from episode length quartiles
        _ep_lens = cp_df["episode_length"]
        _q25, _q75 = np.percentile(_ep_lens[_ep_lens > 0], [25, 75])
        cp_df["skill_level"] = "intermediate"
        cp_df.loc[cp_df["episode_length"] <= _q25, "skill_level"] = "novice"
        cp_df.loc[cp_df["episode_length"] >= _q75, "skill_level"] = "expert"

        # Text description for hover
        cp_df["state_description"] = format_state_description(
            cp_df,
            component_cols=["cart_position", "pole_angle"],
            action_col="action_name",
            q_col="q_value",
            reward_col="reward",
        )

        cartpole_enriched = cp_df
        print(f"Enriched {len(cartpole_enriched)} transitions with metadata.")
    else:
        cartpole_enriched = None
        print("Skipped — data not available.")
    return (cartpole_enriched,)


@app.cell(hide_code=True)
async def _(
    async_compute_projection,
    cartpole_enriched,
    prepare_rl_data_for_projection,
):
    """UMAP projection of CartPole (state, action) vectors."""
    cartpole_proj = cartpole_enriched.copy()
    cartpole_proj, _vec = prepare_rl_data_for_projection(
        cartpole_proj,
        observation_columns=["state"],
        action_columns=["action"],
    )
    # Computes UMAP projections
    cartpole_proj = await async_compute_projection(
        cartpole_proj,
        inputs=_vec,
        modality="vector",
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
        umap_args={"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"},
    )

    cartpole_proj = cartpole_proj.drop(
        columns=[c for c in [_vec, "state", "next_state"] if c in cartpole_proj.columns]
    )
    print(f"CartPole projection ready: {len(cartpole_proj)} points")
    return (cartpole_proj,)


@app.cell
def _(EmbeddingAtlasWidget, cartpole_proj, mo):
    if cartpole_proj is not None:
        cartpole_widget = EmbeddingAtlasWidget(
            cartpole_proj,
            x="projection_x",
            y="projection_y",
            neighbors="neighbors",
            text="state_description",
            labels="automatic",
            show_charts=True,
            show_table=True,
            point_size=2.0,
        )
    else:
        cartpole_widget = None
        mo.md("Install SB3 + umap-learn to see this widget.")
    return (cartpole_widget,)


@app.cell
def _(cartpole_widget):
    cartpole_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Activation View

    The projection above uses the raw 4-D state as input. Clusters reflect
    physical similarity (similar cart positions, pole angles).

    Below we re-project using the penultimate-layer activations of the trained
    Q-network instead.
    """)
    return


@app.cell(hide_code=True)
async def _(
    add_activation_column,
    async_compute_projection,
    cartpole_enriched,
    cartpole_model,
    prepare_rl_data_for_projection,
):
    """Re-project CartPole using Q-network penultimate activations."""
    cp_act = cartpole_enriched.copy()
    cp_act = add_activation_column(cp_act, cartpole_model, layer="auto")


    cp_act, _vec = prepare_rl_data_for_projection(
        cp_act,
        observation_columns=["activations"],
        action_columns=None,
    )

    print(cp_act.head)

    print(f"{_vec=}")

    # Compute UMAP projection
    # 64D Q-Network Hidden State
    cp_act = await async_compute_projection(
        cp_act,
        inputs=_vec,
        modality="vector",
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
        umap_args={"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"},
    )
    cp_act = cp_act.drop(
        columns=[c for c in [_vec, "state", "next_state", "activations"] if c in cp_act.columns]
    )

    print(f"CartPole activation projection ready: {len(cp_act)} points")
    return (cp_act,)


@app.cell
def _(EmbeddingAtlasWidget, cp_act):
    cartpole_act_widget = EmbeddingAtlasWidget(
        cp_act,
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
        text="state_description",
        labels="automatic",
        show_charts=True,
        show_table=True,
        point_size=2.0,
    )
    return (cartpole_act_widget,)


@app.cell
def _(cartpole_act_widget):
    cartpole_act_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## LunarLander DQN

    ### Environment Inputs

    1. x position
    2. y position
    3. x velocity
    4. y velocity
    5. lander angle
    6. angular velocity
    7. left leg contact (0 or 1)
    8. right leg contact (0 or 1)
    """)
    return


@app.cell(hide_code=True)
def _(DQN, gym, pd):
    """Train LunarLander DQN for 200k steps and extract replay buffer."""
    LUNAR_STEPS = 200_000
    LUNAR_BUFFER = LUNAR_STEPS + 2000

    lunar_env = gym.make("LunarLander-v3")
    lunar_model = DQN(
        "MlpPolicy",
        lunar_env,
        learning_rate=5e-4,
        buffer_size=LUNAR_BUFFER,
        learning_starts=2000,
        batch_size=128,
        gamma=0.99, # discount factor
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        target_update_interval=500,
        verbose=0,
        seed=42,
    )
    lunar_model.learn(total_timesteps=LUNAR_STEPS, progress_bar=True)
    lunar_env.close()

    _buf = lunar_model.replay_buffer
    _n = _buf.pos

    _rows = [
        {
            "state": _buf.observations[i].flatten().tolist(),
            "action": int(_buf.actions[i].flat[0]),
            "reward": float(_buf.rewards[i].flat[0]),
            "next_state": _buf.next_observations[i].flatten().tolist(),
            "done": bool(_buf.dones[i].flat[0]),
            "timestep": i,
        }
        for i in range(_n)
    ]
    print(_rows[0])
    lunar_df = pd.DataFrame(_rows)
    print(f"Extracted {len(lunar_df)} LunarLander transitions.")
    return lunar_df, lunar_model


@app.cell(hide_code=True)
def _(
    compute_episode_metadata,
    compute_td_error,
    format_state_description,
    lunar_df,
    lunar_model,
    np,
    pd,
    unpack_state_components,
):
    """Build rich metadata for LunarLander replay buffer."""
    if lunar_df is not None:
        ll_df = lunar_df.copy()

        # Unpack 8D state
        ll_df = unpack_state_components(
            ll_df,
            "state",
            [
                "pos_x",
                "pos_y",
                "vel_x",
                "vel_y",
                "angle",
                "angular_vel",
                "left_leg",
                "right_leg",
            ],
        )

        # Derived physics columns
        ll_df["altitude"] = ll_df["pos_y"]
        ll_df["speed"] = np.sqrt(ll_df["vel_x"] ** 2 + ll_df["vel_y"] ** 2)

        # Action names
        LUNAR_ACTIONS = {
            0: "noop",
            1: "left_engine",
            2: "main_engine",
            3: "right_engine",
        }
        ll_df["action_name"] = ll_df["action"].map(LUNAR_ACTIONS)

        # Episode metadata
        ll_df = compute_episode_metadata(ll_df, done_col="done", reward_col="reward")

        # Training stage
        _n = len(ll_df)
        ll_df["training_stage"] = pd.cut(
            ll_df["timestep"],
            bins=[-1, _n // 3, 2 * _n // 3, _n],
            labels=["early", "mid", "late"],
        ).astype(str)

        # Flight phase based on altitude
        ll_df["flight_phase"] = "approach"
        ll_df.loc[ll_df["pos_y"] > 0.5, "flight_phase"] = "high_altitude"
        ll_df.loc[ll_df["pos_y"] < 0.1, "flight_phase"] = "landing"

        # Reward categories
        ll_df["reward_category"] = pd.cut(
            ll_df["reward"],
            bins=[-np.inf, -0.5, 0, 0.5, np.inf],
            labels=["high_penalty", "small_penalty", "small_reward", "high_reward"],
        ).astype(str)

        # Landing outcome (for terminal states)
        ll_df["landing_outcome"] = "in_flight"
        _terminal = ll_df["done"] == True
        ll_df.loc[_terminal & (ll_df["reward"] > 50), "landing_outcome"] = "landed"
        ll_df.loc[_terminal & (ll_df["reward"] < -50), "landing_outcome"] = "crashed"

        # Q-values
        import torch as _torch

        _q_vals = []
        with _torch.no_grad():
            for _, _r in ll_df.iterrows():
                _s = np.array(_r["state"], dtype=np.float32).reshape(1, -1)
                _ot, _ = lunar_model.policy.obs_to_tensor(_s)
                _qa = lunar_model.q_net(_ot).cpu().numpy()[0]
                _q_vals.append(float(_qa[int(_r["action"])]))
        ll_df["q_value"] = _q_vals

        # TD error
        ll_df = compute_td_error(ll_df, lunar_model, gamma=0.99)

        # Text description for hover
        ll_df["state_description"] = format_state_description(
            ll_df,
            component_cols=["pos_x", "pos_y", "speed", "angle"],
            action_col="action_name",
            q_col="q_value",
            reward_col="reward",
        )

        lunar_enriched = ll_df
        print(f"Enriched {len(lunar_enriched)} LunarLander transitions.")
    else:
        lunar_enriched = None
        print("Skipped — data not available.")
    return (lunar_enriched,)


@app.cell(hide_code=True)
async def _(
    async_compute_projection,
    lunar_enriched,
    prepare_rl_data_for_projection,
):
    """UMAP projection of LunarLander (state, action) vectors."""
    lunar_proj = lunar_enriched.copy()
    lunar_proj, _vec = prepare_rl_data_for_projection(
        lunar_proj,
        observation_columns=["state"],
        action_columns=["action"],
    )
    lunar_proj = await async_compute_projection(
        lunar_proj,
        inputs=_vec,
        modality="vector",
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
        umap_args={"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"},
    )
    lunar_proj = lunar_proj.drop(
        columns=[c for c in [_vec, "state", "next_state"] if c in lunar_proj.columns]
    )
    print(f"LunarLander projection ready: {len(lunar_proj)} points")
    return (lunar_proj,)


@app.cell
def _(EmbeddingAtlasWidget, lunar_proj):
    lunar_widget = EmbeddingAtlasWidget(
        lunar_proj,
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
        text="state_description",
        labels="disabled",
        show_charts=True,
        show_table=True,
        point_size=2.0,
        trajectory_id_field="episode",
        trajectories={
            "group_by": "episode",
            "order_by": "step_in_episode",
            "color_by": "episode",
            "colors": {
                "landed": "#16a34a",
                "crashed": "#dc2626",
                "in_flight": "#64748b",
            },
            "max_groups": 50,
            "width": 1,
            "opacity": 0.020,
        },
    )
    return (lunar_widget,)


@app.cell
def _(lunar_widget):
    lunar_widget
    return


@app.cell(hide_code=True)
async def _(
    add_activation_column,
    async_compute_projection,
    lunar_enriched,
    lunar_model,
    prepare_rl_data_for_projection,
):
    """Re-project LunarLander using Q-network penultimate activations."""
    ll_act = lunar_enriched.copy()
    ll_act = add_activation_column(ll_act, lunar_model, layer="auto")
    ll_act, _vec = prepare_rl_data_for_projection(
        ll_act,
        observation_columns=["activations"],
        action_columns=None,
    )
    ll_act = await async_compute_projection(
        ll_act,
        inputs=_vec,
        modality="vector",
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
        umap_args={"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"},
    )
    ll_act = ll_act.drop(
        columns=[c for c in [_vec, "state", "next_state", "activations"] if c in ll_act.columns]
    )
    print(f"LunarLander activation projection ready: {len(ll_act)} points")
    return (ll_act,)


@app.cell
def _(EmbeddingAtlasWidget, ll_act):
    lunar_act_widget = EmbeddingAtlasWidget(
        ll_act,
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
        text="state_description",
        labels="disabled",
        show_charts=True,
        show_table=True,
        point_size=2.0,
        trajectory_id_field="episode",
        trajectories={
            "group_by": "episode",
            "order_by": "step_in_episode",
            "color_by": "episode",
            "colors": {
                "landed": "#16a34a",
                "crashed": "#dc2626",
                "in_flight": "#64748b",
            },
            "max_groups": 50,
            "width": 1,
            "opacity": 0.020,
        },
    )
    return (lunar_act_widget,)


@app.cell
def _(lunar_act_widget):
    lunar_act_widget
    return


if __name__ == "__main__":
    app.run()
