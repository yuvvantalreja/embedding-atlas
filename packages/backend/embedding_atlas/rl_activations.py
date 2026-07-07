# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
"""
Project trained-network activations to embedding space.

``extract_activations`` hooks an intermediate layer of a stable-baselines3 model,
runs a batch of observations through it, and returns a ``(N, feature_dim)``
ndarray suitable for UMAP. Projecting these instead of raw ``(state, action)``
vectors yields an embedding that clusters states by what the *policy* treats as
similar rather than by raw physical similarity — the canonical Mnih-2015 /
Vizarel view.

Supported SB3 algorithms with sensible defaults for ``layer="auto"``:

================  ==========================================
Algorithm         Default hooked layer
================  ==========================================
DQN               ``model.q_net.q_net[-2]`` (penultimate ReLU)
PPO / A2C         ``model.policy.mlp_extractor.policy_net``
SAC / TD3 / DDPG  ``model.actor.latent_pi``
================  ==========================================

Pass ``layer=<torch.nn.Module>`` or a dotted path string
(e.g. ``"policy.mlp_extractor.value_net"``) to escape the defaults — useful for
custom architectures or for inspecting non-default sub-nets (e.g. the SAC critic
trunk).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _default_layer_and_forward(model) -> tuple[Any, Callable]:
    """Return (module_to_hook, forward_callable) for ``layer="auto"``."""
    name = type(model).__name__
    if name == "DQN":
        return model.q_net.q_net[-2], lambda x: model.q_net(x)
    if name in ("PPO", "A2C"):
        return model.policy.mlp_extractor.policy_net, lambda x: model.policy(x)
    if name in ("SAC", "TD3", "DDPG"):
        return model.actor.latent_pi, lambda x: model.actor(x)
    raise ValueError(
        f"No default layer for algorithm {name!r}. Pass `layer=<nn.Module>` or a "
        f"dotted-path string explicitly. Use `list_extractable_layers(model)` to see "
        f"the available paths."
    )


def _resolve_dotted_path(model, path: str):
    """Resolve ``'policy.mlp_extractor.policy_net'`` against the model object."""
    obj = model
    for part in path.split("."):
        if part.lstrip("-").isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def _forward_for_root(model, root: str) -> Callable:
    """Pick the smallest sub-net whose forward will fire a hook under ``root``."""
    if root == "q_net":
        return lambda x: model.q_net(x)
    if root == "actor":
        return lambda x: model.actor(x)
    # `policy` covers PPO/A2C and any features_extractor reachable from policy.
    return lambda x: model.policy(x)


def _resolve_layer(model, layer) -> tuple[Any, Callable]:
    """Resolve a layer spec to (module_to_hook, forward_callable)."""
    import torch.nn as nn

    if isinstance(layer, nn.Module):
        # Custom module: route through model.policy by default; users with exotic
        # setups can call extract_activations multiple times if needed.
        return layer, lambda x: model.policy(x)

    if layer == "auto":
        return _default_layer_and_forward(model)

    name = type(model).__name__

    if layer == "features_extractor":
        # The features extractor lives in different places per algorithm:
        # DQN keeps it on q_net; SAC keeps it on actor; PPO/A2C keep one on policy.
        for container_name in ("policy", "q_net", "actor"):
            container = getattr(model, container_name, None)
            fe = (
                getattr(container, "features_extractor", None)
                if container is not None
                else None
            )
            if fe is not None:
                return fe, _forward_for_root(model, container_name)
        raise ValueError(
            "Could not locate a features_extractor on policy / q_net / actor"
        )

    if layer == "q_net":
        if name != "DQN":
            raise ValueError(f"layer='q_net' is only valid for DQN, got {name}")
        return model.q_net.q_net[-2], lambda x: model.q_net(x)

    if layer == "actor":
        if name in ("PPO", "A2C"):
            return model.policy.mlp_extractor.policy_net, lambda x: model.policy(x)
        if name in ("SAC", "TD3", "DDPG"):
            return model.actor.latent_pi, lambda x: model.actor(x)
        raise ValueError(f"layer='actor' not supported for {name}")

    if isinstance(layer, str) and "." in layer:
        module = _resolve_dotted_path(model, layer)
        root = layer.split(".", 1)[0]
        return module, _forward_for_root(model, root)

    raise ValueError(
        f"Unrecognized layer spec: {layer!r}. Expected 'auto', 'actor', 'q_net', "
        f"'features_extractor', a dotted path, or a torch.nn.Module instance."
    )


def _stack_states(states) -> np.ndarray:
    """Stack a list / Series / ndarray-of-lists into a (N, *obs_shape) float32 array."""
    if isinstance(states, np.ndarray):
        return states.astype(np.float32, copy=False)
    if isinstance(states, pd.Series):
        return np.vstack(states.apply(np.asarray).to_numpy()).astype(np.float32)
    return np.vstack([np.asarray(s) for s in states]).astype(np.float32)


def extract_activations(
    model,
    states,
    *,
    layer: Any = "auto",
    batch_size: int = 4096,
    device: str | None = None,
) -> np.ndarray:
    """
    Run a batch of observations through ``model`` and return the activations at
    ``layer`` as a ``(N, feature_dim)`` ndarray.

    Parameters
    ----------
    model
        A trained stable-baselines3 model (DQN / PPO / A2C / SAC / TD3 / DDPG).
    states
        Iterable of observations. Accepts ``np.ndarray``, a ``pd.Series`` of
        list/array values, or a plain list of arrays. Each observation must be
        flat-vector-shaped (image obs not supported in this first cut).
    layer
        Layer specifier. ``"auto"`` picks an algorithm-appropriate penultimate
        layer (see module docstring). Other accepted values: ``"actor"``,
        ``"q_net"``, ``"features_extractor"``, a dotted path string, or a
        ``torch.nn.Module`` instance.
    batch_size
        Forward-pass batch size.
    device
        Torch device override. Defaults to ``str(model.device)``.
    """
    import torch

    states_arr = _stack_states(states)
    n = len(states_arr)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    module, forward_fn = _resolve_layer(model, layer)
    device = device or str(model.device)

    was_training = bool(getattr(model.policy, "training", False))
    if hasattr(model.policy, "set_training_mode"):
        model.policy.set_training_mode(False)

    buckets: list[np.ndarray] = []

    def _hook(_mod, _inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        arr = t.detach().cpu().float().numpy()
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        buckets.append(arr)

    handle = module.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = torch.as_tensor(states_arr[i : i + batch_size], device=device)
                forward_fn(batch)
    finally:
        handle.remove()
        if was_training and hasattr(model.policy, "set_training_mode"):
            model.policy.set_training_mode(True)

    if not buckets:
        raise RuntimeError(
            f"Forward hook on {type(module).__name__} never fired. The chosen layer "
            f"is not reachable from the dispatched forward — pass a layer that lives "
            f"under the algorithm's actor/policy/q_net."
        )

    result = np.concatenate(buckets, axis=0)
    if len(result) != n:
        logger.warning(
            "Activation rows (%d) != input rows (%d). The hooked module likely "
            "fires more than once per forward (e.g. a shared sub-net). Consider "
            "hooking a non-shared layer.",
            len(result),
            n,
        )
    logger.info(
        "Extracted activations: shape=%s from %s",
        result.shape,
        type(module).__name__,
    )
    return result


def add_activation_column(
    df: pd.DataFrame,
    model,
    *,
    state_col: str = "state",
    output_col: str = "activations",
    layer: Any = "auto",
    batch_size: int = 4096,
    device: str | None = None,
) -> pd.DataFrame:
    """
    Add a vector-valued column ``output_col`` to ``df`` containing the activation
    of ``model`` at ``layer`` for each row's observation in ``state_col``.

    The DataFrame is mutated in place and also returned for chaining.
    """
    activations = extract_activations(
        model,
        df[state_col],
        layer=layer,
        batch_size=batch_size,
        device=device,
    )
    df[output_col] = list(activations)
    return df


def list_extractable_layers(model) -> dict[str, str]:
    """
    Map ``path -> module-class-name`` for every named submodule of the model.
    Useful for picking a value to pass as ``layer=...``.

    Walks both ``model.policy`` and (when present) ``model.q_net``, ``model.actor``,
    and ``model.critic``.
    """
    out: dict[str, str] = {}
    roots: list[tuple[str, Any]] = []
    for attr in ("policy", "q_net", "actor", "critic"):
        sub = getattr(model, attr, None)
        if sub is not None and hasattr(sub, "named_modules"):
            roots.append((attr, sub))
    for root_name, root in roots:
        for name, mod in root.named_modules():
            if not name:
                continue
            out[f"{root_name}.{name}"] = type(mod).__name__
    return out
