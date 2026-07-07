"""
Language-Model Post-Training — Embedding Atlas
==============================================
Visualizing **how SFT and RL move a language model in fundamentally different
directions**, using Embedding Atlas.

Core idea
---------
A language model is a *function* that maps a prompt to a probability
distribution over its vocabulary. If we freeze a fixed set of probe prompts and
record the model's next-token distribution on each of them, we get a single
high-dimensional fingerprint — a **point in vocabulary-probability space**.

Run that probe set on every checkpoint of a training run and each checkpoint
becomes a point. Connect the points in training order and you get a *trajectory*
that shows how post-training drags the model through probability space.

Here we post-train ``distilgpt2`` toward positive-sentiment movie reviews two
different ways and compare the trajectories:

  1. **SFT** — supervised fine-tuning (cross-entropy on positive reviews).
  2. **RL**  — REINFORCE with a sentiment reward and a KL-to-base penalty
     (the canonical RLHF objective).

Both runs start from the same base model (the shared origin of probability
space). Using the base model as a coordinate frame, we measure how far each
checkpoint has drifted (KL divergence in probability space) and plot the two
trajectories.

What you should see
-------------------
  - **SFT shoots far away** from the base model — supervised cross-entropy
    reshapes the whole next-token distribution, inducing a large shift.
  - **RL stays in a tight neighborhood** of the initialization — the KL penalty
    keeps the policy close to base while the reward only re-weights tokens the
    model already considered plausible.

Seeing the two trajectories side-by-side makes the SFT-vs-RL difference
immediately intuitive.

Run with:
    marimo edit lm_post_training.py

Prerequisites:
    pip install transformers torch umap-learn matplotlib
    (all already in the embedding-atlas backend environment)

Notes:
    The first run trains both checkpoints (~3-6 min on CPU for distilgpt2) and
    caches the resulting fingerprints to a parquet file next to the notebook, so
    re-running picks up the cache instead of retraining.
"""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full", app_title="LM Post-Training — Embedding Atlas")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    return mo, np, pd


@app.cell
def _():
    import math
    import os
    import re

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from embedding_atlas.widget import EmbeddingAtlasWidget
    from embedding_atlas.projection import async_compute_projection

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        EmbeddingAtlasWidget,
        async_compute_projection,
        os,
        plt,
        re,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Language-Model Post-Training in Probability Space

    A language model is a function $f_\theta$ that turns a prompt into a
    probability distribution over the vocabulary. If we fix a small battery of
    **probe prompts** and record the model's next-token distribution on each
    one, we get a single high-dimensional vector — the model's **fingerprint**,
    i.e. a *point in vocabulary-probability space*.

    Every checkpoint of a training run produces one such point. Connecting the
    points in training order yields a **trajectory** through probability space.
    Using the frozen base model as the coordinate origin, we can measure *how
    far* and *in which direction* fine-tuning moves the model.

    Below we post-train `distilgpt2` toward positive-sentiment movie reviews
    two ways — **SFT** (supervised cross-entropy) and **RL** (REINFORCE with a
    sentiment reward + KL-to-base penalty) — and watch the trajectories diverge.
    """)
    return


@app.cell(hide_code=True)
def _(torch):
    """Configuration: model, probe/training prompts, reward lexicon, hyperparameters."""
    import random

    MODEL_NAME = "distilgpt2"
    # CPU by default: distilgpt2 is tiny, and Apple `mps` sampling can emit
    # NaN logits during RL rollouts. Set to "cuda" if you have a GPU.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    CACHE_PARQUET = "lm_post_training_fingerprints.parquet"

    # ----- Probe prompts: held-out contexts where we read the model's -----
    # ----- next-token distribution. These define the axes of prob-space. -----
    PROBE_PROMPTS = [
        "This movie was",
        "The film",
        "I watched this and",
        "The acting was",
        "Overall, the story",
        "In my opinion the movie",
        "The plot",
        "This was",
        "After seeing it I",
        "The characters were",
        "Honestly, the film",
        "By the end I",
        "The director",
        "The ending was",
        "I would say this film",
        "The cinematography was",
        "My impression of the movie",
        "The screenplay",
        "Watching this film",
        "The performances were",
        "To be honest the movie",
        "The first half",
        "The soundtrack",
        "When the credits rolled I",
        "The pacing of the film",
        "As a whole the movie",
        "The lead actor",
        "The dialogue was",
        "I left the theater",
        "This is a film that",
        "The visual effects",
        "The storyline",
        "Comparing it to others, the movie",
        "The atmosphere of the film",
        "It is a movie",
        "The whole experience was",
    ]

    # ----- Prompts the RL policy generates continuations from (same style). -----
    RL_PROMPTS = PROBE_PROMPTS
    SAMPLE_PROMPT = "This movie was"

    # ----- Sentiment reward lexicon (transparent, fully offline). -----
    POS_WORDS = [
        "wonderful", "brilliant", "amazing", "fantastic", "great", "love",
        "loved", "excellent", "beautiful", "superb", "masterpiece", "enjoyable",
        "delightful", "perfect", "best", "captivating", "charming", "stunning",
        "gripping", "heartwarming", "hilarious", "impressive", "incredible",
        "magnificent", "outstanding", "remarkable", "terrific", "touching",
        "joy", "happy", "good", "awesome", "fun", "engaging", "enjoyed",
        "fascinating", "memorable", "beautifully", "powerful", "moving",
    ]
    NEG_WORDS = [
        "terrible", "awful", "boring", "bad", "worst", "horrible", "dull",
        "disappointing", "waste", "hate", "hated", "poor", "stupid", "lame",
        "mediocre", "bland", "forgettable", "painful", "annoying", "ugly",
        "weak", "tedious", "predictable", "mess", "failure", "worse",
        "disgusting", "cringe", "sad", "dreadful", "pointless", "ridiculous",
    ]

    # ----- Build a small positive-review corpus for SFT (templated, offline). -----
    _rng = random.Random(SEED)
    _adjs = [
        "wonderful", "brilliant", "amazing", "fantastic", "superb", "delightful",
        "captivating", "stunning", "gripping", "heartwarming", "hilarious",
        "magnificent", "outstanding", "remarkable", "touching", "beautiful",
        "engaging", "memorable", "powerful", "moving",
    ]
    _nouns = [
        "film", "movie", "story", "drama", "comedy", "thriller", "picture",
        "adventure", "tale", "performance",
    ]
    _aspects = [
        "acting", "story", "cinematography", "soundtrack", "direction",
        "screenplay", "cast", "pacing", "dialogue", "ending",
    ]
    _templates = [
        "This {noun} was absolutely {adj} and truly {adj2}.",
        "I loved this {noun}; the {aspect} was {adj} and the {aspect2} was {adj2}.",
        "What a {adj} {noun} — every scene felt {adj2} and the {aspect} was perfect.",
        "An absolutely {adj} {noun} with a {adj2} plot and {adj3} characters.",
        "The {aspect} was {adj}, the {aspect2} was {adj2}, and the whole {noun} was a joy.",
        "I enjoyed every minute of this {adj} {noun}; it was {adj2} from start to finish.",
        "A {adj} and {adj2} {noun} that left me feeling happy and moved.",
        "Easily one of the best {noun}s I have seen — {adj}, {adj2}, and deeply touching.",
    ]
    SFT_CORPUS = []
    for _ in range(280):
        _t = _rng.choice(_templates)
        SFT_CORPUS.append(
            _t.format(
                noun=_rng.choice(_nouns),
                adj=_rng.choice(_adjs),
                adj2=_rng.choice(_adjs),
                adj3=_rng.choice(_adjs),
                aspect=_rng.choice(_aspects),
                aspect2=_rng.choice(_aspects),
            )
        )

    # ----- Hyperparameters. -----
    N_SFT_STEPS = 28      # gradient steps (one checkpoint each)
    N_RL_STEPS = 28
    LR_SFT = 5e-5
    LR_RL = 3e-5
    KL_BETA = 0.2         # KL-to-base penalty weight for RL (the RLHF anchor)
    GEN_LEN = 24          # tokens generated per RL rollout / sample
    SFT_BATCH = 8
    RL_SEQS_PER_STEP = 16
    TOPK = 40             # vocab tokens kept per probe for the fingerprint vector

    print(f"Device: {DEVICE} | probes: {len(PROBE_PROMPTS)} | SFT corpus: {len(SFT_CORPUS)}")
    return (
        CACHE_PARQUET,
        DEVICE,
        GEN_LEN,
        KL_BETA,
        LR_RL,
        LR_SFT,
        MODEL_NAME,
        NEG_WORDS,
        N_RL_STEPS,
        N_SFT_STEPS,
        POS_WORDS,
        PROBE_PROMPTS,
        RL_PROMPTS,
        RL_SEQS_PER_STEP,
        SAMPLE_PROMPT,
        SEED,
        SFT_BATCH,
        SFT_CORPUS,
        TOPK,
    )


@app.cell
def _(np, re, torch):
    """Pure helpers: seeding, reward, probability fingerprints, log-probs, generation."""

    def set_seed(seed):
        import random as _random

        _random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    _word_re = re.compile(r"[a-z']+")

    def sentiment_reward(text, pos_words, neg_words):
        """Lexicon sentiment: (#positive words) - (#negative words) in the text."""
        toks = _word_re.findall(text.lower())
        pos = sum(1 for t in toks if t in pos_words)
        neg = sum(1 for t in toks if t in neg_words)
        return float(pos - neg)

    def full_vocab_probs(model, tokenizer, prompts, device):
        """Next-token distribution at the last real token for each prompt.

        Returns an (n_prompts, vocab_size) float array. Right padding keeps the
        position ids of real tokens correct, so the last-real-token logits are
        exact.
        """
        enc = tokenizer(prompts, return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits  # (B, L, V)
        last_idx = enc["attention_mask"].sum(dim=1) - 1  # (B,)
        b = logits.shape[0]
        last_logits = logits[torch.arange(b, device=device), last_idx]  # (B, V)
        probs = torch.softmax(last_logits.float(), dim=-1).cpu().numpy()
        return probs

    def kl_to_base(cur_probs, base_probs, eps=1e-8):
        """Mean over probes of KL(cur || base) (nats), measured on the full vocab."""
        c = cur_probs + eps
        b = base_probs + eps
        kl = np.sum(c * (np.log(c) - np.log(b)), axis=1)  # (n_probes,)
        return float(np.clip(kl, 0.0, None).mean())

    def l2_to_base(cur_probs, base_probs):
        """Mean L2 distance in probability space across probes."""
        return float(np.linalg.norm(cur_probs - base_probs, axis=1).mean())

    def make_fingerprint(cur_probs, topk_idx):
        """Reduce (n_probes, V) probs to the per-probe top-K base tokens, flattened."""
        n = cur_probs.shape[0]
        reduced = cur_probs[np.arange(n)[:, None], topk_idx]  # (n_probes, K)
        return reduced.flatten().astype(np.float32)

    @torch.no_grad()
    def generate_one(model, tokenizer, prompt, device, gen_len, sample=False):
        """Greedy (or sampled) continuation, returned as decoded text."""
        ids = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **ids,
            max_new_tokens=gen_len,
            do_sample=sample,
            top_k=50 if sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen = out[0, ids["input_ids"].shape[1]:]
        return tokenizer.decode(gen, skip_special_tokens=True).strip()

    def seq_logprob_sum(model, full_ids, prompt_len):
        """Sum of per-token log-probs over the *generated* positions (no padding).

        Returns a scalar tensor (keeps grad if ``model`` params require grad).
        """
        logits = model(full_ids).logits  # (1, L, V)
        logp = torch.log_softmax(logits.float(), dim=-1)
        targets = full_ids[:, 1:]  # (1, L-1)
        tok_logp = logp[:, :-1].gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (1, L-1)
        gen_logp = tok_logp[:, prompt_len - 1:]  # (1, gen_len)
        return gen_logp.sum(dim=1).squeeze(0)

    return (
        full_vocab_probs,
        generate_one,
        kl_to_base,
        l2_to_base,
        make_fingerprint,
        sentiment_reward,
        seq_logprob_sum,
        set_seed,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train both checkpoints and fingerprint every step

    The cell below runs the full pipeline (cached to parquet after the first
    run):

    1. Load the frozen **base** `distilgpt2`, read its next-token distribution
       on the probe prompts, and pick the per-probe top-K vocabulary tokens —
       these fixed indices define the axes of our probability-space fingerprint.
    2. **SFT run:** cross-entropy on the positive-review corpus. After every
       gradient step we record the fingerprint, the KL divergence to base, and
       a greedy sample.
    3. **RL run:** REINFORCE with the sentiment reward and a KL-to-base penalty.
       Same per-step bookkeeping.

    Both runs start from a step-0 row whose fingerprint is the base model, so
    each trajectory emanates from the shared origin.
    """)
    return


@app.cell(hide_code=True)
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    CACHE_PARQUET,
    DEVICE,
    GEN_LEN,
    KL_BETA,
    LR_RL,
    LR_SFT,
    MODEL_NAME,
    NEG_WORDS,
    N_RL_STEPS,
    N_SFT_STEPS,
    POS_WORDS,
    PROBE_PROMPTS,
    RL_PROMPTS,
    RL_SEQS_PER_STEP,
    SAMPLE_PROMPT,
    SEED,
    SFT_BATCH,
    SFT_CORPUS,
    TOPK,
    full_vocab_probs,
    generate_one,
    kl_to_base,
    l2_to_base,
    make_fingerprint,
    np,
    os,
    pd,
    sentiment_reward,
    seq_logprob_sum,
    set_seed,
    torch,
):
    """Build (or load cached) per-checkpoint fingerprints for the SFT and RL runs."""

    def _load_model():
        m = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
        return m

    def _build():
        set_seed(SEED)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # ---- Frozen base model + probe distributions ----
        base_model = _load_model()
        base_model.eval()
        for p in base_model.parameters():
            p.requires_grad_(False)

        base_probs = full_vocab_probs(base_model, tokenizer, PROBE_PROMPTS, DEVICE)
        # Per-probe top-K base tokens fix the fingerprint axes for all checkpoints.
        topk_idx = np.argsort(-base_probs, axis=1)[:, :TOPK]  # (n_probes, K)
        base_fp = make_fingerprint(base_probs, topk_idx)
        base_sample = generate_one(base_model, tokenizer, SAMPLE_PROMPT, DEVICE, GEN_LEN)
        base_reward = sentiment_reward(base_sample, POS_WORDS, NEG_WORDS)
        print(f"Base fingerprint dim: {base_fp.shape[0]} | base sample: {base_sample!r}")

        rows = []

        def _record(run_id, method, step, model):
            cur = full_vocab_probs(model, tokenizer, PROBE_PROMPTS, DEVICE)
            sample = generate_one(model, tokenizer, SAMPLE_PROMPT, DEVICE, GEN_LEN)
            reward = sentiment_reward(sample, POS_WORDS, NEG_WORDS)
            kl = kl_to_base(cur, base_probs)
            l2 = l2_to_base(cur, base_probs)
            fp = make_fingerprint(cur, topk_idx)
            label = (
                f"{method.upper()} step {step} | KL→base={kl:.3f} | "
                f"reward={reward:.1f} | {SAMPLE_PROMPT}{sample!r}"
            )
            rows.append(
                {
                    "run_id": run_id,
                    "method": method,
                    "step": int(step),
                    "kl_to_base": kl,
                    "l2_to_base": l2,
                    "sample_reward": reward,
                    "sample_generation": sample,
                    "label": label,
                    "fingerprint": fp.tolist(),
                }
            )

        def _record_base_as(run_id, method):
            rows.append(
                {
                    "run_id": run_id,
                    "method": method,
                    "step": 0,
                    "kl_to_base": 0.0,
                    "l2_to_base": 0.0,
                    "sample_reward": base_reward,
                    "sample_generation": base_sample,
                    "label": f"BASE (origin) | reward={base_reward:.1f} | "
                    f"{SAMPLE_PROMPT}{base_sample!r}",
                    "fingerprint": base_fp.tolist(),
                }
            )

        # Standalone base point.
        _record_base_as("base", "base")

        # ---- SFT run ----
        print(f"SFT: {N_SFT_STEPS} steps...")
        set_seed(SEED)
        sft_model = _load_model()
        sft_model.train()
        sft_opt = torch.optim.AdamW(sft_model.parameters(), lr=LR_SFT)
        _record_base_as("sft", "sft")  # step 0 = base
        for step in range(1, N_SFT_STEPS + 1):
            batch = list(np.random.choice(SFT_CORPUS, size=SFT_BATCH, replace=False))
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=48,
            ).to(DEVICE)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            out = sft_model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=labels,
            )
            sft_opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(sft_model.parameters(), 1.0)
            sft_opt.step()
            sft_model.eval()
            _record("sft", "sft", step, sft_model)
            sft_model.train()
        del sft_model, sft_opt

        # ---- RL run: REINFORCE + KL-to-base penalty ----
        print(f"RL: {N_RL_STEPS} steps...")
        set_seed(SEED)
        rl_model = _load_model()
        # Eval mode throughout: disables dropout so the REINFORCE log-probs match
        # the sampling distribution. Gradients still flow in eval mode.
        rl_model.eval()
        rl_opt = torch.optim.AdamW(rl_model.parameters(), lr=LR_RL)
        _record_base_as("rl", "rl")  # step 0 = base
        for step in range(1, N_RL_STEPS + 1):
            prompts = list(
                np.random.choice(RL_PROMPTS, size=RL_SEQS_PER_STEP, replace=True)
            )
            seq_logps = []
            rewards = []
            ref_sums = []
            for prompt in prompts:
                ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
                prompt_len = ids["input_ids"].shape[1]
                with torch.no_grad():
                    gen = rl_model.generate(
                        **ids,
                        max_new_tokens=GEN_LEN,
                        do_sample=True,
                        top_k=50,
                        temperature=1.0,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                full_ids = gen  # (1, prompt_len + gen_len)
                text = tokenizer.decode(
                    full_ids[0, prompt_len:], skip_special_tokens=True
                )
                rewards.append(sentiment_reward(text, POS_WORDS, NEG_WORDS))
                # Policy log-prob (with grad) and frozen-base log-prob (no grad).
                lp = seq_logprob_sum(rl_model, full_ids, prompt_len)
                with torch.no_grad():
                    rp = seq_logprob_sum(base_model, full_ids, prompt_len)
                seq_logps.append(lp)
                ref_sums.append(float(rp))

            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
            adv = rewards_t - rewards_t.mean()
            logp_stack = torch.stack(seq_logps)  # (B,) with grad
            ref_stack = torch.tensor(ref_sums, dtype=torch.float32, device=DEVICE)
            pg_loss = -(adv.detach() * logp_stack).mean()
            kl_loss = (logp_stack - ref_stack).mean()  # E[log pi - log ref]
            loss = pg_loss + KL_BETA * kl_loss

            rl_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rl_model.parameters(), 1.0)
            rl_opt.step()
            _record("rl", "rl", step, rl_model)
        del rl_model, rl_opt, base_model

        df = pd.DataFrame(rows)
        df.to_parquet(CACHE_PARQUET)
        print(f"Saved {len(df)} checkpoint fingerprints to {CACHE_PARQUET}")
        return df

    if os.path.exists(CACHE_PARQUET):
        print(f"Loading cached fingerprints from {CACHE_PARQUET}")
        fingerprint_df = pd.read_parquet(CACHE_PARQUET)
        fingerprint_df["fingerprint"] = fingerprint_df["fingerprint"].apply(
            lambda v: np.asarray(v, dtype=np.float32)
        )
        print(f"Loaded {len(fingerprint_df)} checkpoint fingerprints.")
    else:
        fingerprint_df = _build()

    print(
        fingerprint_df.groupby("method")["kl_to_base"].agg(["count", "mean", "max"])
    )
    return (fingerprint_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reference-frame view — base model as the origin

    We use the base model as the coordinate frame, exactly as in the
    "models as functions" framing:

    - **x-axis** points from the base toward the *final SFT* checkpoint
      (the direction supervised fine-tuning travels).
    - **y-axis** is the component of the *final RL* checkpoint orthogonal to
      the SFT direction (how RL departs from the SFT axis).

    Every checkpoint is projected onto these two axes, so the base sits at the
    origin and distance from the origin is literal distance in probability
    space. The two trajectories make the contrast obvious: **SFT races out
    along x, RL barely leaves the origin.**
    """)
    return


@app.cell
def _(fingerprint_df, np):
    """Project fingerprints onto a base→SFT / base→RL reference frame."""
    _df = fingerprint_df.reset_index(drop=True)
    _M = np.vstack(_df["fingerprint"].to_numpy()).astype(np.float64)

    _base_vec = _M[_df.index[_df["method"] == "base"][0]]

    def _final_vec(run_id):
        mask = _df["run_id"] == run_id
        sub = _df[mask]
        idx = sub.index[int(np.argmax(sub["step"].to_numpy()))]
        return _M[idx]

    _sft_final = _final_vec("sft")
    _rl_final = _final_vec("rl")

    _u = _sft_final - _base_vec
    _u = _u / (np.linalg.norm(_u) + 1e-12)
    _w = _rl_final - _base_vec
    _w = _w - (_w @ _u) * _u
    _v = _w / (np.linalg.norm(_w) + 1e-12)

    _D = _M - _base_vec
    refframe_df = _df.drop(columns=["fingerprint"]).copy()
    refframe_df["x"] = _D @ _u
    refframe_df["y"] = _D @ _v
    print("Reference-frame coordinates ready.")
    print(refframe_df.groupby("method")[["x", "y"]].mean())
    return (refframe_df,)


@app.cell
def _(EmbeddingAtlasWidget, refframe_df):
    refframe_widget = EmbeddingAtlasWidget(
        refframe_df,
        x="x",
        y="y",
        text="label",
        color="method",
        labels="disabled",
        show_charts=True,
        show_table=True,
        point_size=8.0,
        trajectory_id_field="run_id",
        trajectories={
            "group_by": "run_id",
            "order_by": "step",
            "color_by": "method",
            "colors": {
                "base": "#111827",
                "sft": "#dc2626",
                "rl": "#2563eb",
            },
            "max_groups": 10,
            "width": 2,
            "opacity": 0.9,
        },
    )
    return (refframe_widget,)


@app.cell
def _(refframe_widget):
    refframe_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## UMAP view — the probability-space manifold

    The same fingerprints, projected with UMAP (the standard Embedding Atlas
    pipeline). UMAP emphasizes local neighborhood structure rather than
    absolute magnitude, so here you can see how the SFT checkpoints form their
    own arc while the RL checkpoints stay bunched near the base point. Hover any
    point to read its KL-to-base, reward, and a greedy sample.
    """)
    return


@app.cell
async def _(async_compute_projection, fingerprint_df):
    """UMAP projection of the checkpoint fingerprints."""
    umap_df = fingerprint_df.copy()
    umap_df = await async_compute_projection(
        umap_df,
        inputs="fingerprint",
        modality="vector",
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
        umap_args={"n_neighbors": 10, "min_dist": 0.3, "metric": "cosine"},
    )
    umap_df = umap_df.drop(columns=["fingerprint"])
    print(f"UMAP projection ready: {len(umap_df)} checkpoints.")
    return (umap_df,)


@app.cell
def _(EmbeddingAtlasWidget, umap_df):
    umap_widget = EmbeddingAtlasWidget(
        umap_df,
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
        text="label",
        color="method",
        labels="disabled",
        show_charts=True,
        show_table=True,
        point_size=8.0,
        trajectory_id_field="run_id",
        trajectories={
            "group_by": "run_id",
            "order_by": "step",
            "color_by": "method",
            "colors": {
                "base": "#111827",
                "sft": "#dc2626",
                "rl": "#2563eb",
            },
            "max_groups": 10,
            "width": 2,
            "opacity": 0.9,
        },
    )
    return (umap_widget,)


@app.cell
def _(umap_widget):
    umap_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distribution shift over training

    The headline result, plotted directly: KL divergence from the base model
    (in vocabulary-probability space) at every checkpoint. **SFT climbs far
    and fast; RL stays within a tight neighborhood of the initialization** — a
    direct consequence of the KL-to-base penalty that RLHF uses.
    """)
    return


@app.cell(hide_code=True)
def _(fingerprint_df, plt):
    """Plot KL-to-base vs training step for SFT and RL."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for run_id, color, name in [("sft", "#dc2626", "SFT"), ("rl", "#2563eb", "RL")]:
        sub = fingerprint_df[fingerprint_df["run_id"] == run_id].sort_values("step")
        ax.plot(
            sub["step"],
            sub["kl_to_base"],
            marker="o",
            markersize=4,
            color=color,
            label=name,
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel("KL(checkpoint ‖ base)  [nats]")
    ax.set_title("Distribution shift from the base model: SFT vs RL")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - Treating a model as a **function from prompt to vocabulary distribution**
      turns each checkpoint into a single point, and a training run into a
      **trajectory** through probability space.
    - With the base model as the coordinate frame, the trajectories tell the
      story at a glance: **SFT induces a large distribution shift**, while
      **RL fine-tunes within a tight neighborhood** of the initialization.
    - Embedding Atlas makes this exploratory: brush the trajectories, hover to
      read each checkpoint's KL, reward, and sample, and watch how the SFT and
      RL paths diverge from the same origin.
    """)
    return


if __name__ == "__main__":
    app.run()
