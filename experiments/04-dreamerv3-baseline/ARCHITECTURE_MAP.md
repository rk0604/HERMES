# DreamerV3: paper concepts mapped to code

This document traces every piece of the DreamerV3 architecture from the paper to the
exact file and line that implements it. It exists so that the later reconstruction-free
ablation can be written as a small, precise change instead of a guess.

## How to read the citations

* **Which code.** All `dreamerv3/...` and `embodied/...` citations refer to upstream
  `github.com/danijar/dreamerv3` at commit **`e3f02248693a79dc8b0ebd62c93683888ddaccfe`**
  (2026-05-25, "Fix Atari frame maxpooling on reset"). They have **not** yet been checked
  against the HERMES fork. If the fork is at a different commit, line numbers may shift.
* **Helper libraries.** DreamerV3 depends on three small libraries by the same author
  that are not in the repo: `elements` (config, flags, checkpointing, logging), `ninjax`
  (the module system on top of JAX) and `scope` (log viewer). `requirements.txt` only
  gives lower bounds (`elements>=3.19.1`, `ninjax>=3.5.1`). Citations into them refer to
  the versions I read: **`elements==3.22.1`, `ninjax==3.6.3`, `scope==0.7.1`**. The
  notebook must pin and print the versions it actually installs.
* **Paper.** "the paper" means the Nature version in `references/` plus its arXiv
  preprint. Hyperparameter values quoted from the preprint's Table 4 hyperparameter table.
* **Confidence.** Anything I did not verify by reading the code line by line is marked
  **UNCERTAIN**, and the list at the end collects all of them.

## 1. File map

| Concept | File | Lines |
| :--- | :--- | :--- |
| Entry point, config assembly | `dreamerv3/main.py` | 19-124 |
| All hyperparameters | `dreamerv3/configs.yaml` | 1-220 |
| Agent: builds modules, world-model + actor-critic loss | `dreamerv3/agent.py` | 24-379 |
| Actor and critic losses (imagination) | `dreamerv3/agent.py` | 382-446 |
| Critic loss on replayed data | `dreamerv3/agent.py` | 449-479 |
| λ-return | `dreamerv3/agent.py` | 482-490 |
| RSSM (sequence model, posterior, prior, KL) | `dreamerv3/rssm.py` | 16-176 |
| Encoder | `dreamerv3/rssm.py` | 179-250 |
| Decoder | `dreamerv3/rssm.py` | 253-359 |
| MLP heads (reward, continue, actor, critic) | `embodied/jax/heads.py` | 16-162 |
| Output distributions and their losses | `embodied/jax/outs.py` | 11-330 |
| Optimizer (one for everything) | `embodied/jax/opt.py` | 16-81 |
| JIT/sharding wrapper, checkpoint save/load, RNG seeds | `embodied/jax/agent.py` | 36-493 |
| JAX/XLA environment setup | `embodied/jax/internal.py` | 15-109 |
| Training loop | `embodied/run/train.py` | 9-119 |
| Replay buffer | `embodied/core/replay.py`, `embodied/core/chunk.py` | |

## 2. One training step, end to end

```
replay batch  (B sequences, each T+1 steps: 1 context step + T training steps)
   │
   │  agent.py:312-340  first step restores h,z stored in replay (see §10)
   ▼
Encoder          obs x_t ──────────────────────────────► tokens e_t          rssm.py:210-250
   ▼
RSSM.observe     for t in 1..T (scan):                                        rssm.py:61-92
                   h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})       _core           rssm.py:135-159
                   z_t ~ q(z | h_t, e_t)   posterior          obslogit        rssm.py:81-87
RSSM.loss        prior logits p(z | h_t) on every h_t          _prior          rssm.py:123
                 L_dyn = max(1, KL[sg(q) || p])                               rssm.py:125,128
                 L_rep = max(1, KL[q || sg(p)])                               rssm.py:126,129
   ▼  repfeat = {deter: h_t, stoch: z_t, logit}
Decoder          (h_t, z_t) ─► x̂_t ; L_image = Σ_pixels (x̂ - x/255)²          agent.py:170-182
Reward head      (h_t, z_t) ─► r̂_t ; L_rew  (two-hot)                          agent.py:172-173
Continue head    (h_t, z_t) ─► ĉ_t ; L_con  (logistic)                          agent.py:174-177
   ▼
Imagination      start from every (h_t, z_t), roll H=15 steps on the prior:  agent.py:188-200
                   a ~ π(h, z);  h' = GRU(h, z, a);  ẑ' ~ p(z | h')          rssm.py:94-118
Actor, critic    on imagined (h, ẑ) with predicted reward and continue       agent.py:202-216
Replay critic    critic on real (h_t, z_t) with real rewards                 agent.py:218-235
   ▼
total = Σ_k  mean(L_k) × loss_scales[k]                                       agent.py:237-240
one optimizer step over ALL modules                                           agent.py:74-78, opt.py:43-63
```

`B` is batch size and `T` is batch length (`configs.yaml:10-11`, defaults 16 and 64).

## 3. Encoder

**Where.** Class `Encoder`, `dreamerv3/rssm.py:179-250`. Built at `dreamerv3/agent.py:41-43`
with config `agent.enc.simple` (`configs.yaml:94`).

**Which observations it sees.** Every observation key except `is_first`, `is_last`,
`is_terminal`, `reward` (`agent.py:38-39`). Keys starting with `log/` were already removed
in `main.py:130-131`.

**What it does to an image** (`rssm.py:226-245`):

1. Concatenate all image keys along the channel axis, cast to the compute dtype, scale to
   `[-0.5, 0.5]` (`rssm.py:228-230`).
2. Flatten batch and time together (`rssm.py:231`) so the CNN treats every frame
   independently. The encoder has no memory; `initial()` and `truncate()` return `{}`
   (`rssm.py:204-208`).
3. Four stages, channel widths `depth × mults` = `64 × [2,3,4,4]` = 128, 192, 256, 256 by
   default (`rssm.py:197`, `configs.yaml:94`). Each stage is a 5×5 convolution, then a 2×2
   max-pool (because `strided: False`, `rssm.py:238-240`), then RMSNorm and SiLU
   (`rssm.py:241`).
4. A 64×64 image therefore ends at 4×4 spatial resolution, asserted to be between 3 and 16
   (`rssm.py:242-243`), and is flattened into a vector (`rssm.py:244`). By arithmetic from
   the config (not yet observed), that is 4 × 4 × 256 = 4096 numbers at default size.

**Vector observations** (`rssm.py:215-224`) are squashed with `symlog` and pass through a
3-layer MLP. Image and vector features are concatenated into `tokens` (`rssm.py:247-248`).

**Important mismatch with the paper's naming.** The paper's "encoder" is the posterior
`q(z_t | h_t, x_t)`. In the code, that is split in two: `Encoder` only turns `x_t` into a
deterministic vector `tokens`. The part that combines it with `h_t` and produces the
stochastic `z_t` lives inside the RSSM (§4).

## 4. RSSM

**Where.** Class `RSSM`, `dreamerv3/rssm.py:16-176`, built at `agent.py:44-46` with config
`agent.dyn.rssm` (`configs.yaml:91`).

**State.** A dict with `deter` (the paper's `h`, a vector of size `deter`) and `stoch` (the
paper's `z`, one-hot tensor of shape `stoch × classes`) (`rssm.py:39-49`). Defaults:
`deter=8192`, `stoch=32`, `classes=64` (`configs.yaml:91`).

### 4a. Where `h` is updated: `_core`, `rssm.py:135-159`

`h_t = f(h_{t-1}, z_{t-1}, a_{t-1})`:

* `z_{t-1}` is flattened (`136`); actions are divided by `max(1, |a|)` (`137`).
* `h_{t-1}`, `z_{t-1}`, `a_{t-1}` each pass through their own Linear + RMSNorm + SiLU
  (`141-146`).
* This is a **block GRU**: `h` is split into `blocks=8` groups and each group gets its own
  weights (`BlockLinear`, `150`, `152`). The three gate vectors are computed at `152-157`
  and the update is `h = u·cand + (1-u)·h` (`158`). The `-1` in `sigmoid(update - 1)`
  (`157`) biases the GRU towards keeping its old state.

`_core` is called from exactly two places: the observe step (`rssm.py:80`) and the imagine
step (`rssm.py:98`).

### 4b. Posterior `q(z | h, x)`: `_observe`, `rssm.py:75-92`

* On episode boundaries (`reset = is_first`) the previous `h`, `z` and action are zeroed
  (`76-79`).
* `h_t` is computed with `_core` (`80`).
* Input to the posterior is `concat(h_t, tokens)` (`82`). The config key `absolute: True`
  would drop `h_t` and use `tokens` alone (`82`, default `False` at `configs.yaml:91`).
* `obslayers=1` Linear + norm + act (`83-85`), then `_logit('obslogit', ...)` gives logits
  of shape `stoch × classes` (`86`, `168-171`).
* `z_t` is **sampled** from that distribution (`87`).
* Over a whole sequence, `observe` runs `_observe` inside `nj.scan` (`rssm.py:69-72`).
  *scan* is JAX's compiled for-loop: it applies the same function step by step, threading
  a carry (here `h, z`) through time.

### 4c. Prior `p(z | h)`: `_prior`, `rssm.py:161-166`

Only `h` goes in: `imglayers=2` Linear + norm + act (`163-165`), then `priorlogit` (`166`).
There is no separate network for the prior during training vs imagination; it is the same
function called at `rssm.py:123` (training) and `rssm.py:99` (imagination).

### 4d. The categorical distribution: `_dist`, `rssm.py:173-176`

`OneHot` with `unimix=0.01`, i.e. 99% network softmax + 1% uniform
(`outs.py:208-217`, `outs.py:243-246`). Samples use straight-through gradients: the forward
value is a hard one-hot, the backward gradient is that of the softmax probabilities
(`outs.py:265-270`). `Agg(..., 1, sum)` sums log-probs and KLs over the 32 latent
variables (`rssm.py:175`, `outs.py:73-76`).

### 4e. Dynamics and representation losses: `RSSM.loss`, `rssm.py:120-133`

```python
prior = self._prior(feat['deter'])                        # 123: p(z|h) on posterior h's
post  = feat['logit']                                     # 124: q(z|h,x) logits
dyn = self._dist(sg(post)).kl(self._dist(prior))          # 125: trains the prior
rep = self._dist(post).kl(self._dist(sg(prior)))          # 126: trains the posterior
dyn = jnp.maximum(dyn, self.free_nats)                    # 128: free bits, 1 nat
rep = jnp.maximum(rep, self.free_nats)                    # 129
```

This matches paper Eq. (3) exactly, including the direction of both KLs and the 1-nat clip
(`free_nats: 1.0`, `configs.yaml:91`). Categorical KL is at `outs.py:236-240`.

## 5. Decoder and the reconstruction loss

**Where.** Class `Decoder`, `dreamerv3/rssm.py:253-359`, built at `agent.py:47-49` with
config `agent.dec.simple` (`configs.yaml:97`).

**Which keys it reconstructs.** `dec_space` is the same set as the encoder's: every
observation except `is_first`, `is_last`, `is_terminal`, `reward` (`agent.py:38-40`).

**What it does to an image** (`rssm.py:308-356`):

1. Input is `h` and flattened `z` (`316-319`). With `bspace=8` (default,
   `configs.yaml:97`), `h` goes through a block-linear layer straight to a
   4×4×256 feature map (`320-323`), `z` through two Linear layers to the same shape
   (`324-326`), and the two are summed (`327`).
2. Three upsampling stages: nearest-neighbour 2× repeat, then 5×5 conv, norm, act
   (`331-338`). A final repeat and conv to 3 channels (`345-348`).
3. `sigmoid` so pixels are in `[0, 1]` (`349`).
4. Wrapped as `MSE` summed over height, width and channels (`354-356`, `outs.py:129-141`).
   So the image loss per time step is the **sum** of squared pixel errors, not the mean.

**Where the reconstruction loss is computed** (`agent.py:170-182`):

```python
dec_carry, dec_entries, recons = self.dec(dec_carry, repfeat, reset, training)  # 170-171
...
for key, recon in recons.items():                                               # 178
  space, value = self.obs_space[key], obs[key]
  target = f32(value) / 255 if isimage(space) else value                        # 181
  losses[key] = recon.loss(sg(target))                                          # 182
```

Note that `repfeat` goes into the decoder **without** a stop-gradient (`170-171`), so
reconstruction gradients flow back into the RSSM and the encoder. Each loss is named after
its observation key, so for an image-only task it appears as `losses['image']` and is
logged as `train/loss/image` (`agent.py:239`).

**Where it is added to the total** (`agent.py:237-240`):

```python
metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})   # 239
loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])   # 240
```

## 6. Reward and continue predictors

Both are `embodied.jax.MLPHead` (`heads.py:16-41`): an MLP followed by an output layer.
Both take `feat2tensor(repfeat)` = `concat(h, flatten(z))` (`agent.py:51-53`).

| | Reward | Continue |
| :--- | :--- | :--- |
| Built | `agent.py:57` | `agent.py:58` |
| Config | `agent.rewhead`, `configs.yaml:98` | `agent.conhead`, `configs.yaml:99` |
| Output | `symexp_twohot`, 255 bins (`heads.py:132-144`, `outs.py:273-330`) | `binary` logistic (`heads.py:96-99`, `outs.py:189-205`) |
| Loss computed | `agent.py:172-173` | `agent.py:174-177` |
| Target | `obs['reward']` | `not is_terminal`, times `1 - 1/horizon` when `contdisc: True` (`176`) |
| Init | `outscale: 0.0`, so it predicts 0 at start | `outscale: 1.0` |
| Stop-gradient into latent | `sg(..., skip=reward_grad)`, `reward_grad: True` so **no** stop (`agent.py:172`, `configs.yaml:114`) | none, gradient always flows (`agent.py:177`) |

With `contdisc: True` (default), the discount γ = 1 - 1/333 is folded into the continue
target (`agent.py:175-176`) rather than applied separately; imagination then uses
`disc = 1` (`agent.py:401`).

## 7. Loss weights: paper β vs code

Paper Eq. (2): `L = β_pred L_pred + β_dyn L_dyn + β_rep L_rep`, with β_pred = 1,
β_dyn = 1, β_rep = 0.1 (preprint p. 4 and Table 4). In the code there is no single
`β_pred`. Instead one dict, `agent.loss_scales` (`configs.yaml:86`):

```yaml
loss_scales: {rec: 1.0, rew: 1.0, con: 1.0, dyn: 1.0, rep: 0.1, policy: 1.0, value: 1.0, repval: 0.3}
```

It is turned into a per-loss dict at `agent.py:80-83`:

```python
scales = self.config.loss_scales.copy()
rec = scales.pop('rec')                           # 81: remove the 'rec' entry...
scales.update({k: rec for k in dec_space})        # 82: ...and give its value to EVERY decoder key
self.scales = scales
```

| Paper | Code key | Default | Applied to loss |
| :--- | :--- | :--- | :--- |
| β_pred (decoder part) | `rec` | 1.0 | every key in `dec_space`, e.g. `image` |
| β_pred (reward part) | `rew` | 1.0 | `rew` |
| β_pred (continue part) | `con` | 1.0 | `con` |
| β_dyn | `dyn` | 1.0 | `dyn` |
| β_rep | `rep` | 0.1 | `rep` |
| β_pol | `policy` | 1.0 | `policy` |
| β_val | `value` | 1.0 | `value` |
| β_repval | `repval` | 0.3 | `repval` |

`agent.py:237-238` asserts that the set of computed losses and the set of scales are
identical, so a loss can never be silently unweighted.

**Note: all of these, world model and actor-critic, are summed into one scalar and updated
by one optimizer over all modules** (`agent.py:74-78`, `agent.py:240`, `opt.py:43-44`).
Separation between world model and actor-critic comes only from stop-gradients (§11).
The logged `train/opt/loss` is therefore a mixture of everything and is not a
world-model metric.

## 8. The critical question: can image reconstruction be zeroed alone?

### Short answer

**Yes for image-only tasks, via an existing key, with caveats. There is no per-key knob,
and the paper's own version of this ablation (stop-gradient) is not exposed.**

### Details

1. **The existing key is `agent.loss_scales.rec`** (`configs.yaml:86`). Command-line form:
   `--agent.loss_scales.rec 0`. Config-block form:
   ```yaml
   norec:
     agent.loss_scales.rec: 0.0
   ```
   Reward (`rew`) and continue (`con`) have their own scales and are untouched.

2. **It is not image-specific.** Line `agent.py:82` copies the one `rec` value onto *every*
   decoder key. On a task whose only non-bookkeeping observation is `image` this is
   exactly "zero the image term and nothing else". Examples:
   * Crafter: obs are `image` plus bookkeeping plus `log/reward` (`embodied/envs/crafter.py:101-108`), and `log/` is stripped (`main.py:130-131`).
   * DMC vision: `configs.yaml:186` sets `env.dmc.proprio: False`, which drops all vector keys (`embodied/envs/dmc.py:53-56`).

   On tasks that also have vector observations (e.g. `dmc_proprio` style or `dummy`), it
   also zeroes vector reconstruction.

3. **A per-key scale cannot be added from YAML or the command line.** `elements.Config`
   refuses to create keys that do not exist in `defaults` (`elements/config.py:118-119`),
   and unknown flags raise (`elements/flags.py:15-16`). Even if `loss_scales` had an `image`
   entry, `agent.py:82` would overwrite it. Image-only zeroing on a mixed-observation task
   needs a code change at `agent.py:80-83`.

4. **What `rec: 0` does not do** (read from code; runtime cost is **UNCERTAIN**):
   * The decoder is still built, still run forward every step (`agent.py:170-171`), and
     JAX still differentiates through it. Whether XLA removes that dead backward pass is
     unverified. **Assume it costs the same compute until measured.**
   * Decoder weights stay in the optimizer and checkpoints but receive zero gradient, so
     they stay at initialization. With zero gradient the update is exactly zero
     (`opt.py:136-140`: `0 / (sqrt(0) + 1e-20) = 0`), so no NaNs.
   * `train/loss/image` is still logged (`agent.py:239`). It will show the error of an
     untrained decoder, not zero. Anyone reading curves must know this.
   * The open-loop video report (`agent.py:273-307`) becomes meaningless.

5. **The paper's ablation is a stop-gradient, not a zero scale.** The Nature paper
   (p. 5) describes it as "stopping ... the task-agnostic reconstruction gradients from
   shaping its representations". That is different from `rec: 0`: the decoder still
   trains, but on a detached latent. No config key does this. Minimal change, copying the
   existing `reward_grad` pattern at `agent.py:172`:
   * `configs.yaml`, `defaults.agent`: add `rec_grad: True` (a new default is required
     because blocks cannot create keys, point 3).
   * `agent.py:170-171`: pass `sg(repfeat, skip=self.config.rec_grad)` to `self.dec`
     instead of `repfeat`.
   * Ablation block: `norecgrad: {agent.rec_grad: False}`.

   Under this variant the decoder becomes a **probe**: its loss measures how much pixel
   information the latent still carries without being shaped by pixels. That is the
   DreamerV3-scale analogue of the background-readability R² in Experiment 2. Under both
   `rec: 0` and `rec_grad: False`, the gradient reaching encoder and RSSM from the image
   term is zero, so the two should produce identical representation updates. That can be
   asserted in code on one batch.

## 9. Imagination

**Where.** `agent.py:188-216` calls `RSSM.imagine`, `rssm.py:94-118`.

* **Start states** (`agent.py:189-191`, `rssm.py:56-59`). `imag_last: 0`
  (`configs.yaml:104`) means `K = T`: **every** posterior state of the batch starts a
  rollout, so there are `B × T` imagined trajectories.
* **One imagined step** (`rssm.py:95-104`):
  ```python
  action = policy(sg(carry))                              # 96: actor sees a detached state
  deter  = self._core(carry['deter'], carry['stoch'], actemb)   # 98: h' = GRU(h, z, a)
  logit  = self._prior(deter)                             # 99: p(z | h')
  stoch  = self._dist(logit).sample(seed=nj.seed())       # 100: ẑ' sampled from the PRIOR
  ```
  No encoder and no observation appears here. This is the rollout on predicted `ẑ`.
* **Length.** `imag_length: 15` (`configs.yaml:105`), run by `nj.scan`
  (`rssm.py:108-110`).
* **Assembly** (`agent.py:194-201`). The real start state is prepended to the 15 imagined
  states, giving `H + 1 = 16` steps. Both parts are stop-gradiented: the start via
  `sg(first, skip=ac_grads)` with `ac_grads: False` (`configs.yaml:88`), the imagined part
  always (`agent.py:196`).
* Reward and continue for imagined states come from the same heads, on imagined features
  (`agent.py:202-206`).

## 10. Replay context (not in the paper's main description)

The replay buffer stores the model's latent `h, z` for every step and training writes
updated latents back:

* `ext_space` adds `dyn/deter` and `dyn/stoch` to what is stored (`agent.py:94-98`).
* In `train`, updated entries go into `outs['replay']` (`agent.py:144-150`) and are written
  back via `replay.update` (`train.py:78-79`).
* `_apply_replay_context` (`agent.py:312-340`) uses the first `replay_context=1` step
  (`configs.yaml:15`) of each sampled sequence to restore `h, z` instead of starting from
  zeros (`rssm.py:51-54`). Sequences therefore have length `T + 1` (`main.py:187`).

Consequence for the ablation: the initial `h` of each training sequence comes from an
older version of the model. This is the same in both arms, so it is not a confound, but
it is one more piece of hidden state to know about.

## 11. Actor and critic

**Actor loss**, `imag_loss`, `agent.py:382-446`, config `agent.imag_loss`
(`configs.yaml:108`):

* λ-return over imagined rewards and critic values (`agent.py:405`, `482-490`,
  `lam: 0.95`).
* Returns scaled by a percentile normaliser, 5th to 95th percentile, `retnorm`
  (`agent.py:407`, `configs.yaml:111`, `embodied/jax/utils.py:16-91`).
* REINFORCE with normalised advantage and entropy bonus `actent: 3e-4`
  (`agent.py:408-415`). Each step is weighted by the cumulative product of predicted
  continue probabilities (`agent.py:402`).

**Critic loss**, same function (`agent.py:417-422`): two-hot loss to the stop-gradiented
normalised λ-return, plus `slowreg: 1.0` times a loss towards a slow EMA critic
(`slowvalue`, rate 0.02, `configs.yaml:110`, updated at `agent.py:142`,
`embodied/jax/utils.py:94-115`).

**Replay critic loss** `repl_loss`, `agent.py:449-479`, called at `agent.py:218-235`: the
same critic trained on real replayed features with real rewards, bootstrapped from the
imagination return at each start state (`agent.py:222`). Scale `repval: 0.3`.

## 12. What actually shapes the latent (the ablation's real signal set)

Gradients reach the encoder and RSSM parameters from these terms:

| Term | Reaches the latent? | Controlled by | Code |
| :--- | :--- | :--- | :--- |
| image / vector reconstruction | yes | `loss_scales.rec` | `agent.py:170-171` (no `sg`) |
| reward | yes by default | `reward_grad: True` | `agent.py:172` |
| continue | **always** | none | `agent.py:177` |
| `dyn` KL | yes, into prior and GRU | `loss_scales.dyn` | `rssm.py:125` |
| `rep` KL | yes, into posterior and encoder | `loss_scales.rep` | `rssm.py:126` |
| replay critic (`repval`) | yes by default | `repval_grad: True` | `agent.py:220` |
| actor, imagination critic | **no** | `ac_grads: False` | `agent.py:196`, `rssm.py:96` |

So a `rec: 0` arm is trained by **reward + continue + replay value + dyn KL + rep KL**. The
research question says "reward, value, next-latent". Value is already there through
`repval`; continue is an extra signal that should be named in the write-up; "next-latent"
here means KL between categorical distributions, not a regression onto a target latent.

## 13. `configs.yaml`: how blocks stack

**Assembly order** (`main.py:23-31`):

1. Load the whole YAML (`23-24`). YAML anchors are resolved here: `&size1m` defines a
   block and `<<: *size1m` merges it (`configs.yaml:120`, `178-179`).
2. Start from `defaults` (`26`).
3. Apply each block named in `--configs`, **left to right** (`27-28`). Later blocks win.
4. Apply remaining command-line flags last (`29`).
5. Replace `{timestamp}` in `logdir` (`30-31`).
6. Save the final config to `<logdir>/config.yaml` (`42`).

**Merge rules** (`elements/config.py`):

* Nested YAML is flattened to dotted keys: `agent: {loss_scales: {rec: 0}}` and
  `agent.loss_scales.rec: 0` are equivalent (`134-146`).
* A key can only **override**, never create (`118-119`). Every new knob needs a default in
  `defaults` first.
* The new value is converted to the default's type (`121-131`). Writing `rec: 0` where the
  default is `1.0` gives float `0.0`; giving a fractional value to an int key errors.
* A key containing any character outside `[A-Za-z0-9_.-]` is a **regular expression**
  (`11`) and overrides every flat key it matches from the start (`113-115`). That is how
  `.*\.units: 64` in `size1m` resizes every MLP at once (`configs.yaml:123`). A dict under
  a regex key flattens to `pattern\.subkey` (`139-140`), so `.*\.rssm: {deter: 512}` hits
  `agent.dyn.rssm.deter`.

**Command-line rules** (`elements/flags.py`): `--key value` on an existing flat key
(`86-88`); regex flags must fully match (`78-83`); lists are comma-separated (`95-96`);
booleans must be literally `True` or `False` (`104-109`); `--help` prints every key
(`25-29`); unknown flags raise (`15-16`).

**Idiom for a new named variant:** add a top-level block to `configs.yaml` and put it last
on the command line, for example `--configs dmc_vision size12m norec`. Two traps:

* `size*` and `debug` blocks are regexes that touch many keys. A block placed after them
  can undo them, and vice versa.
* The size presets change encoder, decoder and every MLP width together, not just the
  latent. To vary only the latent, override `agent.dyn.rssm.deter`, `.stoch`, `.classes`
  directly.

## 14. Latent sizes by preset (arithmetic from the config, not yet observed)

`feat2tensor` concatenates `h` and flattened `z` (`agent.py:51-53`), so the vector every
head sees has `deter + stoch × classes` numbers.

| Preset | deter | stoch × classes | Head input size |
| :--- | ---: | ---: | ---: |
| `defaults` (same numbers as `size200m`) | 8192 | 32 × 64 = 2048 | 10240 |
| `size1m` | 512 | 32 × 4 = 128 | 640 |
| `size12m` | 2048 | 32 × 16 = 512 | 2560 |
| `size25m` | 3072 | 32 × 24 = 768 | 3840 |
| `size50m` | 4096 | 32 × 32 = 1024 | 5120 |
| `debug` | 8 | 2 × 4 = 8 | 16 |

Sources: `configs.yaml:91`, `121`, `126`, `131`, `136`, `146`, `215-217`. Parameter counts
are **not** implied by the block names. The optimizer prints the true count at startup
(`opt.py:48-50`), and the notebook will record it.

## 15. Runtime facts that matter on Colab

* **Step counting.** `step` goes up by 1 per environment per env step (`train.py:59`).
  Logged steps are multiplied by the action repeat of the suite
  (`main.py:155`, `elements/logger.py:34`).
* **Train ratio.** Train calls per env step = `train_ratio / (B × T)`
  (`train.py:24-25`, `73`). Training starts only once replay holds `B × T` steps
  (`train.py:71`). Crafter uses 512, DMC vision 256 (`configs.yaml:176`, `187`).
* **Log, report and save intervals are wall-clock seconds, not steps.** `LocalClock` uses
  `time.time()` (`embodied/core/clock.py:97-118`). Defaults are 120 s, 300 s and **900 s**
  (`configs.yaml:52-54`). A disconnect can cost up to 15 minutes by default.
* **Checkpoints** go to `<logdir>/ckpt/<timestamp>-<step>/` with `agent.pkl`, `replay.pkl`,
  `step.pkl` and an empty `done` marker. The `latest` pointer is written only after
  `done`, and older folders are deleted after that (`keep=1`)
  (`elements/checkpoint.py:53`, `92-111`, `140-147`, `155-179`). A disconnect during a save
  leaves the previous checkpoint intact. On startup, `load_or_save` resumes if one exists
  (`train.py:83-90`).
* **The replay buffer is written to `<logdir>/replay/`** as compressed `.npz` chunks of
  1024 steps on every checkpoint (`main.py:190`, `replay.py:294-309`, `chunk.py:64-74`) and
  reloaded on resume (`replay.py:311-356`). Nothing in `replay.py` or `chunk.py` deletes
  chunk files, so **the replay folder on Drive grows for the whole run**.
* **Agent checkpoint contents** are the parameter dict plus counters
  (`embodied/jax/agent.py:339-353`). **UNCERTAIN:** I believe optimizer moments and
  normaliser statistics are included because they are `ninjax` variables inside modules
  (`opt.py:60`, `utils.py:91`). Verify by listing checkpoint keys.
* **Metrics.** `<logdir>/metrics.jsonl` receives every scalar (`main.py:160`,
  `elements/logger.py` `JSONLOutput`, default pattern `.*`). `<logdir>/scores.jsonl`
  receives only `episode/score` (`main.py:161-162`). `logger.filter` (`configs.yaml:24`)
  only affects the terminal (`main.py:157`). Non-scalars (open-loop videos, the
  `params/summary` text at `embodied/jax/agent.py:323`) only go to the `scope` output.
* **`scope` viewer** is a FastAPI/uvicorn web server reading `SCOPE_BASEDIR` on port 8000
  (`scope_viewer/__main__.py`, `scope_viewer/config.py:6-15`). **UNCERTAIN** whether it
  works through Colab's port proxy; untested. `scope/reader.py` has a `Reader` class that
  could load the videos directly in Python.
* **JAX profiler** starts at update 100 and stops at 120, writing into the logdir
  (`embodied/jax/agent.py:296-307`). The `profiler` option exists (`agent.py:28`) but is
  not in `configs.yaml`, so it cannot be turned off by flag. A smoke test must run past
  120 updates to exercise this path.
* **Environments run in the main process, one after another**, because `run.debug: True`
  is the default (`configs.yaml:70`) and sets `parallel=not debug` (`train.py:58`).
* **The `debug` block switches to CPU** (`jax: {platform: cpu, debug: True}`,
  `configs.yaml:208`). A `debug` run does not test the CUDA path.
* **GPU XLA flags are inactive under the stock config.** They apply only when
  `platform == 'gpu'` (`internal.py:54`) but the default is `cuda` (`configs.yaml:73`).
* **Compute dtype** is `bfloat16` (`configs.yaml:74`, `internal.py:107-109`). `float16`
  would switch on loss scaling (`opt.py:25-29`).
* **Env seeds.** Only suites with `use_seed` get a seed (`main.py:241-242`); in the stock
  config that is DMLab only (`configs.yaml:34`). Crafter gets `seed=None`
  (`crafter.py:87`). JAX sampling seeds are derived from `config.seed` and a counter
  (`embodied/jax/agent.py:405-408`).
* **`--run.from_checkpoint` is broken as shipped.** `train.py:89` reads
  `args.from_checkpoint_regex`, which is not defined in `configs.yaml`. Resuming via the
  same `--logdir` does not touch this path.

## 16. Discrepancies between paper and code noticed while reading

* LaProp β2: preprint text p. 18 says 0.99; code default is `beta2: 0.999`
  (`configs.yaml:87`). (Paper text via PDF extraction; worth a second look.)
* The paper presents world model and actor-critic as separate objectives; the code sums
  them into one loss with one optimizer (§7). Mathematically equivalent given the
  stop-gradients, but it matters when reading `opt/*` metrics.
* Install: `requirements.txt` pins `jax[cuda12]==0.4.33` and `numpy<2`; the `Dockerfile`
  installs `jax[cuda]==0.5.0` first and then `requirements.txt`, which would downgrade it
  (`Dockerfile`, "Requirements" section).

## 17. Uncertainties to resolve in the notebook

1. Line numbers are for upstream `e3f0224`, not the HERMES fork.
2. `elements`, `ninjax`, `scope` versions read may differ from what gets installed.
3. Whether `rec: 0` saves any compute (XLA dead-code elimination).
4. Whether optimizer state is in the checkpoint.
5. Whether `bfloat16` runs, and how fast, on the GPU Colab assigns. My expectation, not
   verified, is that pre-Ampere cards like the T4 lack native bfloat16 support.
6. Whether the `scope` viewer is reachable from Colab.
7. The encoder output size and latent sizes in §3 and §14 are arithmetic; the shape dump
   will confirm them.
8. From the paper I read only the text and captions of the learning-signal ablation, not
   the curve values.
