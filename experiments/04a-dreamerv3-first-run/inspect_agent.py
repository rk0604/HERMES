"""Look at a DreamerV3 agent without training it.

Builds the agent exactly the way dreamerv3/main.py does, from the same config
flags, and prints two things:

  1. The parameter tree: every weight tensor with its shape, and per-module
     totals. This is the true size of the model behind a preset like size12m.
  2. The tensor shapes that flow through the RSSM for one batch: image ->
     tokens -> h (deter) -> z (stoch) -> prior logits -> imagined h and z-hat.

Everything after the script's own flags is passed to DreamerV3's config
system unchanged, for example:

  python inspect_agent.py --repo /content/dreamerv3 --configs atari100k size12m
"""
import argparse
import json
import pathlib
import sys
import tempfile

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--repo', required=True, help='path of the dreamerv3 clone')
parser.add_argument('--out', default='', help='optional JSON file to write the results to')
own, rest = parser.parse_known_args()

REPO = pathlib.Path(own.repo)
sys.path.insert(0, str(REPO))
import elements                            # noqa: E402  config/flags library used by DreamerV3
import jax                                 # noqa: E402
import ninjax as nj                        # noqa: E402  the module system DreamerV3 is written in
import ruamel.yaml as yaml                 # noqa: E402
from dreamerv3 import main as dv3          # noqa: E402
from dreamerv3.agent import Agent, sample  # noqa: E402

# --- 1. Assemble the config exactly like dreamerv3/main.py lines 23-29 -------
configs = yaml.YAML(typ='safe').load((REPO / 'dreamerv3' / 'configs.yaml').read_text())
parsed, other = elements.Flags(configs=['defaults']).parse_known(rest)
config = elements.Config(configs['defaults'])
for name in parsed.configs:                # later blocks override earlier ones
  config = config.update(configs[name])
config = elements.Flags(config).parse(other)   # command-line flags win last
print('Config blocks:', list(parsed.configs), ' task:', config.task)

# --- 2. Build the agent like dreamerv3/main.py make_agent(), minus precompile
env = dv3.make_env(config, 0)              # only to read the observation/action spaces
obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith('log/')}
act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
env.close()
agent = Agent(obs_space, act_space, elements.Config(
    **config.agent,
    logdir=tempfile.mkdtemp(),             # only used by the profiler, which we never reach
    seed=config.seed,
    jax={**config.jax, 'precompile': False},   # skip compiling train(); we only run a forward pass
    batch_size=config.batch_size,
    batch_length=config.batch_length,
    replay_context=config.replay_context,
    report_length=config.report_length,
    replica=config.replica,
    replicas=config.replicas,
))
model = agent.model                        # the dreamerv3.agent.Agent; `agent` is its JAX wrapper
B, T, C = config.batch_size, config.batch_length, config.replay_context
results = {'config_blocks': list(parsed.configs), 'task': config.task}

# --- 3. Parameter tree -------------------------------------------------------
print('\n' + '=' * 78 + '\nPARAMETER TREE\n' + '=' * 78)
counts = {}
for key, value in agent.params.items():
  module = key.split('/')[0]
  counts[module] = counts.get(module, 0) + int(np.prod(value.shape))
  print(f'{key:<52} {str(value.dtype):<9} {tuple(value.shape)}')
print('-' * 78)
model_modules = ('enc', 'dyn', 'dec', 'rew', 'con', 'pol', 'val')
for module, n in sorted(counts.items(), key=lambda kv: -kv[1]):
  tag = '' if module in model_modules else '   (not a network: optimizer state / EMA copy / running stats)'
  print(f'{module:<10} {n:>14,}{tag}')
n_model = sum(counts[m] for m in model_modules)
print(f'{"model":<10} {n_model:>14,}   (enc+dyn+dec+rew+con+pol+val, what the size presets describe)')
results['param_counts'] = counts
results['param_shapes'] = {k: list(v.shape) for k, v in agent.params.items()}

# --- 4. A random batch with the right shapes ---------------------------------
# Real replay data is not needed to see shapes; random pixels are enough.
rng = np.random.default_rng(0)
data = agent._zeros(agent.spaces, (B, T + C))
for key, space in agent.spaces.items():
  if space.dtype == np.uint8 and len(space.shape) == 3:          # images
    data[key] = rng.integers(0, 256, data[key].shape, np.uint8)
  elif key in act_space and space.discrete and space.shape == ():  # discrete actions
    data[key] = rng.integers(0, int(space.high), data[key].shape).astype(space.dtype)
data['is_first'][:, 0] = True
# DreamerV3 forbids implicit host->device copies, so move the batch explicitly.
data = jax.device_put(data)
seed = jax.device_put(np.array([config.seed, 0], np.uint32))

# --- 5. Shapes through the RSSM ----------------------------------------------
def probe(data):
  carry = model.init_train(B)
  carry, obs, prevact, _ = model._apply_replay_context(carry, data)
  enc_carry, dyn_carry, dec_carry = carry
  reset = obs['is_first']
  _, _, tokens = model.enc(enc_carry, obs, reset, training=False)          # encoder
  dyn_carry, entries, feat = model.dyn.observe(                             # h update + posterior q(z|h,x)
      dyn_carry, tokens, prevact, reset, training=False)
  prior_logits = model.dyn._prior(feat['deter'])                            # prior p(z|h)
  K = min(model.config.imag_last or T, T)
  H = model.config.imag_length
  starts = model.dyn.starts(entries, dyn_carry, K)
  policy = lambda f: sample(model.pol(model.feat2tensor(f), 1))
  _, imgfeat, _ = model.dyn.imagine(starts, policy, H, training=False)      # rollout on z-hat
  _, _, recons = model.dec(dec_carry, feat, reset, training=False)          # decoder
  kl = model.dyn._dist(feat['logit']).kl(model.dyn._dist(prior_logits))     # KL[q || p] per step
  return {'kl': kl}, {
      'obs/image': obs['image'],
      'enc/tokens': tokens,
      'rssm/h (deter)': feat['deter'],
      'rssm/z (stoch, one-hot)': feat['stoch'],
      'rssm/posterior logits q(z|h,x)': feat['logit'],
      'rssm/prior logits p(z|h)': prior_logits,
      'heads/input = concat(h, flat z)': model.feat2tensor(feat),
      'imag/h': imgfeat['deter'],
      'imag/z-hat (sampled from prior)': imgfeat['stoch'],
      'dec/reconstruction': recons['image'].pred(),
  }

_, (extra, out) = jax.jit(nj.pure(probe))(agent.params, data, seed=seed)
out, extra = jax.device_get((out, extra))
print('\n' + '=' * 78 + f'\nSHAPES FOR ONE BATCH  (B={B} sequences, T={T} steps, imag_length={model.config.imag_length})\n' + '=' * 78)
for key, value in out.items():
  print(f'{key:<36} {str(value.dtype):<9} {tuple(value.shape)}')
z, zhat = out['rssm/z (stoch, one-hot)'], out['imag/z-hat (sampled from prior)']
assert np.allclose(z.astype(np.float32).sum(-1), 1), 'z is not one-hot over classes'
assert np.allclose(zhat.astype(np.float32).sum(-1), 1), 'z-hat is not one-hot over classes'
print('check: z and z-hat are one-hot over the last axis (every latent picks exactly one class)')
K = min(model.config.imag_last or T, T)
assert zhat.shape[:2] == (B * K, model.config.imag_length), zhat.shape
print(f'check: imagination starts from every one of the B*T = {B * K} posterior states '
      f'and rolls {model.config.imag_length} steps')
kl, free = extra['kl'].astype(np.float32), model.dyn.free_nats
print(f'KL[posterior || prior] at initialisation: mean {kl.mean():.3f} nats; {(kl < free).mean():.0%} of steps '
      f'are below the {free} free nat(s), where the KL losses are clipped and send no gradient yet')
results['shapes'] = {k: list(v.shape) for k, v in out.items()}
results['kl_init'] = dict(mean=float(kl.mean()), frac_below_free_nats=float((kl < free).mean()))
if own.out:
  pathlib.Path(own.out).write_text(json.dumps(results, indent=1))
