"""Faster, numerically equivalent LSTM forward pass for sb3-contrib's RecurrentPPO.

Why this exists
---------------
During a PPO update, sb3-contrib's `RecurrentActorCriticPolicy._process_sequence`
runs the LSTM with a single call ONLY if no sequence contains an episode start.
Otherwise it falls back to a Python loop that calls the LSTM once per timestep
(up to ~800 calls per minibatch, for the actor LSTM and again for the critic
LSTM, forward and backward). In this project virtually every minibatch hits that
slow path, because the rollout buffer's sequences begin at episode starts, so
the update dominated the wall time (~89 % of a training cycle on a 2-vCPU
machine) while the CPUs were mostly idle.

What it changes
---------------
The rollout buffer cuts its sequences at every episode start, so a state reset
can only happen at the FIRST timestep of a sequence. For that case the reset can
be applied once, up front, by multiplying the initial (h, c) by `1 - episode_start`,
followed by ONE `nn.LSTM` call over the whole sequence. This is exactly what the
loop computes (same equations, same reset), only executed as one fused call.
If a reset ever occurs in the middle of a sequence (never the case with
sb3-contrib's buffer, but possible in principle), the original loop is used, so
the result is always correct.

Nothing is stored in the saved models: checkpoints trained with or without the
patch load and run identically. Gradients match the original to ~1e-6 (float32
summation order), so runs are not bit-identical to an unpatched run, but they
are bit-reproducible with respect to each other.

Safety
------
`apply()` only patches sb3-contrib 2.7.x, checks the original function's
signature, and runs a small equivalence self-test (both code paths, forward and
backward) before keeping the patch. In any doubt it leaves sb3-contrib untouched
and training simply runs at the original speed.
"""
import inspect
import warnings

import torch as th
from torch import nn

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

# Plain function (the attribute on the class is a staticmethod).
_original = RecurrentActorCriticPolicy.__dict__["_process_sequence"].__func__
_SUPPORTED_PREFIX = "2.7."
_EXPECTED_PARAMS = ["features", "lstm_states", "episode_starts", "lstm"]


def _fast_process_sequence(features, lstm_states, episode_starts, lstm):
    n_seq = lstm_states[0].shape[1]
    # (padded batch, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
    features_sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
    starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

    # A state reset in the MIDDLE of a sequence needs the step-by-step loop.
    # (max length == 1 during data collection, hence no middle there.)
    if starts.shape[0] > 1 and bool(th.any(starts[1:] != 0.0)):
        return _original(features, lstm_states, episode_starts, lstm)

    # Reset the state if the sequence begins a new episode, then a single LSTM call.
    keep = (1.0 - starts[0]).view(1, n_seq, 1)
    lstm_output, lstm_states = lstm(features_sequence, (keep * lstm_states[0], keep * lstm_states[1]))
    lstm_output = th.flatten(lstm_output.transpose(0, 1), start_dim=0, end_dim=1)
    return lstm_output, lstm_states


def _self_test():
    """Compare the fast path with sb3-contrib's own function on random data.

    Covers: reset at the first step, no reset, reset in the middle (fallback),
    data-collection shape (length 1). Compares outputs, final states and gradients.
    """
    gen = th.Generator().manual_seed(0)
    lstm = nn.LSTM(8, 16)
    n_seq, length = 5, 12
    for starts_kind in ("first", "none", "middle", "single"):
        L = 1 if starts_kind == "single" else length
        features = th.randn(n_seq * L, 8, generator=gen, requires_grad=True)
        h = th.randn(1, n_seq, 16, generator=gen)
        c = th.randn(1, n_seq, 16, generator=gen)
        starts = th.zeros(n_seq, L)
        if starts_kind in ("first", "single"):
            starts[:, 0] = th.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        elif starts_kind == "middle":
            starts[:, 0] = th.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
            starts[1, 5] = 1.0
        starts = starts.reshape(-1)

        results = []
        for fn in (_original, _fast_process_sequence):
            lstm.zero_grad()
            features.grad = None
            out, (h_n, c_n) = fn(features, (h, c), starts, lstm)
            (out.pow(2).sum() + h_n.sum() + c_n.sum()).backward()
            grads = th.cat([p.grad.flatten() for p in lstm.parameters()] + [features.grad.flatten()])
            results.append((out.detach(), h_n.detach(), c_n.detach(), grads.clone()))
        for a, b in zip(*results):
            if a.shape != b.shape or not th.allclose(a, b, atol=1e-5, rtol=1e-4):
                return False
    return True


def apply(verbose=True):
    """Install the fast LSTM forward pass. Returns True if active, False if skipped."""
    try:
        import sb3_contrib
        version = getattr(sb3_contrib, "__version__", "unknown")
        params = list(inspect.signature(_original).parameters)
        if not version.startswith(_SUPPORTED_PREFIX) or params != _EXPECTED_PARAMS:
            raise RuntimeError(f"unsupported sb3-contrib (version {version}, signature {params})")
        RecurrentActorCriticPolicy._process_sequence = staticmethod(_fast_process_sequence)
        if not _self_test():
            raise RuntimeError("equivalence self-test failed")
    except Exception as exc:  # never break training because of an optimisation
        RecurrentActorCriticPolicy._process_sequence = staticmethod(_original)
        warnings.warn(f"fast_lstm not applied, using sb3-contrib's original LSTM loop: {exc}")
        return False
    if verbose:
        print("fast_lstm: single-call LSTM forward pass active (equivalence self-test passed)", flush=True)
    return True


def revert():
    """Restore sb3-contrib's original implementation."""
    RecurrentActorCriticPolicy._process_sequence = staticmethod(_original)
