"""
perlin_noise.py

Deterministic, pure-numpy 2D Perlin noise (fractal Brownian motion).

This replaces the third-party `noise` package's pnoise2, which was found to
produce NON-DETERMINISTIC output across separate Python process launches for
certain (base, octaves) combinations — almost certainly an out-of-bounds or
uninitialized-memory read in its C extension that only manifests at higher
octave counts. Because environment topology generation used pnoise2 with
octaves=4 and octaves=6, the SAME seed could silently produce a different
vessel layout (and therefore a different agent/target placement and a
different true difficulty) every time a script was re-run as a new process —
even though everything else in the pipeline was fully deterministic.

This implementation is:
  - Fully vectorized (computes an entire 2D grid in one call instead of
    looping cell by cell in Python), which is also dramatically faster.
  - Provably deterministic: identical (coords, seed, octaves, persistence,
    lacunarity) always produces bit-identical output, in any process, on any
    machine, forever — because it only ever touches numpy arrays it
    allocates itself and a permutation table built from `np.random.default_rng`
    (whose bit-stream stability across numpy versions is guaranteed).
"""
import numpy as np

# The 8 unit gradient directions used by this classic ("Ken Perlin style")
# 2D noise formulation.
_GRAD2 = np.array([
    [1, 1], [-1, 1], [1, -1], [-1, -1],
    [1, 0], [-1, 0], [0, 1], [0, -1],
], dtype=np.float64)


def _make_permutation(seed: int) -> np.ndarray:
    """
    Deterministic 256-entry permutation table, duplicated to 512 entries so
    that lookups never need a modulo/wraparound branch.
    """
    rng = np.random.default_rng(int(seed))
    p = np.arange(256, dtype=np.int64)
    rng.shuffle(p)
    return np.concatenate([p, p])


def _fade(t: np.ndarray) -> np.ndarray:
    """Perlin's smoothstep-like fade curve: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _gradient(perm: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """Looks up one of the 8 gradient vectors for each (ix, iy) lattice point."""
    idx = perm[(perm[ix & 255] + (iy & 255)) & 511] & 7
    return _GRAD2[idx]


def perlin2d(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """
    Single-octave 2D Perlin noise, vectorized over numpy arrays x and y
    (must share the same shape). Returns values roughly within [-1, 1].
    Deterministic: depends only on (x, y, seed).
    """
    perm = _make_permutation(seed)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    sx = x - x0
    sy = y - y0

    g00 = _gradient(perm, x0, y0)
    g10 = _gradient(perm, x1, y0)
    g01 = _gradient(perm, x0, y1)
    g11 = _gradient(perm, x1, y1)

    n00 = g00[..., 0] * sx       + g00[..., 1] * sy
    n10 = g10[..., 0] * (sx - 1) + g10[..., 1] * sy
    n01 = g01[..., 0] * sx       + g01[..., 1] * (sy - 1)
    n11 = g11[..., 0] * (sx - 1) + g11[..., 1] * (sy - 1)

    u = _fade(sx)
    v = _fade(sy)

    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    nxy = nx0 + v * (nx1 - nx0)

    return nxy


def fbm2d(x: np.ndarray, y: np.ndarray, base: int,
          octaves: int = 1, persistence: float = 0.5, lacunarity: float = 2.0) -> np.ndarray:
    """
    Fractal Brownian motion: a weighted sum of multiple octaves of Perlin
    noise at increasing frequency and decreasing amplitude. Drop-in
    deterministic replacement for the `noise` package's
    `pnoise2(x, y, base=base, octaves=octaves, persistence=persistence, lacunarity=lacunarity)`,
    but vectorized over full coordinate arrays instead of one point at a time.
    Output is normalized to stay approximately within [-1, 1].
    """
    total = np.zeros_like(x, dtype=np.float64)
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0
    for _ in range(octaves):
        total += amplitude * perlin2d(x * frequency, y * frequency, base)
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return total / max_amplitude