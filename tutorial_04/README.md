# Tutorial 4 — Neural Networks as Nonlinear Ansatz Spaces

**MM845 — Tópicos de Geometria III: AI for Geometry**
Paired with **Lecture 4: Neural Networks — Theory, Architecture, Training**

---

## The idea

Tutorial 3 fixed a feature map $\phi$ by hand and fitted a linear model on it:
one minimum, a closed form, a convergence rate that was a ratio of eigenvalues.

A neural network keeps the shape of that idea and changes one thing — **$\phi$ is
learned rather than designed** — at the cost of the entire apparatus of guarantees.
This tutorial measures what the trade buys and what it costs, on geometric targets
of controlled regularity.

## Files

| File | What it is |
|---|---|
| [`neural_ansatz_spaces.ipynb`](neural_ansatz_spaces.ipynb) | The tutorial. 5 sections, 3 figures, 3 exercises. ~1 hour. |
| `README.md` | This file. |

## Running it

```bash
$ conda activate aigeo
$ jupyter lab neural_ansatz_spaces.ipynb
```

Needs the `aigeo` environment from [Tutorial 1](../tutorial_01/README.md). PyTorch
is used throughout. The notebook runs top to bottom in about 40 seconds on a laptop
CPU — §2 trains 15 networks and §3 trains 8 more.

## Contents

| § | Topic | Lecture 4 connection |
|---|---|---|
| 1 | An MLP as a space of functions; a linear model on a *moving* basis | MLPs, architecture |
| 2 | Polynomials vs networks at matched parameter count, on three regularities | approximation, capacity |
| 3 | The non-convex landscape: 8 seeds, 8 different minima | loss landscape geometry |
| 4 | Spectral bias — which part of a function is learned first | training dynamics |
| 5 | Summary | |

## Results worth watching for

**§2 — the honest comparison, which does not flatter neural networks.** At matched
parameter count on $S^2$:

| target | best polynomial | best MLP |
|---|---|---|
| degree-3 harmonic | $1.0\times10^{-30}$ | $3.9\times10^{-5}$ |
| Gaussian bump | $7.0\times10^{-8}$ | $1.7\times10^{-5}$ |
| $\lvert z\rvert$ | $1.0\times10^{-4}$ | $1.9\times10^{-5}$ |

The polynomial wins both analytic targets — the first by twenty-six orders of
magnitude, because the truth lies in its span. The network wins only on the kink.
The tutorial states the moral plainly: on smooth targets in low dimension,
classical approximation beats a neural network comfortably. Networks earn their
keep when the target is rough, when the dimension makes a polynomial basis
combinatorially impossible, or when no good basis is known — which is the regime
the rest of the course lives in.

**§3 — non-convexity, quantified.** Eight seeds, same architecture and data: final
training losses spread by $8.6\times$ and test MSE by $11.5\times$, with the runs
landing in genuinely different minima (they disagree pointwise) of comparable
quality. The practical conclusions — report a spread rather than a number, fix and
record seeds, and read between-run disagreement as a rough uncertainty map — are
the ones students should carry into their mini-projects.

**§4 — spectral bias, measured.** Training on $\sin\theta + \tfrac12\sin 9\theta$,
the network drives mode 1 below threshold at **epoch 50** and mode 9 only at
**epoch 1200**. The residual-spectrum heat map shows the error draining from low
frequencies upward. Three consequences are drawn: early stopping is approximately a
low-pass filter (hence approximately Tutorial 3 §4's Dirichlet-energy penalty, by a
completely different route); it partly explains why networks generalise better than
their parameter count suggests; and it is the obstacle PINNs fight in Lecture 13,
where it shows up as stiffness.

## Exercises

| # | § | Topic |
|---|---|---|
| 1 | 2 | Where is the crossover? Regularity, activations, and learning curves |
| 2 | 3 | Is it the seed or the landscape? Width, SGD noise, ensembling |
| 3 | 4 | Spectral bias as a power law; Fourier features as the fix |

## What to take away

- An MLP is **a linear model on a learned feature map** — the last layer does
  Tutorial 3's least squares, the rest chooses the basis.
- The hypothesis space is **not a vector space**, and that one fact removes the
  projection theorem, the closed form and convexity together.
- **Regularity decides who wins.** If the truth is in your span, use the span.
- **Non-convexity is survivable but not free** — results are distributions.
- **Networks learn smooth structure first**: a free regulariser when you want
  smoothness, an obstacle when you do not.

## Next

**Lecture 5** constrains the architecture instead of enlarging it: convolutions are
linear maps that commute with translation. **Tutorial 5** measures what that
symmetry constraint is worth on geometric image data.

## Further reading

- Rahaman et al., "On the spectral bias of neural networks", *ICML* 2019 — §4 in full.
- Tancik et al., "Fourier features let networks learn high frequency functions in low dimensional domains", *NeurIPS* 2020 — the fix in Exercise 3(b).
- Goodfellow, Bengio & Courville, *Deep Learning*, ch. 6 & 8. Free online.
- Trefethen, *Approximation Theory and Approximation Practice*, ch. 9 — why polynomials struggle with $|x|$.
