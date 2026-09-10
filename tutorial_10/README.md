# Tutorial 10 — Symmetry, Group Actions, and Equivariant Models

**MM845 — Tópicos de Geometria III: AI for Geometry**
Paired with **Lecture 10: Geometry-Aware ML I — Equivariance and Symmetry**

---

## The idea

This tutorial makes explicit a theme that has run through every practical session:
Tutorial 2 split data by orbit, Tutorial 3 found invariant features turned a hopeless
regression exact, Tutorial 5 watched a CNN survive translations, Tutorial 6 the same
for permutations.

Lecture 10 gives the general statement. A group $G$ acts on the data, the label is
invariant, and there are exactly **three** ways to build a model that respects it:

| strategy | how | cost |
|---|---|---|
| **augment** | show the model $g\cdot x$ in training | samples, capacity; only approximate |
| **invariant features** | map to a $G$-invariant descriptor first | must find the invariants; may lose information |
| **equivariant architecture** | layers that commute with $G$ | exact; constrains the design |

We compare all three on the same data and the same budget.

## Files

| File | What it is |
|---|---|
| [`symmetry_and_equivariant_models.ipynb`](symmetry_and_equivariant_models.ipynb) | The tutorial. 5 sections, 2 figures, 2 exercises. ~1 hour. |
| `README.md` | This file. |

## Running it

```bash
$ conda activate aigeo
$ jupyter lab symmetry_and_equivariant_models.ipynb
```

Needs the `aigeo` environment from [Tutorial 1](../tutorial_01/README.md). Runs in
about 25 seconds.

## Contents

| § | Topic | Lecture 10 connection |
|---|---|---|
| 1 | Point clouds on ellipsoids with an exact $SO(3)$ action | Lie group actions |
| 2 | Three strategies compared, one of them an equivariant layer | equivariance, symmetry-preserving models |
| 3 | Sample efficiency — where symmetry actually pays | why constraints help |
| 4 | A different group: $S_n$ on graphs; message passing against spectra | GNNs, graph symmetry |
| 5 | Summary | |

## Results worth watching for

**§2 — all three strategies, one dataset, one budget.**

| model | params | canonical | rotated |
|---|---|---|---|
| raw MLP | 25,985 | 0.333 | 0.334 |
| + augmentation | 25,985 | 0.221 | 0.227 |
| equivariant net | 26,497 | 0.103 | 0.101 |
| invariant features | 17,153 | **0.011** | **0.012** |
| *for contrast:* the raw Gram matrix | 55,169 | *0.454* | *0.443* |

Baseline (predict the mean): 1.170.

The raw model does **not** collapse under rotation — but its score is only about a
third of baseline, meaning it barely learned the task at all. There was no sharp
orientation-dependent rule for rotation to break. The tutorial draws the conclusion
that matters: **symmetry is not only about robustness, it is about difficulty**.

The equivariant net is a single linear layer acting on the *point* index, which
commutes with $X \mapsto XR^\top$, followed by norms and the same MLP. Ten lines,
verified equivariant to $10^{-7}$ **before** it is trained, and on raw coordinates it
beats both the raw network and augmentation at essentially the same parameter budget.

The last row is there to stop an easy misreading. Hand the same MLP the **complete**
invariant in raw form — the whole $24\times24$ Gram matrix, from which the cloud is
determined up to rotation — and it does *worse than the raw coordinates*. Nothing was
lost to the group; everything is still there. What is missing is that no network of
this size extracts three eigenvalues from a Gram matrix. The fortyfold gap between the
last two rows measures that nonlinear work, **not** the symmetry.

**§3 — a seventeenfold saving in data.** Learning curves on rotated test data:

| $N$ | raw | augmented | invariant |
|---|---|---|---|
| 100 | 0.945 | 0.698 | **0.295** |
| 1500 | 0.515 | 0.309 | **0.013** |
| 4000 | 0.332 | 0.205 | **0.011** |

A learning curve is read **horizontally**: not which curve is lower at a given $N$, but
how far right you must travel to buy the same accuracy. The augmented model needs about
**1700** clouds to reach what the invariant model reaches with **100** — a seventeenfold
saving, which the notebook interpolates off the curve rather than obtaining by dividing
the endpoints of the axis. That is the currency symmetry is paid in, and it matters most
exactly where mathematical data is expensive to generate — an invariant of a variety, a
homology group, a numerical metric.

**§4 — the same framework, a different group, and two theorems.** With $S_n$ acting on
graphs by relabelling and the label the triangle count $\tfrac16\operatorname{tr}(A^3)$:

| model | params | as generated | permuted |
|---|---|---|---|
| raw MLP on $A$ | 35,201 | 4.368 | 4.814 |
| message passing | 27,169 | 1.901 | 1.813 |
| spectrum of $A$ | 18,305 | **0.287** | **0.271** |

Baseline: 140.6. Message passing — $h_i \leftarrow \phi(h_i, \sum_{j\sim i} h_j)$, a
few rounds, then a sum over nodes — is exactly permutation-invariant and beats the raw
MLP with *fewer* parameters, but it loses to the spectrum by a factor of nearly seven.

**That ordering is not a tuning failure, and the section demonstrates why in both
directions** on two graphs of five lines each:

| pair | triangles | spectra differ by | embeddings differ by |
|---|---|---|---|
| $C_6$ vs $C_3 \sqcup C_3$ | **0 vs 2** | 1.00 | **0.00** |
| $C_4 \sqcup K_1$ vs $K_{1,4}$ | 0 vs 0 | $4\times10^{-16}$ | 1.36 |

Both graphs in the first pair are $2$-regular, so Weisfeiler–Leman gives every vertex
the same colour in both and message passing returns *exactly* the same embedding — yet
their triangle counts differ. The second pair is cospectral, the smallest such, so the
spectral descriptor sends them to the same point and no model built on it can ever
separate them.

So the two descriptors are **incomparable**: neither refines the other, and the common
claim that equivariant architectures are strictly more expressive than spectral ones is
false on these graphs. The question is not which method is strongest but **which
incompleteness you can afford for the label you actually have** — and here the label is
$\tfrac16\sum_i \lambda_i^3$, which is exactly why the spectrum wins.

## Exercises

| # | § | Topic |
|---|---|---|
| 1 | 2 | Incomplete invariants; non-invariant labels; chirality; permutation-equivariance |
| 2 | 4 | Cospectral pairs as a loss floor; fixing WL's blind spot; reversing the ranking |

## What to take away

- **Three strategies, one framework** — augmentation, invariant descriptors,
  equivariant architectures — trading samples, mathematical effort, and design freedom.
- **Augmentation works and is not free**: approximate, slower, and only covers the
  transformations you thought of.
- **An equivariant architecture buys exactness, not feature extraction.** It beat both
  the raw network and augmentation on raw coordinates, and was invariant to machine
  precision before training began — but three hand-computed eigenvalues still beat it
  eightfold. Symmetry constrains the hypothesis space; it does not do your algebra.
- **Invariants are free symmetry when you can find them**, and finding them is
  mathematics — this audience's advantage over a practitioner with more GPUs.
- **Sample efficiency is where symmetry pays**, and the gap is widest where data is
  scarcest.
- **Ask whether your invariant is complete, and what it is complete *for*.** The
  covariance spectrum is complete for $O(3)$ on point clouds; the adjacency spectrum is
  not complete for $S_n$ on graphs — and message passing is not an improvement on it
  either, merely incomparable to it. What the descriptor forgets, the model can never
  learn.

## Next

**Lecture 11** continues into manifolds, spectral methods and diffusion models;
**Lectures 12–13** turn to PINNs. The mini-project begins in week 7, and a symmetry
analysis of whatever data you choose is usually the highest-value hour you can spend
on it.

## Further reading

- Bronstein, Bruna, Cohen & Veličković, *Geometric Deep Learning* (2021) — the blueprint this tutorial follows.
- Cohen & Welling, "Group equivariant convolutional networks", *ICML* 2016.
- Villar et al., "Scalars are universal", *NeurIPS* 2021 — the completeness question of §2 and §4, done properly.
- Xu et al., "How powerful are graph neural networks?", *ICLR* 2019 — the expressivity limits behind Exercise 2.
- Chen, Villar, Chen & Bruna, "Can graph neural networks count substructures?", *NeurIPS* 2020 — why §4's message-passing model cannot count triangles.
