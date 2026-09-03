# Tutorial 8 — Spectral and Nonlinear Methods for Geometric Data

**MM845 — Tópicos de Geometria III: AI for Geometry**
Paired with **Lecture 8: Dimensionality Reduction — Kernel PCA & Nonlinear Visualisation**

---

## The idea

Tutorial 7 asked an unsupervised method for *labels*. This one asks for
**coordinates**: given points in $\mathbb{R}^n$, produce a faithful picture in two
dimensions.

For a geometer the question is not "does it look nice" but **what is preserved and
what is destroyed**. Every dimension reduction distorts something; the methods differ
in what they sacrifice. The aim is to leave you able to say, of any such picture,
what you are entitled to read off it.

## Files

| File | What it is |
|---|---|
| [`spectral_and_nonlinear_methods.ipynb`](spectral_and_nonlinear_methods.ipynb) | The tutorial. 5 sections, 3 figures, 3 exercises. ~1 hour. |
| `README.md` | This file. |

## Running it

```bash
$ conda activate aigeo
$ jupyter lab spectral_and_nonlinear_methods.ipynb
```

Needs the `aigeo` environment from [Tutorial 1](../tutorial_01/README.md). NumPy and
SciPy only — PCA, kernel PCA and t-SNE are all written out (t-SNE in about 30 lines,
including the perplexity bisection). Runs in about 50 seconds — §4 fits t-SNE five
times, four of them for the simplex test. Exercise 3 optionally
wants `pip install umap-learn`.

## Contents

| § | Topic | Lecture 8 connection |
|---|---|---|
| 1 | PCA as an eigenproblem; span vs manifold dimension | PCA as an eigenproblem |
| 2 | Two concentric shells; kernel PCA and the RKHS trick | kernel PCA, feature maps |
| 3 | The flat torus $S^1\times S^1$: three obstructions to a faithful picture | product manifolds |
| 4 | t-SNE — what to believe and what not to | t-SNE, perplexity, pitfalls |
| 5 | Summary | |

## Results worth watching for

**§1 — PCA answers a different question than you may think.** A sphere embedded in
$\mathbb{R}^{60}$ gives variances `0.346 0.334 0.320 0.0001 …` — PCA correctly
reports **3**, the dimension of the *span*, not the **2** of the manifold. It is not
wrong; it is measuring the smallest flat containing the data.

**§2 — the separating component is not the first.** For two concentric shells, no
linear component separates them at all (separation $\approx 0.01$ std units, as it
must be — both shells share a centroid). But neither does the *first* kernel
component: the discriminating direction is **component 3**, and it separates by about
two standard deviations.

The reason is geometric and worth the detour. The data is $O(3)$-symmetric, so the
largest variance in feature space is angular — which shell a point is on is a single
radial bit, while *where on the sphere* it sits varies far more. Kernel PCA orders by
variance, so the interesting coordinate is not the loudest. **"Plot PC1 against PC2"
is a convention, not a method.**

**§3 — distortion that no method can avoid.** The flat torus is genuinely flat, so
by the Theorema Egregium no *smooth* surface in $\mathbb{R}^3$ is isometric to it
(every compact one has a point of positive curvature). All four PCA variances come
out $\approx 0.5$ (the exact value of $\operatorname{var}\cos$), so PCA has no
preferred direction and any 2-D projection discards half the variance — and kernel
PCA inherits the same degeneracy, so it does not help either.

The ambient-vs-intrinsic distance correlation is **0.974** — and the tutorial argues
why that high number is misleading rather than reassuring: the chord $2\sin(d/2)$ is
concave in the arc $d$ and *saturates*, so ambient distance systematically compresses
large intrinsic distances, which a correlation coefficient hides completely.

**§4 — t-SNE, quantified rather than admired.**

| method | $k$-NN preserved | correlation with geodesic |
|---|---|---|
| PCA | 43.5% | **0.637** |
| t-SNE | **81.4%** | 0.562 |

Exactly the specification: t-SNE optimises neighbourhoods and delivers them; global
distance is not in its objective and it does worse than PCA there. The torus also
comes out **torn** — forced, since a closed surface admits no injective continuous
map to the plane.

**§4 also shows the method earning its keep.** Twelve clusters in $\mathbb{R}^{50}$,
their centres a regular simplex spanning an $11$-flat. PCA resolves **4 of 12** — the
centre covariance is isotropic in that flat, so no plane is preferred, and no plane can
hold twelve mutually equidistant points apart. t-SNE resolves **12 of 12** at every
perplexity and seed, at $100\%$ neighbourhood purity against a chance level of $8.3\%$.
It never projects, so the eleven dimensions cost it nothing.

The section closes on the reading discipline: believe which points are near which; do
not believe distances between clusters, cluster sizes, empty space, or component
counts; always vary perplexity and seed. The cluster count is trustworthy in the
simplex test only because those groups are genuine metric clusters — on a connected
manifold like the torus, the pieces you see are tear lines, and their number is a
property of the run.

## Exercises

| # | § | Topic |
|---|---|---|
| 1 | 2 | The kernel *is* the geometry: $\gamma$ sweeps, polynomial kernels, intrinsic dimension |
| **2** | 4 | **Stress-test the picture: 9 runs, intrinsic vs ambient input, the trade-off curve** |
| 3 | 4 | UMAP, compared with the same two numbers rather than by eye |

## What to take away

- **PCA finds the smallest flat containing your data** — know which question you asked.
- **Kernel PCA moves the eigenproblem into feature space**, separating what no linear
  coordinate can, at the price of uninterpretable components and a kernel width that
  is really a choice of geometry.
- **Some distortion is forced by theorems**, not by algorithms.
- **t-SNE optimises neighbourhoods and nothing else** — and that is a specification,
  not a defect.
- **Turn the knobs and keep what does not move.** The claim that survived every
  perplexity and seed is the one you may act on.
- **Measure distortion rather than eyeballing it.** Two cheap numbers turn "this
  looks like clusters" into a claim you can defend.

## Next

**Lecture 9** changes the setting entirely: no fixed dataset, but an agent generating
its own data by acting. **Tutorial 9** puts a Markov decision process on a
triangulation.

## Further reading

- Schölkopf, Smola & Müller, "Nonlinear component analysis as a kernel eigenvalue problem", *Neural Computation* **10** (1998).
- van der Maaten & Hinton, "Visualizing data using t-SNE", *JMLR* **9** (2008).
- Wattenberg, Viégas & Johnson, "How to use t-SNE effectively", *Distill* (2016) — strongly recommended.
- McInnes, Healy & Melville, "UMAP", arXiv:1802.03426.
