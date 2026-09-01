# Tutorial 7 — Clustering and Metric Geometry

**MM845 — Tópicos de Geometria III: AI for Geometry**
Paired with **Lecture 7: Unsupervised Learning — Clustering**

---

## The idea

The labels go away. Given only a point cloud, what can be recovered?

Lecture 7's three algorithms encode three *different* definitions of "cluster", and
the differences are geometric rather than technical:

| method | a cluster is… | the geometry it sees |
|---|---|---|
| **k-means** | points near a common centre | Euclidean, convex cells |
| **spectral** | a region weakly connected to the rest | the graph Laplacian |
| **DBSCAN** | a maximal chain of dense points | local density, any shape |

We make the differences bite with a question that has an exactly known answer: **how
many connected components does a real algebraic curve have?**

## Files

| File | What it is |
|---|---|
| [`clustering_and_metric_geometry.ipynb`](clustering_and_metric_geometry.ipynb) | The tutorial. 5 sections, 5 figures, 3 exercises. ~1 hour. |
| `README.md` | This file. |

## Running it

```bash
$ conda activate aigeo
$ jupyter lab clustering_and_metric_geometry.ipynb
```

Needs the `aigeo` environment from [Tutorial 1](../tutorial_01/README.md). NumPy and
SciPy only — every algorithm is written out rather than imported, which is about
fifteen lines each and worth reading. Runs in under 10 seconds.

## Contents

| § | Topic | Lecture 7 connection |
|---|---|---|
| 1 | Blobs, concentric circles, and an elliptic curve $y^2 = x^3+px+q$ | geometric datasets |
| 2 | k-means as Voronoi cells — and why it cannot work on the circles | k-means |
| 3 | Spectral clustering; counting components from the spectrum | spectral clustering, graphs |
| 4 | DBSCAN, and recovering the discriminant of an elliptic curve family | density-based methods |
| 5 | Summary | |

## Results worth watching for

**§2 — a failure that is predictable in advance.** k-means scores **50.2%** on the
concentric circles, exactly chance. The reason is printed alongside: the two true
components have the *same centroid*, so no assignment of points to centres can
separate them. k-means is not performing badly; it is answering a different question
correctly.

**§3 — counting components by the spectral gap, including where it breaks.** Three
datasets, one rule (largest jump in the sorted spectrum):

| dataset | first six eigenvalues of $L$ | count |
|---|---|---|
| blobs | `0.0000 0.0000 0.0011 0.0201 0.0219 0.0266` | **3** ✓ |
| circles | `0.0000 0.0000 0.0015 0.0017 0.0026 0.0035` | **2** ✓ |
| elliptic | `0.0000 0.0000 0.0005 0.0005 0.0005 0.0019` | 5 ✗ |

The third row is the valuable one. The two branches are cleanly separated in space,
yet there is no clean gap — because each component is a long, thin curve, and a long
thin graph has a small Fiedler value of its own. Cutting a curve in half is nearly
as cheap as separating the branches. Spectral *counting* works for fat components
and fails for filamentary ones, which is the shape most algebraic curves have. (Given
$k=2$, spectral clustering still *separates* them correctly — it is the counting that
fails.) This motivates §4.

**§4 — an unsupervised method recovers the discriminant.** The family
$y^2 = x^3 - 3x + q$ has two real components exactly when $4p^3 + 27q^2 < 0$, i.e.
for $\lvert q\rvert < 2$. Sweeping $q$ and asking DBSCAN — which knows no algebra —
to count components reproduces the exact count on **88%** of the sweep, with the
disagreements concentrated near $\lvert q\rvert = 2$ where the two branches nearly
touch and the sampled data genuinely cannot distinguish "just connected" from "just
disconnected".

## Exercises

| # | § | Topic |
|---|---|---|
| 1 | 2 | Initialisation, the elbow method, and what an elbow means when the model is wrong |
| 2 | 3 | The graph *is* the model: $k$-NN vs Gaussian kernels; the Fiedler vector |
| **3** | 4 | **How much did $\varepsilon$ decide? Persistent homology; drawing the discriminant cusp** |

## What to take away

- The three algorithms encode **three different definitions of a cluster**. Choosing
  one is a modelling decision about expected geometry, not a matter of accuracy.
- **k-means presumes convex clusters** — diagnose that from the shape of your data,
  not from a bad score.
- **Spectral methods count components with the Laplacian** ($b_0 = \dim\ker\Delta$,
  discretised) — but the gap can vanish for thin sets.
- **The similarity graph is the real model.** Everything after it is linear algebra;
  the geometric judgement is in deciding who is a neighbour.
- **Unsupervised methods can recover exact mathematics** — and are ambiguous exactly
  where the mathematics is degenerate.

## Next

**Lecture 8** keeps the unsupervised setting but asks for coordinates rather than
labels. **Tutorial 8** applies PCA, kernel PCA and nonlinear embeddings to manifolds
where intrinsic and extrinsic geometry disagree.

## Further reading

- von Luxburg, "A tutorial on spectral clustering", *Statistics and Computing* **17** (2007) — the definitive account of §3.
- Ester et al., "A density-based algorithm for discovering clusters", *KDD* 1996 — the original DBSCAN.
- Silverman, *The Arithmetic of Elliptic Curves*, ch. III — the discriminant and the real locus.
- Chung, *Spectral Graph Theory* — the Laplacian facts used here, in full.
