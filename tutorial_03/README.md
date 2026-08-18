# Tutorial 3 — Variational Problems, Spectra, and Linear Models on Geometric Data

**MM845 — Tópicos de Geometria III: AI for Geometry**
Paired with **Lecture 3: Linear Models, Optimisation & Regularisation**

---

## The idea

Lecture 3 is the one lecture where everything can be proved: least squares is an
orthogonal projection, the loss is convex, the minimiser is unique and closed-form,
and the convergence rate of gradient descent is a ratio of eigenvalues. This tutorial
spends that certainty on geometry.

It covers two syllabus items, and they turn out to be the same idea twice.

**Part I — *Variational Problems and Function Approximation on Manifolds*.**
Approximating a function on $S^2$ *is* orthogonal projection in $L^2(S^2)$. Choose the
eigenbasis of the Laplace–Beltrami operator and the projection becomes diagonal, the
condition number drops to $1$, and Tikhonov regularisation becomes literally a penalty
on the Dirichlet energy $\int_M|\nabla f|^2$. We also recover that eigenbasis from a
bare point cloud with a graph Laplacian.

**Part II — *Linear Models for Geometric Classification and Regression*.** We build
random plane curves, label them with quantities computable in closed form (area,
length, convexity), and predict those labels by linear and logistic regression. The
theme is Lecture 3's design principle: the feature map is worth more than the model.

## Files

| File | What it is |
|---|---|
| [`variational_and_linear_models.ipynb`](variational_and_linear_models.ipynb) | The tutorial. 8 sections, 10 figures, 6 exercises. |
| `README.md` | This file. |

## Running it

```bash
$ conda activate aigeo
$ jupyter lab variational_and_linear_models.ipynb
```

Needs the `aigeo` environment from [Tutorial 1](../tutorial_01/README.md). NumPy and
SciPy throughout; **PyTorch is used in §8 only**. Runs top to bottom in about a
minute, most of it in §3 (a dense eigendecomposition) and §8 (two PyTorch fits).

This is the longest tutorial in the course, because it carries two syllabus items.
If you are short of time, **§1–2 and §5–6 are the core**; §3 and §7 are the most
self-contained to skip.

## Contents

| § | Topic | Lecture 3 |
|---|---|---|
| **Part I** | | |
| 1 | Least squares as orthogonal projection; empirical vs exact inner product | slide 3 |
| 2 | The Laplace–Beltrami eigenbasis, and why the basis decides $\kappa$ | slides 5–6, 9 |
| 3 | Recovering that eigenbasis from a bare point cloud | — |
| 4 | Spectral decay, and ridge $=$ Dirichlet energy | slides 3, 8 |
| **Part II** | | |
| 5 | Random plane curves; symmetry and invariant features | slides 9, 10 |
| 6 | Regression: area (exactly) and length (identifiably) | slides 3, 5–6, 11 |
| 7 | Classification, ridge and lasso | slides 4, 8 |
| 8 | PyTorch: the canonical training loop | slide 11 |

## Results worth watching for

**§2 — the numerics rediscover spherical harmonics, and the basis decides $\kappa$.**
Degree-ordered Gram–Schmidt on monomials produces blocks of size $1,3,5,7,9,11,13$ —
that is $\dim\mathcal{H}_\ell = 2\ell+1$ — without a single spherical harmonic being
written down. The same subspace in two bases then gives
$\kappa \approx 1.2\times10^4$ (monomials) versus $\kappa \approx 1.5$ (harmonics):
about 28,000 gradient steps to gain a digit, versus about 4.

**§3 — the spectrum of $S^2$ from distances alone.** A graph Laplacian built only from
pairwise distances returns eigenvalues $0, 2, 6, 12, 20$ with multiplicities
$1,3,5,7,9$. The multiplicities are the convincing part — they are integers, so no
fitted constant can fake them. The eigenvalues are biased low, and the bias is $O(t)$:
halve the kernel bandwidth, halve the error.

**§4 — regularisation is geometry.** In the harmonic basis the Dirichlet energy is
$\sum_k \lambda_k c_k^2$, so weighted ridge with $w_k = \ell(\ell+1)$ *is* a penalty on
$\int|\nabla f|^2$. The fitted coefficients satisfy
$\hat c_k = \hat c_k^{\mathrm{LS}}/(1+\lambda\lambda_k)$ — verified to $3\times10^{-4}$,
with the measured shrinkage factors falling exactly on that curve — and the Dirichlet
penalty beats plain ridge by about 35% in test MSE.

**§6 — a model that recovers $\pi$.** By Parseval the area of a polar curve is exactly
$\pi a_0^2 + \frac{\pi}{2}\sum_k(a_k^2+b_k^2)$, hence *linear* in the rotation-invariant
power spectrum. Least squares returns $R^2 = 1.000000000000$, intercept $=\pi$ to ten
decimals, and eight weights all $=\pi/2$ to thirteen. On the raw Fourier coefficients —
more features, same data — it returns $R^2 = 0.003$.

The length regression is the counterweight: it recovers the second-order prediction
$\frac{\pi}{2}k^2$ for low $k$ and wanders for high $k$. That is *unidentifiability*,
not error — $p_k$ has standard deviation falling like $k^{-5}$, so by $k=8$ the feature
varies too little to determine its own coefficient. Bootstrap error bars make the
distinction visible, and it is the distinction between a result and a number.

Incidentally the weights $k^2$ are the eigenvalues of $-d^2/d\theta^2$: Part I's
Dirichlet energy reappears, unprompted, as the answer to a regression about arc length.

**§8 — a small loss is not a right answer.** The canonical PyTorch loop, run for 20000
epochs, reaches training MSE $3\times10^{-11}$ and gets the last three weights badly
wrong. Re-run with a flatter spectrum — same optimiser, same epochs, same target, only
the conditioning changed — and it returns $\pi/2$ in every coordinate to
$2\times10^{-16}$. Training longer does not fix the first case; nothing does, because
those directions of parameter space are flat.

## Exercises

| # | § | Topic |
|---|---|---|
| 1 | 1 | The projection theorem and Pythagoras, numerically |
| 2 | 3 | Breaking the symmetry: degeneracies split on an ellipsoid; Weyl's law |
| 3 | 4 | Other regularisers, coloured noise, and the zig-zag predicted by $\kappa$ |
| **4** | 6 | **What invariance buys: augmentation vs invariant features** |
| 5 | 7 | Better features for convexity; ridge vs lasso paths; elastic net |
| 6 | 8 | The condition number, visible in training curves |

Exercise 4 is the one to do if you do only one — it is the Part II analogue of
Tutorial 2's orbit-splitting trap.

## What to take away

- **Least squares is orthogonal projection**, on a manifold as much as in
  $\mathbb{R}^n$; the design matrix is a discretised Gram matrix, and having only
  samples means having only an approximate inner product.
- **The Laplacian's eigenbasis is the good basis** — it diagonalises the problem,
  makes $\kappa\approx1$, and is recoverable from a bare point cloud.
- **Regularisation is geometry.** Choosing a penalty is choosing which functions you
  consider plausible; on a manifold that choice has a name and a formula.
- **Approximation error is coefficient decay**, hence regularity. No optimiser repairs
  a poor ansatz.
- **The feature map is the model.** Invariant features made the area regression exact
  and its weights readable; raw coordinates, with more parameters, gave nothing.
- **Read the error bars, not the point estimates.** Coefficients are trustworthy only
  where their features vary.

## Next

**Lecture 4** replaces the fixed feature map $\phi$ with a *learned* one — neural
networks — giving up convexity, closed forms and guarantees in exchange for expressive
power. Because this tutorial pinned down what those guarantees were worth,
[Tutorial 4](../tutorial_04/) can say precisely what the trade bought.

## Further reading

- Atkinson & Han, *Spherical Harmonics and Approximation on the Unit Sphere* (Springer, 2012) — Part I in full rigour.
- Belkin & Niyogi, "Laplacian eigenmaps for dimensionality reduction and data representation", *Neural Computation* **15** (2003) — the construction in §3.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, ch. 3 — ridge, lasso, and the ball picture. Free online.
- Trefethen, *Approximation Theory and Approximation Practice* (SIAM, 2013), ch. 7–8 — why smoothness governs decay.
