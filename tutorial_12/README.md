# Tutorial 12 — Physics-Informed Neural Networks on a Disk and a Sphere

**MM845 — Tópicos de Geometria III: AI for Geometry**
Paired with **Lecture 12: Physics-Informed Neural Networks for Geometry I**

---

## The idea

Until now every network learned from data. A **physics-informed neural network** learns
from an equation instead: the network $u_\theta$ is a trial function, and training makes
the PDE hold at sampled collocation points, with autodiff supplying the derivatives.

This tutorial does what Lecture 12's summary promises: **Poisson on a disk and on a
surface, compared with classical baselines**. Every problem has a manufactured exact
solution, so every error reported is a real error, not a training loss. It also keeps the
lecture's promise to be honest about the comparison. On these problems a thirty-line
finite-element solver wins on speed. What the PINN offers instead is flexibility: going
from the disk to the sphere changes *only the differential operator*.

## Files

| File | What it is |
|---|---|
| [`pinns_on_a_disk_and_a_sphere.ipynb`](pinns_on_a_disk_and_a_sphere.ipynb) | The tutorial. 5 sections, 3 figures, 4 exercises. ~1 hour. |
| `README.md` | This file. |

## Running it

```bash
$ conda activate aigeo
$ jupyter lab pinns_on_a_disk_and_a_sphere.ipynb
```

Needs the `aigeo` environment from [Tutorial 1](../tutorial_01/README.md): NumPy, SciPy
(sparse solvers, `Delaunay`, `ConvexHull`) and PyTorch. Runs top to bottom in a little
over a minute on a laptop CPU. Six small networks are trained, about 10 seconds each, all
in `float64`, because PINNs aiming at $10^{-5}$ accuracy need it.

*The seeds are fixed, but trained-network errors can move in the second significant digit
on a different torch build, and all timings depend on your machine. Read the numbers below
as the shape of the result.*

## Contents

| § | Topic | Lecture 12 |
|---|---|---|
| 1 | Autodiff Laplacian, hard boundary constraint, Adam then L-BFGS, the error measured honestly | slides 4, 5, 7 |
| 2 | Soft vs hard boundary conditions; a P1 finite-element baseline and its convergence | slides 3, 6, 11 |
| 3 | Laplace–Beltrami on $S^2$ by autodiff; the curvature term; gauge fixing; cotangent FEM | slide 8 |
| 4 | Deep Ritz: the Dirichlet energy, quadrature, and an energy that certifies the solution | slide 9 |
| 5 | What to take away | |

## Results worth watching for

**§1 — the derivatives are exact, the solution is not.** The problem is slide 7's:
$-\Delta u = 4 + 8x_1$ on the unit disk, $u = 0$ on the circle, with
$u^\ast = (1-\lVert x\rVert^2)(1+x_1)$. The autodiff Laplacian of $u^\ast$ matches $-f$ to
$8.9\times10^{-16}$, so the errors in a PINN come from approximation and sampling, never
from differentiation. With the hard ansatz $u_\theta = (1-\lVert x\rVert^2)N_\theta$, 1000 Adam
steps on freshly resampled points reach a relative $L^2$ error of $9.4\times10^{-4}$. A
300-iteration L-BFGS finish brings it to **$5.4\times10^{-5}$**, and in the training
curve the loss visibly drops off a cliff when L-BFGS starts.

**§2 — soft constraints compete, and FEM wins the race.**

| boundary condition | rel. $L^2$ error | max $\lvert u\rvert$ on circle | residual loss |
|---|---|---|---|
| hard, $(1-\lVert x\rVert^2)N_\theta$ | **$5.4\times10^{-5}$** | $4\times10^{-16}$ | $5.3\times10^{-6}$ |
| soft, $\lambda_b = 1$ | $3.7\times10^{-4}$ | $1.3\times10^{-3}$ | $1.1\times10^{-6}$ |
| soft, $\lambda_b = 100$ | $1.8\times10^{-4}$ | $4.7\times10^{-4}$ | $5.3\times10^{-5}$ |

Raising $\lambda_b$ a hundredfold cuts the boundary error by a factor of three and raises
the residual fiftyfold: this is slide 11's loss imbalance, and no fixed weight makes it
go away. The soft $\lambda_b = 1$ network even has the *lowest* residual of the three,
yet it is seven times less accurate than the hard one. A small residual does not mean
a correct solution.

The P1 finite-element baseline, with the local stiffness $e_a\cdot e_b/(4A)$ and a sparse
solve:

| nodes | 135 | 471 | 1,742 | 6,684 | 26,167 | 52,002 |
|---|---|---|---|---|---|---|
| rel. $L^2$ error | $2.3\times10^{-2}$ | $5.6\times10^{-3}$ | $1.5\times10^{-3}$ | $3.5\times10^{-4}$ | $9.6\times10^{-5}$ | $4.9\times10^{-5}$ |
| time | 1 ms | 3 ms | 10 ms | 38 ms | 0.22 s | 0.6 s |

This is clean $O(h^2)$ convergence: four times the nodes gives a quarter of the error. The
best PINN took about 11 s. FEM matches its accuracy in about 0.6 s. **For a standard 2-D
forward problem, the PINN is not the tool.** Exercise 2(c) shows where that changes.

**§3 — geometry enters through the operator.** For an ambient network restricted to $S^2$,

$$\Delta_{S^2}u = \operatorname{tr}H - x^\top Hx - 2\,x\cdot\nabla u .$$

Checked on spherical harmonics:

| harmonic | $\ell$ | error, full operator | error, tangential Hessian only |
|---|---|---|---|
| $z$ | 1 | $0$ | 2.00 |
| $xy$ | 2 | $0$ | 2.00 |
| $x^3 - 3xy^2$ | 3 | $4\times10^{-15}$ | 5.96 |

Without the curvature term the operator reports the eigenvalue $\ell(\ell-1)$ instead of
$\ell(\ell+1)$, because Euler's theorem gives $x\cdot\nabla u = \ell u$ for a homogeneous
harmonic. A PINN trained with that operator converges happily to the wrong answer.

On the closed sphere the solution is only defined up to a constant, so the loss adds a
gauge term $(\text{mean } u)^2$, and errors are measured after subtracting the mean. For
$u^\ast = e^z + xy - \sinh 1$, the same network and recipe as on the disk reach
**$4.6\times10^{-5}$** in about 14 s. The cotangent FEM, solved as a bordered system with
a Lagrange multiplier for $\int u = 0$, again converges as $O(h^2)$:

| nodes | 200 | 800 | 3,200 | 12,800 | 51,200 |
|---|---|---|---|---|---|
| rel. $L^2$ error | $2.7\times10^{-2}$ | $6.1\times10^{-3}$ | $1.6\times10^{-3}$ | $3.9\times10^{-4}$ | $9.9\times10^{-5}$ |
| time | 2 ms | 5 ms | 19 ms | 94 ms | 0.75 s |

**§4 — Deep Ritz: quadrature decides the answer, and the energy certifies it.**

| method | rel. $L^2$ error | $E[u] - E[u^\ast]$ | $\tfrac12\lVert\nabla(u-u^\ast)\rVert^2$ |
|---|---|---|---|
| residual PINN (§1) | $5.4\times10^{-5}$ | | |
| Deep Ritz, 4000 random points | $4.1\times10^{-2}$ | $5.32\times10^{-2}$ | $5.32\times10^{-2}$ |
| Deep Ritz, 4000-point sunflower lattice | $7.3\times10^{-4}$ | $1.79\times10^{-5}$ | $1.79\times10^{-5}$ |

The same budget and the same optimiser give a **56-fold** difference, decided only by the
point set. The residual loss is minimised by $u^\ast$ on *any* point set. The energy is an
integral, and a poor quadrature changes which function minimises it. The last two columns
agree to three digits, confirming the identity
$E[u] - E[u^\ast] = \tfrac12\int\lVert\nabla(u-u^\ast)\rVert^2$. So a lower energy is a
certificate: it ranks solutions by their true error *without knowing the true solution*,
which slide 11 says no residual can do.

## Exercises

| # | § | Topic |
|---|---|---|
| 1 | 1 | L-BFGS vs more Adam; why ReLU fails; residual-based adaptive sampling |
| **2** | 2 | **The $\lambda_b$ Pareto front; gradient-balanced weights; Poisson on the ball in $d = 2,\dots,20$, where meshes die** |
| 3 | 3 | Training with the wrong operator; radial-projection inputs; the first eigenfunction of $S^2$ |
| 4 | 4 | Ranking solutions by energy; resampling vs fixed points; **an inverse conductivity problem** |

Exercise 2(c) is the one to do if you do only one. It is the honest counterpart to §2's
verdict: the same PINN code runs in 20 dimensions, where no mesh can. Exercise 4(c) makes
a good mini-project seed.

## What to take away

- **A PINN is a network used as a trial function.** Autodiff gives exact derivatives, so
  the errors come from approximation and sampling. Measure them against something other
  than the training loss.
- **Build boundary conditions into the ansatz when you can.** A soft penalty makes two
  losses compete, and no single weight removes the trade-off.
- **Benchmark against a classical solver and report the result honestly.** Here FEM was
  faster at equal accuracy, with a predictable convergence rate. PINNs earn their place on
  hard-to-mesh domains, in high dimension, and for inverse and parametric problems.
- **On a surface only the operator changes**, provided it is the right operator:
  forgetting the curvature term turns $\ell(\ell+1)$ into $\ell(\ell-1)$.
- **Residuals and energies fail differently.** An energy loss inherits its quadrature
  error, but a lower energy certifies a better solution.

## Next

**Lecture 13** adds time and nonlinearity, with heat-type and curvature flows. It turns
slide 11's failure modes into remedies and introduces neural operators. **Tutorial 13**
applies PINNs to a time-dependent geometric flow.

## Further reading

- Raissi, Perdikaris & Karniadakis, "Physics-informed neural networks", *J. Comput. Phys.* **378** (2019). The framework of §1.
- E & Yu, "The Deep Ritz method", *Commun. Math. Stat.* **6** (2018). §4.
- Lu et al., "Physics-informed neural networks with hard constraints for inverse design", *SIAM J. Sci. Comput.* **43** (2021). Hard constraints, §2.
- Wang, Teng & Perdikaris, "Understanding and mitigating gradient flow pathologies in physics-informed neural networks", *SIAM J. Sci. Comput.* **43** (2021). Exercise 2(b).
- Crane, de Goes, Desbrun & Schröder, "Digital geometry processing with discrete exterior calculus", SIGGRAPH course (2013). The cotangent Laplacian of §3.
