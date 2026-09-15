# Tutorial 11 — Diffusion Models, from a Circle to a Torus

**MM845 — Tópicos de Geometria III: AI for Geometry**
Paired with **Lecture 11: Geometry-Aware ML II — Manifolds & Diffusion Models**

---

## The idea

Every model so far approximated a *function*. Lecture 11 changes the task: given samples
from a measure, build a machine that produces **new samples** from it. Diffusion models
do this in four separate steps — **corrupt** the data with a fixed Gaussian chain,
define the **score** $\nabla_x\log p_t$, learn it by **denoising**, and **sample**
backwards — and this tutorial takes them one at a time on points lying on a circle.

The course's habit of grading methods against known answers pays off unusually well
here. For points on a circle with von Mises-distributed angles, **the noised density and
its score have a closed form in Bessel functions**. That lets us check the learned score
directly, separate the error of *learning* from the error of *sampling*, and locate
precisely where a trained diffusion model is weak. The final section moves to the flat
torus and diffuses *on the manifold itself*, where the heat kernel replaces the Gaussian
and the exact score is again available.

## Files

| File | What it is |
|---|---|
| [`diffusion_on_manifolds.ipynb`](diffusion_on_manifolds.ipynb) | The tutorial. 5 sections, 6 figures, 4 exercises. ~1 hour. |
| `README.md` | This file. |

## Running it

```bash
$ conda activate aigeo
$ jupyter lab diffusion_on_manifolds.ipynb
```

Needs the `aigeo` environment from [Tutorial 1](../tutorial_01/README.md): NumPy, SciPy
(for the scaled Bessel functions `ive`) and PyTorch. Runs top to bottom in about a
minute on a laptop CPU — two small networks are trained, one per part, about 15 seconds
each.

*The seeds are fixed, but the third significant digit of the trained-network results can
move on a different torch build — read the numbers below as the shape of the result.*

## Contents

| § | Topic | Lecture 11 |
|---|---|---|
| 1 | The forward chain; the exact score in Bessel functions; what the score is and is not | slides 7–8 |
| 2 | Learning the score by denoising; the irreducible loss; where the network fails | slide 9 |
| 3 | Sampling backwards with the exact and the learned score; not un-noising | slide 10 |
| 4 | Diffusion *on* the flat torus: wrapped heat kernel, spectral damping, intrinsic steps | slides 3, 11 |
| 5 | What to take away | |

## Results worth watching for

**§1 — the score is exact, and it is not a projection.** The Bessel-function formula for
$\nabla\log p_t$ agrees with finite differences of $\log p_t$ to $3\times10^{-7}$ at
$t=1$, where the score itself has magnitude $1.5\times10^{4}$.

Decomposing it on the circle shows two separate jobs. The radial part pulls onto the
manifold; the tangential part flows *along* it, and at low noise equals the
log-derivative of the angular density to within $0.02$ on a scale of $5.8$. A
streamplot makes this visible: integral curves land on the ring, then run along it into
the dense bumps. Even the radial part does not vanish *on* the noised circle — it equals
$-1/(2\alpha_t)$ there, so the density's ridge sits inside the circle. That is a curvature
effect, derived in Exercise 1(a).

**§2 — the irreducible loss at low noise is the manifold's dimension.** Conditional
expectation is orthogonal projection in $L^2$ (Tutorial 3 §1), so the denoising loss
splits exactly into an irreducible floor plus what training can remove — and with an
exact score, the floor is computable:

| $t$ | $\sigma_t$ | floor (exact) | trained network | relative score error |
|---|---|---|---|---|
| 1 | 0.010 | **0.9995** | 1.639 | 0.631 |
| 5 | 0.059 | 0.993 | 1.024 | 0.029 |
| 30 | 0.355 | 0.772 | 0.777 | 0.004 |
| 80 | 0.787 | 0.393 | 0.394 | 0.0004 |
| 200 | 0.999 | 0.002 | 0.005 | 0.002 |

At $t=1$ the floor is $1 = \dim S^1$. Noise *normal* to the circle is recoverable — it is
how far $x_t$ sits from the circle — but noise *tangent* to it is not, since sliding $x_0$
along the circle gives the same $x_t$. The floor counts tangent directions, so a diffusion
model carries an estimate of **intrinsic dimension**: Tutorial 2 §7's quantity, from a
completely different direction.

The network sits on the floor from $t\approx30$ up, but its score is **63% wrong** at
$t=1$. There the ideal output varies over a distance of $\sigma_1 = 0.01$ — the sharpest
function in the problem, and exactly what Tutorial 4 §4's spectral bias fits last.

**§3 — learning error and sampling error, separated.**

| | mean $\lvert r-1\rvert$ | angle TV | mass near $\theta=0$ |
|---|---|---|---|
| fresh data (finite-sample floor) | 0 | 0.0430 | 0.692 |
| sampler with **exact** score | $5\times10^{-5}$ | 0.0409 | 0.701 |
| sampler with **learned** score | 0.0130 | 0.0533 | 0.703 |

True mass near $\theta=0$: 0.694. With the exact score, 200 Gaussian reverse steps lose
essentially nothing — the angular error is at the finite-sample floor and the points lie
on the circle to $10^{-4}$. With the learned score the *angles* are nearly as good, but
the points land **near** the circle rather than on it. That residual is §2's low-noise
error, concentrated in the final steps, and it is about **twice as large on the sparse
arcs** (0.0245) as where the data are dense (0.0118) — the score is learned only as well as
the data covers it.

A second experiment addresses the delivery note that the reverse chain is not pathwise
inversion. Noise one data point, then run the reverse chain 400 times *from that single
noisy point*. From $t=200$ only **64.5%** of runs return to $x_0$'s side of the circle, close
to that side's 69.4% share of the data; from $t=25$, **100%** do. Both are one statement: from
$x_t$ the chain samples $p(x_0\mid x_t)$, which forgets its origin as the noise grows.

**§4 — diffusion on the torus, graded twice.** On the flat torus the heat kernel is a
wrapped Gaussian, diagonal in the Laplacian's eigenfunctions $e^{i(k_1\theta+k_2\varphi)}$.
The measured damping of each Fourier mode matches $e^{-\lvert k\rvert^2\sigma^2/2}$:

| $\sigma$ | mode $k$ | $\lvert k\rvert^2$ | measured | predicted |
|---|---|---|---|---|
| 0.3 | $(2,-1)$ | 5 | 0.7995 | 0.7985 |
| 0.3 | $(5,0)$ | 25 | 0.3264 | 0.3247 |
| 0.6 | $(3,1)$ | 10 | 0.1664 | 0.1653 |

Slide 3's spectral representation and slide 11's heat kernel turn out to be the same
object. Trained with the network reading periodic features and the dynamics stepping in
intrinsic angles, the learned sampler puts $0.192$–$0.209$ of its mass in each of the
five bumps (true: $0.2$), with spread $0.243$ against the true $0.22$ — and **every sample
lies exactly on $T^2$**. That is not because the network learned the manifold, but because
no step ever left it.

## Exercises

| # | § | Topic |
|---|---|---|
| 1 | 1 | The curvature shift of the density ridge; score of an ellipse; why a flow cannot hit a circle |
| **2** | 2 | **The floor as an intrinsic-dimension estimator on $S^2$; $x_0$- vs $\varepsilon$-prediction; Fourier features** |
| 3 | 3 | The probability-flow ODE; step count vs error; stopping early |
| 4 | 4 | Conditional generation with independent verification; Brownian motion on $S^2$; extrinsic vs intrinsic |

Exercise 2 is the one to do if you do only one: it turns a denoising loss into a
geometric measurement you can test against a manifold whose dimension you know.

## What to take away

- **Generative modelling learns a measure, not a function**, and diffusion does it in four
  separable steps of which only one involves learning.
- **The score is geometry**: onto the manifold, then along it towards high density. It is
  not a nearest-point projection, and curvature shifts its zero set.
- **The achievable denoising loss is computable**, because conditional expectation is a
  projection — and at low noise it equals the intrinsic dimension.
- **Trained models are weakest at low noise**, which is exactly where samples are pulled
  onto the manifold.
- **Generation reverses distributions, not trajectories.**
- **When the manifold is known, diffuse on it**: heat kernel for Gaussian, uniform volume
  for $\mathcal N(0,I)$, exponential map for vector addition.

## Next

**Lecture 12** returns to approximating functions, with a differential equation as the
training signal instead of data: **physics-informed neural networks**, of which the
neural Calabi–Yau metrics of Lecture 11's slide 4 are an example. **Tutorial 12** solves
an elliptic PDE with one.

## Further reading

- Ho, Jain & Abbeel, "Denoising diffusion probabilistic models", *NeurIPS* 2020 — §2 and §3.
- Song et al., "Score-based generative modeling through stochastic differential equations", *ICLR* 2021 — continuous time and the probability-flow ODE of Exercise 3(a).
- Efron, "Tweedie's formula and selection bias", *JASA* **106** (2011) — why predicting noise gives the score.
- De Bortoli et al., "Riemannian score-based generative modelling", *NeurIPS* 2022 — §4 on general manifolds.
- Stanczuk et al., "Your diffusion model secretly knows the dimension of the data manifold", arXiv:2212.12611 — §2's floor, made into an estimator.
