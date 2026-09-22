# Tutorial 13 — Time-Dependent Flows and Neural Operators

**MM845 — Tópicos de Geometria III: AI for Geometry**
Paired with **Lecture 13: PINNs for Geometry II & Neural Operators**

---

## The idea

Tutorial 12 solved static problems. This one adds time, and then changes the question
entirely.

A **space-time PINN** treats $t$ as just another input, so one network holds a whole
trajectory. The convenience hides a trap: the residual treats every instant alike, and
nothing tells the optimiser to get the early dynamics right before the late ones. Lecture
13 spends two slides on the cures, and this tutorial measures which of them actually pay.

Then **geometric flows** — curves moving by their own curvature — which are the reason
this audience is here. They also come with a gift: exact laws like $R(t)=\sqrt{R_0^2-2t}$
and $dA/dt=-2\pi$ that verify a network *without ever knowing the solution*. Every
geometric problem has invariants of this kind, and finding them is the most valuable hour
you can spend on a project.

Finally **neural operators**, which learn the solution *map* rather than one solution, and
close the course by amortising work across a whole family of problems.

## Files

| File | What it is |
|---|---|
| [`time_dependent_flows_and_operators.ipynb`](time_dependent_flows_and_operators.ipynb) | The tutorial. 5 sections, 4 figures, 4 exercises. ~1 hour. |
| `README.md` | This file. |

## Running it

```bash
$ conda activate aigeo
$ jupyter lab time_dependent_flows_and_operators.ipynb
```

Needs the `aigeo` environment from [Tutorial 1](../tutorial_01/README.md): NumPy and
PyTorch only. Runs top to bottom in about three minutes on a laptop CPU — the longest of
the course, because it trains fourteen small networks in `float64`.

*Seeds are fixed, but trained-network errors can move in the second significant digit on a
different torch build, and all timings depend on your machine. Read the numbers below as
the shape of the result.*

## Contents

| § | Topic | Lecture 13 |
|---|---|---|
| 1 | Time as an input; hard initial conditions; periodicity from Laplacian eigenfunctions | slide 3 |
| 2 | Propagation failure over long horizons; causal weights, time features, window marching | slides 6, 7 |
| 3 | Curve shortening flow, checked against exact geometric laws; the singularity | slide 4 |
| 4 | DeepONet and a spectral (FNO) layer; discretisation invariance and its limits | slides 8–10 |
| 5 | What to take away, the assembled toolkit, and the mini-project | slides 11, 12 |

## Results worth watching for

**§1 — the ansatz is worth two orders of magnitude.** The heat equation on the circle,
where every Fourier mode decays as $e^{-k^2t}$, so the exact solution is known. Same
architecture, same budget, three ways of imposing the same two constraints:

| setup | rel. $L^2$ error |
|---|---|
| periodic features + hard IC $u_0(x)+t\,N_\theta$ | **$2.1\times10^{-3}$** |
| periodic features + soft IC | $3.3\times10^{-2}$ |
| plain $(x,t)$ + soft IC + soft periodicity | $2.8\times10^{-1}$ |

A factor of 135 between the first and last row, for free. Two checks that need no exact
solution also pass: the error at $t=0$ is $5\times10^{-17}$ by construction, and the energy
$\int u^2$ decreases monotonically, ending at $1.4471$ against the exact $1.4508$.

**§2 — the horizon is the difficulty.** Transport, $u_t + 2u_x = 0$, whose solution
translates without ever decaying — the opposite of the heat equation's forgiving smoothing.
One network, one budget, three horizons:

| horizon $T$ | 1 | 2 | 4 |
|---|---|---|---|
| rel. $L^2$ error | $1.7\times10^{-2}$ | $2.0\times10^{-1}$ | $7.7\times10^{-1}$ |

At $T=4$ the prediction is no better than zero. The cures, all at $T=4$:

| cure | time | rel. $L^2$ error |
|---|---|---|
| vanilla | 6 s | $7.7\times10^{-1}$ |
| causal weighting $w_i=\exp(-\epsilon\sum_{j<i}r_j)$ | 6 s | $5.0\times10^{-1}$ |
| Fourier features in $t$ as well as $x$ | 7 s | $1.3\times10^{-1}$ |
| marching over 4 time windows | 23 s | $1.2\times10^{-1}$ |

The two structural cures win by about $6\times$ and land within a factor of two of each
other; the reweighting manages only $1.5\times$ — and its final weights never fall below
$0.71$, meaning the causal gate barely closed. That parameter needs tuning per problem,
and the notebook says so rather than presenting the method as a fix.

The error-versus-time panel is the one to show students. The vanilla network is already
65% wrong at $t=1$ and 97% by $t=4$: **the error grows into the future**, which is exactly
the propagation failure of slide 3. Marching grows too — each window inherits the previous
one's error — but stays five to eight times lower throughout.

**§3 — a flow you can verify without the answer.** Curve shortening,
$\partial_t X = \partial_s^2 X$, with the arclength derivative built by autodiff through
the curve's own induced metric. For a circle the exact law is $R(t)=\sqrt{1-2t}$, extinct
at $t=0.5$:

| trained to | fraction of extinction time | max relative radius error |
|---|---|---|
| $T=0.3$ | 60% | **$9.1\times10^{-3}$** |
| $T=0.45$ | 90% | $4.9\times10^{-1}$ |

Same network, same budget, fifty times worse — bought by nothing but moving the finish
line. This is slide 4's "singularities are the hard part", made quantitative, and it
motivates the lecture's remedies (rescaling, adaptive sampling, weak formulations).

For an ellipse there is no closed form, so the check is the **area law**: for *any*
embedded closed curve $dA/dt = -\oint\kappa\,ds = -2\pi$, by the turning-number theorem.
Trained with §2's own cure — three time windows — against a classical polygon solver:

| | fitted $dA/dt$ | max $\lvert A - (A_0-2\pi t)\rvert$ | time |
|---|---|---|---|
| PINN, 3 windows | $-6.047$ | $7.7\times10^{-2}$ | 43 s |
| polygon baseline | $-6.331$ | $1.5\times10^{-2}$ | 1.0 s |
| exact | $-2\pi = -6.283$ | 0 | |

The classical solver is again far cheaper and more accurate — Tutorial 12's verdict
survives into time-dependent problems. Meanwhile the isoperimetric ratio falls from
$1.2305$ to $1.0487$: the ellipse is becoming round, which is Gage–Hamilton–Grayson
appearing in a numerical experiment.

**§4 — twenty parameters beat twenty-five thousand.** Learning the heat semigroup
$u_0\mapsto u(\cdot,T)$ at $T=0.05$, with training data generated exactly:

| architecture | parameters | training | test rel. $L^2$ |
|---|---|---|---|
| DeepONet (branch/trunk) | 25,792 | 2.2 s | $3.7\times10^{-2}$ |
| one spectral (FNO) layer | **20** | 0.2 s | **$1.3\times10^{-13}$** |

The spectral layer learns each multiplier to machine precision — $|R_4| = 0.449329$
against the exact $e^{-16T} = 0.449329$ — because on the circle the Laplacian's
eigenfunctions *are* the Fourier modes, so the true operator lies inside the layer's
hypothesis space. It is slide 10's geometric operator learning in the simplest geometry,
and Lecture 10's symmetry argument applied to an operator.

Two consequences follow, one encouraging and one cautionary:

- **Discretisation invariance is real.** The same weights, trained on 64 grid points, give
  *identical* error ($1.3\times10^{-13}$) on 128 and 256 points. The multipliers attach to
  modes, not to grid points.
- **You only learn what the data excites.** Training inputs contained modes $k\le 8$, so
  the multiplier for $k=9$ never received a gradient and sits at its initial value of
  $1.000000$ instead of $0.017422$. Feed it inputs with modes up to 14 and the error jumps
  to $9.8\times10^{-2}$ — and a finer grid does not repair it. An operator is accurate on
  the family it was trained on, and nothing guarantees it beyond.

## Exercises

| # | § | Topic |
|---|---|---|
| 1 | 1 | The $\lambda_0$ trade-off; a discrete-time (backward Euler) PINN; too few Fourier features |
| 2 | 2 | Tuning $\epsilon$ until the causal gate closes; gradient balancing; window count vs horizon; FBPINN |
| **3** | 3 | **Parametrisation health; non-convex initial curves; the level-set formulation and topology change; rescaling the singularity** |
| 4 | 4 | Retraining on wider data; nonlinear operators; a physics-informed operator; a parametric family |

Exercise 3 is the one to do if you do only one: it takes the flow into the regimes where
the parametric formulation genuinely breaks and the level-set one does not.

## What to take away

- **Time is an input, but the optimiser does not know time flows.** Nothing in a space-time
  residual makes the network learn the early dynamics first.
- **Build the initial condition into the ansatz**, and on a periodic domain use the
  Laplacian's eigenfunctions as features.
- **The horizon is the difficulty**, and marching in windows turns one hard problem into
  several easy ones.
- **Geometric flows come with exact laws.** Use them; they verify a network with no
  reference solution.
- **Singularities are the honest limit** of a smooth global ansatz.
- **Operators amortise**, transfer across discretisations, and hold only over the family
  they were trained on.

## Next

This is the last tutorial. **The mini-project**: a 10-minute presentation (01/10), a
five-page report (04/10), and a reproducible Git repository — environment, seeds, README,
code. One clear question, a dataset you understand, a method matched to it, a comparison
against a simple baseline, and independent verification of whatever the model suggests.

A modest, well-understood result is enough. So is a negative result with a clear
diagnosis — and this tutorial is largely a demonstration that measuring where a method
*fails* is often more informative than the headline number.

Boa sorte!

## Further reading

- Raissi, Perdikaris & Karniadakis, "Physics-informed neural networks", *J. Comput. Phys.* **378** (2019).
- Krishnapriyan et al., "Characterizing possible failure modes in physics-informed neural networks", *NeurIPS* 2021 — §2's horizon experiment, done properly.
- Wang, Sankaran & Perdikaris, "Respecting causality for training physics-informed neural networks", *CMAME* **421** (2024) — §2's causal weights.
- Moseley, Markham & Nissen-Meyer, "Finite basis physics-informed neural networks", *Adv. Comput. Math.* **49** (2023) — Exercise 2(d).
- Lu, Jin, Pang, Zhang & Karniadakis, "Learning nonlinear operators via DeepONet", *Nat. Mach. Intell.* **3** (2021) — §4.
- Li et al., "Fourier neural operator for parametric partial differential equations", *ICLR* 2021 — §4's spectral layer.
- Gage & Hamilton, *J. Diff. Geom.* **23** (1986); Grayson, *J. Diff. Geom.* **26** (1987) — the theorems behind §3.
