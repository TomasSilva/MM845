# Tutorial 9 — Reinforcement Learning on Discrete Geometric Structures

**MM845 — Tópicos de Geometria III: AI for Geometry**
Paired with **Lecture 9: Introduction to Reinforcement Learning**

---

## The idea

Every tutorial so far started from a dataset. RL does not have one: there is a state
space, a set of legal moves, and a reward, and the agent generates its own data by
acting.

For a geometer the appeal is that many natural problems already have this shape — a
triangulation with Pachner moves, a knot diagram with Reidemeister moves, a polytope
with bistellar flips. Asking for an optimal sequence of moves *is* asking for a
policy.

We do two problems, both chosen because the answer is **exactly checkable**: shortest
paths (graded against Dijkstra) and edge flips (graded against the Delaunay theorem).

## Files

| File | What it is |
|---|---|
| [`rl_on_discrete_geometry.ipynb`](rl_on_discrete_geometry.ipynb) | The tutorial. 5 sections, 5 figures, 2 exercises. ~1 hour. |
| `README.md` | This file. |

## Running it

```bash
$ conda activate aigeo
$ jupyter lab rl_on_discrete_geometry.ipynb
```

Needs the `aigeo` environment from [Tutorial 1](../tutorial_01/README.md). NumPy and
SciPy only — Q-learning, Dijkstra and the flip machinery are all written out. Runs in
about 30 seconds.

## Contents

| § | Topic | Lecture 9 connection |
|---|---|---|
| 1 | An MDP on a triangulation; Bellman; planning vs learning | MDPs, value functions |
| 2 | Tabular Q-learning, graded against Dijkstra | Q-learning |
| 3 | Stability: what $\varepsilon$, $\alpha$, $\gamma$ actually do | stability challenges |
| 4 | A combinatorial MDP: edge flips, rediscovering Delaunay | discrete action spaces |
| 5 | Summary | |

## Results worth watching for

**§2 — exactly optimal, and a value function that is only locally right.** The greedy
policy recovers Dijkstra's path length to four decimals (1.5062, 0.00% excess). But
the value function tells a subtler story:

```
states with any learned value : 93 of 94
correlation, all states       : 0.356
correlation, visited states   : 1.000
```

On visited states Q-learning has solved the Bellman equation essentially perfectly.
**A single unvisited vertex** — $Q$ still at its initial zero, true value about $-2$ —
drags the overall correlation from $1.00$ to $0.36$.

Two independent lessons are drawn. Q-learning is not a solver for the whole Bellman
equation: it improves estimates only along the trajectories its policy generates,
which is a liability if you want a full value map and exactly the right trade when
the state space cannot be enumerated. And, echoing Tutorial 8's $0.974$ that
concealed real structure: a correlation coefficient is a one-number summary of a
scatter plot, and you should look at the scatter plot.

**§3 — three knobs that fail in three different ways.**

| knob | behaviour |
|---|---|
| $\varepsilon_{\min}$ | optimal at *every* setting, 0.0 to 0.5 |
| $\alpha$ | fails at 0.05 (+32.8%), fine from 0.2 up |
| $\gamma$ | +32.8% below 0.97, optimal only at 1.0 |

The $\varepsilon$ row is explained rather than glossed: this problem has dense rewards
and a small well-connected graph, so exploration is not the binding constraint — on a
sparse-reward problem the panel would look completely different. And $\gamma$ is not a
tuning constant at all: below $1$ the agent is correctly solving a *different*,
short-sighted problem.

**§4 — Q-learning rediscovers Delaunay.** From a deliberately scrambled triangulation
(min angle **0.94°**), searching only by edge flips with reward = improvement in the
minimum angle, the agent reaches **12.842°** — exactly the Delaunay optimum, the
triangulation that provably maximises the minimum angle. It visits 3250 states and
never sees the empty-circumcircle criterion.

The section is careful about what this does and does not show: the flip algorithm
solves this in polynomial time and is provably correct, while Q-learning is slower
and proves nothing. What transfers is the **framing** — a space of combinatorial
structures, legal moves, a scalar to maximise — which applies verbatim where no flip
theorem exists. The printed state count also shows why tabular methods die: the number
of triangulations grows exponentially, which is exactly the motivation for Lecture 10.

## Exercises

| # | § | Topic |
|---|---|---|
| 1 | 2 | Planning vs learning; stochastic transitions; geodesics on a curved surface |
| 2 | 4 | Minimum-weight triangulation; state-space growth; a function approximator |

## What to take away

- **RL needs no dataset** — and much of discrete geometry already has the required
  structure.
- **Q-learning is off-policy**: it learns the value of behaving optimally while
  behaving randomly.
- **Check against exact answers when you can** — Dijkstra and the Delaunay theorem
  turned plausible results into measured ones.
- **$\gamma$ chooses the problem.** A disappointing RL result is often the correct
  solution to a different objective.
- **Tabular methods die with the state space**, which is why function approximation —
  and for geometric states, *geometric* function approximation — comes next.

## Next

**Lecture 10** builds symmetry into the architecture in general: group actions,
equivariance, GNNs. **Tutorial 10** compares invariant and non-invariant models
directly.

## Further reading

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), ch. 6. Free online.
- Wagner, "Constructions in combinatorics via neural networks", arXiv:2104.14516 — RL used to refute conjectures.
- Lawson, "Transforming triangulations", *Discrete Mathematics* **3** (1972) — the flip theorem behind §4.
- de Berg et al., *Computational Geometry*, ch. 9 — Delaunay and the angle-optimality proof.
