# Tutorial 6 — Attention on Structured Geometric Data

**MM845 — Tópicos de Geometria III: AI for Geometry**
Paired with **Lecture 6: Transformers and Attention**

---

## The idea

A convolution fixes its neighbourhood in advance, because the grid says so. Much
geometric data has no grid: point clouds, sets of simplices, vertices of a
triangulation are **sets**, and the only structure available is what the data
supplies.

Attention is the operator for that situation — it computes from the data *which
elements should talk to which*, then averages accordingly. Its native symmetry is
not translation but **permutation**.

## Files

| File | What it is |
|---|---|
| [`attention_on_geometric_data.ipynb`](attention_on_geometric_data.ipynb) | The tutorial. 5 sections, 4 figures, 3 exercises. ~1 hour. |
| `README.md` | This file. |

## Running it

```bash
$ conda activate aigeo
$ jupyter lab attention_on_geometric_data.ipynb
```

Needs the `aigeo` environment from [Tutorial 1](../tutorial_01/README.md). Runs in
about 15 seconds — the models are small.

## Contents

| § | Topic | Lecture 6 connection |
|---|---|---|
| 1 | Attention written out; permutation equivariance verified | self-attention as matrix factorisation |
| 2 | Point clouds on ellipses, labelled by diameter | structured geometric data |
| 3 | Set attention vs an MLP that sees an ordering | architecture and symmetry |
| 4 | Reading the attention weights — and testing them | interpretation |
| 5 | Summary | |

## Results worth watching for

**§1 — the symmetry, verified.** $\lVert\mathrm{Att}(PX) - P\,\mathrm{Att}(X)\rVert_\infty$
is at machine precision for any permutation $P$, and a mean over elements upgrades
that equivariance to exact invariance.

**§3 — the comparison, with a deliberate twist.** The points arrive **sorted by
angle**, which is how such data usually reaches you (traversal order, scan order).
That gives the input a real, learnable ordering:

| | params | as generated | shuffled |
|---|---|---|---|
| set attention | 6,369 | **0.00108** | **0.00108** |
| MLP on flattened cloud | 37,377 | 0.00537 | 0.06293 |

Predict-the-mean baseline: 0.0761.

The attention model is identical to the last digit — invariance here is algebraic,
not learned. The MLP is respectable on sorted input (with points ordered by angle
the diameter pair sits at roughly opposite indices, a regularity a dense net can
exploit) and degrades twelvefold when shuffled, nearly to baseline. Same points,
same diameter; only the labelling changed, and the labelling was never information.

**§4 — attention maps, and how far to trust them.** The diameter pair receives
**4.95×** the attention of an average point, and attention received correlates
$+0.44$ with distance from the centroid: the model learned to look at the extremes
without being told. The tutorial then argues why that is not yet proof — attention
weights and causal importance are known to come apart — and sets an ablation
(Exercise 2a) as the test that would settle it.

## Exercises

| # | § | Topic |
|---|---|---|
| 1 | 3 | DeepSets vs attention; sorting as a canonicalisation; max vs mean labels |
| 2 | 4 | Attention tested by ablation, not admired |
| **3** | 4 | **When order *does* matter: signed area, and why positional encoding is needed** |

Exercise 3 is the natural counterpoint to the whole tutorial — a task where the set
model provably cannot succeed.

## What to take away

- Attention is a **data-dependent averaging operator** — a convolution whose
  neighbourhood is computed rather than declared.
- Its symmetry is **permutation**, and the invariance is algebraic, not learned.
- **The symmetry mismatch is the story.** A model that can see an ordering will use
  it, whether or not it carries information.
- Attention needs **positional encoding** precisely because it cannot see order.
- **Attention maps are evidence, not explanations.** Ablate before believing.

## Next

**Lecture 7** drops labels entirely: what structure can be recovered from a point
cloud with no supervision at all? **Tutorial 7** uses clustering to detect connected
components and geometric phases.

## Further reading

- Vaswani et al., "Attention is all you need", *NeurIPS* 2017.
- Zaheer et al., "Deep Sets", *NeurIPS* 2017 — the minimal permutation-invariant architecture.
- Lee et al., "Set Transformer", *ICML* 2019 — what §3's model is a stripped-down version of.
- Jain & Wallace, "Attention is not explanation", *NAACL* 2019 — the caution in §4.
