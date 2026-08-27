# Tutorial 6 — Attention on Structured Geometric Data

**MM845 — Tópicos de Geometria III: AI for Geometry**
Paired with **Lecture 6: Transformers and Attention**

---

## The idea

A convolution fixes its neighbourhood in advance, because the grid says so. Much
geometric data has no grid: point clouds, sets of simplices, vertices of a
triangulation are **sets** — unordered, and not even all of the same size — and the
only structure available is what the data supplies.

Attention is the operator for that situation — it computes from the data *which
elements should talk to which*, then averages accordingly. Its native symmetry is
not translation but **permutation**.

## Files

| File | What it is |
|---|---|
| [`attention_on_geometric_data.ipynb`](attention_on_geometric_data.ipynb) | The tutorial. 5 sections, 3 figures, 3 exercises. ~1 hour. |
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
| 2 | Point clouds on ellipses, varying in size, labelled by diameter | structured geometric data |
| 3 | A transformer that never pads vs a padded MLP; ragged batching | architecture and symmetry |
| 4 | Reading the attention weights — and testing them | interpretation |
| 5 | Summary | |

## Results worth watching for

*Recorded in `aigeo` with torch 2.13, where they reproduce byte-for-byte between
runs. The seeds are fixed, but the third significant digit still moves on a different
torch build — treat the figures below as the shape of the result, not a checksum.*

**§1 — the symmetry, verified.** $\lVert\mathrm{Att}(PX) - P\,\mathrm{Att}(X)\rVert_\infty$
is at machine precision for any permutation $P$, and a mean over elements upgrades
that equivariance to exact invariance.

**§3 — the comparison, with a deliberate twist.** Each cloud carries its own number
of points (10–22) and is stored as its own array — the dataset is a *list*, not a
tensor. The points arrive **sorted by angle**, which is how such data usually reaches
you (traversal order, scan order). That gives the input a real, learnable ordering:

| | params | pads? | as generated | shuffled |
|---|---|---|---|---|
| set transformer | 10,721 | **no** | **0.00095** | **0.00095** |
| MLP on flattened cloud | 38,913 | yes | 0.01107 | 0.05566 |

Predict-the-mean baseline: 0.0784.

The transformer wins on every axis: **11.6× more accurate** with **a third of the
parameters**, and identical to the last printed digit under shuffling — invariance
here is algebraic, not learned. Not one of its 1500 predictions moves when the points
are reordered; all 1500 of the MLP's do. The MLP is respectable on sorted input (with
points ordered by angle the diameter pair sits at roughly opposite indices, a
regularity a dense net can exploit) and degrades fivefold when shuffled. Same points,
same diameter; only the labelling changed, and the labelling was never information —
give the transformer a positional encoding so it *can* read the order and it gets no
better (0.00124 against 0.00095).

**§3 — and the transformer never pads.** `SetAttention` is the block of Lecture 6
slide 6 — $z = x + \mathrm{Att}(\mathrm{LN}(x))$, $x' = z + \mathrm{MLP}(\mathrm{LN}(z))$
with the MLP **tokenwise** — followed by a mean over tokens. No maximum length appears
in it, so the same weights run at $n = 200$ after training on 10–22. Batching is
handled by `pack_by_length`, which groups clouds of **equal $n$**: thirteen rectangular
packs, and the tensors fed to attention hold 192,372 elements against $\sum 2n_i =
192{,}372$ — zero padding, exactly.

`FlatMLP` is the counterexample, and it is why the padding column exists. `nn.Linear`
fixes an input width at construction, so it pads every cloud to $n_{\max}$ inside its
own `forward` and raises outright on a longer one. The padding is a cost of that
architecture, not a property of the data.

**§4 — attention maps, and how far to trust them.** The diameter pair receives
**5.85×** the attention of an average point, and attention received correlates
$+0.38$ with distance from the centroid: the model learned to look at the extremes
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
  neighbourhood is computed rather than declared, over as many elements as the input
  happens to have.
- **Attention never needs padding.** Not in the model, where $n$ is only a dimension
  the pooling contracts away, and not in the batching once equal lengths are grouped.
  Only the fixed-width baseline pads — which is one more way of saying it has the
  wrong symmetry.
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
