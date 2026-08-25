# Tutorial 5 — Convolutional Networks for Geometric Data

**MM845 — Tópicos de Geometria III: AI for Geometry**
Paired with **Lecture 5: Convolutional Neural Networks**

---

## The idea

Tutorial 4 made the hypothesis space bigger and lost every guarantee. Lecture 5 does
the opposite: it makes the space **smaller**, by requiring the linear maps to commute
with a group action. A convolution is precisely a translation-equivariant linear map,
and everything else — weight sharing, locality, size-independent parameter counts —
follows from that one constraint.

This is the course's first instance of the principle Lecture 10 states in general:
*build the symmetry into the architecture rather than hoping the model learns it*.
We test it on geometric raster data — Tutorial 3's plane curves, rendered as images.

## Files

| File | What it is |
|---|---|
| [`cnns_for_geometric_data.ipynb`](cnns_for_geometric_data.ipynb) | The tutorial. 5 sections, 4 figures, 2 exercises. ~1 hour. |
| `README.md` | This file. |

## Running it

```bash
$ conda activate aigeo
$ jupyter lab cnns_for_geometric_data.ipynb
```

Needs the `aigeo` environment from [Tutorial 1](../tutorial_01/README.md). Runs top
to bottom in about 50 seconds on a laptop CPU (four small networks, float32).

## Contents

| § | Topic | Lecture 5 connection |
|---|---|---|
| 1 | Convolution as *the* translation-equivariant linear map — verified numerically | symmetry-preserving linear maps |
| 2 | Rasterised curves with exact area and convexity labels | structured geometric data |
| 3 | CNN vs MLP, on centred and on translated test sets | architecture and invariance |
| 4 | What the first-layer filters actually learn | filters as feature detectors |
| 5 | Summary | |

## Results worth watching for

**§1 — equivariance, checked rather than assumed.** $\lVert K(T_vx) - T_v(Kx)\rVert_\infty$
is at machine precision for a circular-padded convolution and $O(1)$ for a dense
layer of the same input and output shape — which uses about $10^4$ times as many
parameters to be worse. A global average pool then converts equivariance into exact
invariance.

**§3 — the result that matters, and it is not the obvious one.**

| | params | centred | translated |
|---|---|---|---|
| CNN (convexity) | 1,849 | 68.6% | **68.4%** |
| MLP (convexity) | 69,825 | **75.5%** | 52.6% |
| CNN (area, MSE) | 1,849 | 0.0183 | **0.0186** |
| MLP (area, MSE) | 69,825 | **0.0014** | 0.1484 |

Majority baseline 54.4%; predict-the-mean MSE 0.0421.

The MLP is the *better* model on centred data — this is not a story about
convolutions being uniformly superior. Under translation it collapses, on area to
worse than predicting the mean.

The point the tutorial draws out is sharper than "constraints save parameters".
Area here is essentially the filled-pixel count (correlation $0.974$, measured in
§2) — a function that is translation-invariant, **linear**, and therefore sitting
comfortably inside the MLP's hypothesis space. The invariant solution was available
and nearly optimal, and the MLP did not find it: empirical risk minimisation
optimises the average over the training distribution, and nothing in that objective
rewards a solution for being invariant. If you want the symmetry, you have to impose
it.

**§4 — an honest look at the filters, which contradicts the textbook picture.**
Comparing $\lvert\sum k_{ij}\rvert$ against the typical entry size classifies each
kernel as a difference operator (edge) or an occupancy detector (bulk). The result:
**2 of 8 are edge detectors, 3 are occupancy detectors, 3 are mixed.** For indicator
images that is the right answer — a positive-sum kernel integrates to the area,
which is one of our two labels. The section's real lesson is methodological: the
claim "the first layer learns edge detectors" is checkable in three lines, and here
it is largely false.

## Exercises

| # | § | Topic |
|---|---|---|
| 1 | 3 | How much is the symmetry worth? Augmentation, sample efficiency, padding mode |
| 2 | 4 | Read the filters as operators: moments, $\partial_x$ vs $\partial_y$ vs Laplacian; reflection equivariance |

## What to take away

- A convolution is **the** translation-equivariant linear map, not merely a
  convenient one.
- **Equivariance is not invariance** — features move with the image; invariance
  arrives when you reduce over position.
- **Constraints beat capacity when the constraint is true**, and the failure mode of
  an unconstrained model is not "slightly worse" but "arbitrarily bad off
  distribution".
- **An available invariant solution is not a found one.** Architecture does not just
  make invariance cheaper; often it is what makes it happen at all.
- **Inspect what your model learned.** Assumptions about learned features are
  cheap to test and frequently wrong.

## Next

**Lecture 6** replaces the fixed neighbourhood of a convolution with a learned,
data-dependent one: attention. **Tutorial 6** applies it to structured geometric data
where there is no grid at all.

## Further reading

- Cohen & Welling, "Group equivariant convolutional networks", *ICML* 2016 — the generalisation of §1.
- Bronstein, Bruna, Cohen & Veličković, *Geometric Deep Learning* (2021), ch. 3–5.
- Zhang, "Making convolutional networks shift-invariant again", *ICML* 2019 — why real CNNs are less equivariant than the theory suggests.
