# `scaling_topology.html` — what it is, why it's built this way

This is a design/content record for `viz/scaling_topology.html`, written so a
session that hasn't seen it can get full context without reading the HTML/JS.
Live: https://chenxin-ma.github.io/viz/scaling_topology.html
Source of truth: `~/projects/exp-llm/viz/scaling_topology.html` (public repo
`github.com/schemmy/exp-llm`), mirrored to
`~/projects/chenxin-ma.github.io/viz/scaling_topology.html`.

## 1. Purpose

`request_journey.html` (the companion page, built first) walks through what
happens to one inference request on **a single GPU**. `scaling_topology.html`
answers the next question: a model can be too big to fit on one GPU — or
eight, or sixteen — so how does a single request's data actually move once
the model is sharded across a real multi-node deployment?

Audience: someone with basic ML/DS background but no prior distributed-systems
or GPU-infra vocabulary (see the bilingual glossary pattern in
`request_journey.html` for the target reading level). The page is explicitly
framed as a **worked example from first principles**, not a description of
any specific company's production system — the footer says so directly, and
only the small-scale mechanics (TP overhead, EP break-even point) are backed
by actual measurements from this repo's `experiments/wk09_tensor_parallel/`
and `experiments/wk10_moe/`.

## 2. The four nested levels (the actual content)

A ~1T-parameter MoE model, FP8, deployed as 16× H100-80GB (2 nodes × 8 GPUs),
~1TB weight footprint. Working top-down:

1. **Data center / routing (DP)** — multiple full model replicas exist purely
   to serve more concurrent requests; a load balancer picks one (often by
   cached-prefix match). Nothing about the model is split yet.
2. **Nodes / pipeline parallelism (PP)** — one replica = 2 physical servers.
   The model is cut *by layer* across nodes because inter-node bandwidth
   (InfiniBand, ~50GB/s) is ~18x slower than intra-node (NVLink, ~900GB/s).
   Only the activation vector crosses that slow link — not weights, not a
   per-layer all-reduce. This is the cheapest thing available to ship across
   a slow link.
3. **GPUs / tensor + expert parallelism (TP + EP)** — inside one node, NVLink
   is fast enough to afford a synchronizing step every layer. Dense layers
   use TP (every GPU holds a slice of every layer's weights, results combined
   via all-reduce every layer — measured: TP=4 drops per-GPU throughput to
   53%). MoE layers switch to EP (each GPU holds different experts, a router
   sends each token's activation only to the 1-2 GPUs holding its experts via
   all-to-all — measured: net loss (0.56x) at 60 experts, only pays off in
   the hundreds of experts, which is where real trillion-scale MoE models
   like DeepSeek-V3's 256-expert design actually live).
4. **Inside one GPU / memory hierarchy** — same story as
   `request_journey.html` stage 9, precisely labeled: weights stream HBM →
   shared L2 cache → each SM pulls its slice into on-chip memory → Tensor
   Cores do the multiply-add. This is the **one place on the whole page where
   weights literally move** — everywhere else they're a static per-box label.

## 3. The two design critiques that shaped the current version

The first version (shipped 2026-09-11) was a 4-tab switcher — click a tab,
see that level's diagram, tabs mutually exclusive. The user rejected this
directly and gave two specific critiques before any further code was written:

> "不太喜欢这个设计，我想完全有可能把4层放进一张图里，配合UI users可以把每一层展开、折叠。
> 而且，这个data flow没有讲KV cache、weight是怎么流动的。我们先想想设计、再开始implement。"

1. **Structure**: it should be *one diagram*, not four separate tab-switched
   diagrams — boxes nested inside boxes, expand/collapse per box.
2. **Content**: the original animation moved one generic "packet" through
   every scene, which never actually explained how weights vs. KV cache vs.
   activations differ in how they move.

These two critiques are the whole reason the current design looks the way it
does — see §4 and §5.

## 4. Structural design: single nested diagram

Replaced the tab switcher with one scrollable box-in-box diagram
(`.xnode` elements nesting via `.xnest` containers), built by a pure
`buildDiagram()` function that re-renders from a small state object:

```js
let state = { replica: "A", node: "n1", gpu: 0 };
```

Expand/collapse rules, chosen asymmetrically by cardinality and by how much
each sibling actually differs:

- **Replicas → nodes**: only one replica is ever expanded at a time
  (clicking a collapsed replica auto-collapses whichever was open) — the
  three replicas are identical, so there's no reason to see two at once.
- **Node → node**: **both** nodes are always shown, full width, side by
  side, connected by the InfiniBand connector — deliberately *not*
  accordioned, because node 1 and node 2 hold *different* layers and the
  point of this level is to see the pipeline hop between them.
- **GPU → GPU** and **GPU → SM**: single-path accordion (opening one GPU's
  detail auto-collapses whichever GPU was previously open; same for
  drilling into a GPU's SM chain) — the 8 GPUs in a node are functionally
  identical, so only one needs to be shown in full depth at a time.

Default state on page load expands one full path — Replica A → Node 1 →
GPU 0 → SM — so the entire 4-level structure is visible **without clicking
anything**. This was a deliberate choice over "start fully collapsed +
require autoplay to see structure," which is what the tab-switcher version
effectively forced.

## 5. Content design: three distinct data-flow behaviors

A legend at the top of the diagram makes this explicit before the reader
sees anything move:

| Element | Visual | Behavior |
|---|---|---|
| **Activation** | small purple dot, animates | the *only* thing that ever travels between boxes |
| **Weights** | static badge on the box that holds them | never animates — labeled per box (e.g. "weights: layers 1-30", "w: 1/8") |
| **KV cache** | static badge, green | never travels — grows every token, but stays in the GPU's own HBM the whole time |

The animation ("▶ Play one request's journey") is a sequence of atomic
scenes, and — this is the part that actually answers the original critique —
**the motion shape changes per mechanism** instead of being one generic
packet everywhere:

1. DP routing / PP hop — a single dot travels point-to-point (load balancer
   → replica → node 1 → ... → node 2 across InfiniBand).
2. TP all-reduce (dense layer) — **no dot**; all 8 GPU boxes in the row and
   the connectors between them pulse together, because all-reduce is a
   synchronous everyone-talks-to-everyone operation, not a point-to-point
   trip.
3. EP all-to-all (MoE layer) — only the source GPU and the 2 specific target
   ("expert") GPUs pulse; the other 5 GPUs stay dark, because routing goes
   to a *subset*, not a broadcast.
4. GPU-internal HBM→L2→SM→TensorCore — a dot walks this chain, because this
   is the one place weights (and the KV cache being read) genuinely move.

Scene captions (bilingual, numbered ①-⑦) narrate each step in the caption
bar above the diagram while it plays.

## 6. Playback controls — architecture, not just UI

First version used one `async function playJourney()` with a linear chain of
`await wait(ms)` calls — impossible to pause, since there's no way to
interrupt an in-flight `await`. When asked to add pause, the animation was
restructured around a **flat array of atomic step objects**:

```js
{ cap: CAP.allreduce, dur: 1050, action: () => pulseGroup([...]), cleanup: () => {...} }
```

run by a `setTimeout`-based scheduler (`scheduleJourneyStep`) that tracks a
`journeyIdx` pointer instead of unwinding a promise chain. This is what makes
play / pause / resume / step-forward / step-back all trivial to implement:
pausing just clears the pending `setTimeout`; stepping just calls `gotoStep`
with an arbitrary index; resuming just re-enters the scheduler at the current
index. **This pattern — steps-as-data run by an interruptible scheduler,
not linear async/await — is the reusable part if any future page needs a
pausable/steppable animation.**

One state-management bug worth noting for future pages: `journeyActive`
(true for the whole paused-or-playing duration) was originally used to block
manual diagram clicks, which meant pausing silently disabled all box
expand/collapse until the animation finished. Fixed by blocking only on
`journeyRunning` (true only while actively auto-advancing); a manual click
while paused now cleanly cancels the journey via `finishJourney()` first,
then applies the click. Lesson: when a "some action in progress" flag has an
active sub-state and a paused sub-state, gate interaction on the narrower one.

## 7. Visual/technical system (shared across the whole `viz/` series)

- Single self-contained HTML file, vanilla JS in one `(function(){...})()`
  IIFE, no build step, no framework. Same pattern as `request_journey.html`,
  `phase1.html`, `week11_roofline.html`.
- Dark theme CSS custom properties, reused verbatim across all four pages for
  series consistency: `--ground/--surface` (backgrounds), `--compute`
  (amber, compute-bound), `--bandwidth` (blue, bandwidth-bound), `--hit`
  (green, a win), `--reject` (red, a loss), `--measured` (purple — "the thing
  that actually moves," used here for the activation dot specifically).
  Fonts: Archivo (display), IBM Plex Sans (body), IBM Plex Mono (data/UI).
- Bilingual EN/中文: every string is an `{en, zh}` pair, a `tv(field)`
  accessor picks the active language, and the whole page does a full
  re-render (`renderAll()`) on language toggle. `localStorage` persists the
  choice under the key `journey-lang` (shared key with `request_journey.html`
  so the language choice carries across both pages).
- No incremental DOM patching anywhere — every state change (expand a box,
  toggle language, advance a journey step's box-highlighting) re-renders the
  relevant `innerHTML` from a pure `build*()` function. Simple to reason
  about at this scale; would not scale to a much larger page.

## 8. Why this file exists / how to reuse the pattern elsewhere

This doc was written specifically so a **separate session about visualizing
Spark's map-reduce execution model** could reuse the design thinking here
without re-deriving it. The structural analogy that makes this page's
approach transferable:

- **Static-vs-moving legend** (§5) generalizes to any system where most
  state sits still and only a specific, smaller thing crosses the network.
  For Spark: partition data sitting in an executor's memory/disk is the
  "weights" (static, per-box label); shuffle read/write across the network
  between map and reduce stages is the "activation" (the only thing that
  animates); an RDD's cached/persisted state is the "KV cache" (static,
  grows/changes locally, never itself travels).
- **Synchronization-shape-dependent animation** (§5, scene types 2-3):
  TP all-reduce (everyone pulses together, no point-to-point dot) vs. EP
  all-to-all (only source + selected targets pulse) is directly analogous to
  a Spark **shuffle** (all-to-all between all mapper and all reducer
  partitions — closer to the all-reduce visual) vs. a **broadcast join**
  (one side goes to every executor — closer to a fan-out visual) vs. a
  **narrow transformation** like `map`/`filter` (stays local, no
  cross-network animation needed at all, analogous to level 4's
  GPU-internal HBM loop).
- **Nested single-path-accordion + full-breadth-at-low-cardinality** (§4) is
  a good default for any topology diagram with a natural depth hierarchy:
  show siblings that differ from each other in full (Spark: stages in a
  job, since each stage does something different); accordion siblings that
  are functionally identical (Spark: the N parallel tasks within one stage,
  since only one needs to be shown in full detail — same reasoning as "only
  one of 8 identical GPUs needs its internals expanded").
- **Steps-as-data animation scheduler** (§6) is the reusable implementation
  pattern for any future page that needs play/pause/step controls.

Reasonable starting point for a Spark page: default-expand one path through
Job → Stage → Task → (shuffle read/write), same as this page's
Replica → Node → GPU → SM default path, with the same three-way legend
(static local data / the thing that crosses the network / cached state that
grows but doesn't travel).
