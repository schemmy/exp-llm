# One A100, Twelve Questions: A Roofline Tour of vLLM

*Qwen2.5-7B-Instruct, a single A100 80GB, and vLLM. Every number below is
measured, not estimated, and every experiment is reproducible from
[the repo](https://github.com/schemmy/mle-transition) with one command.*

---

`nvidia-smi` said the GPU was at 100% utilization. By that metric, the card
was maxed out — nothing left on the table.

But utilization only answers one question: *is at least one kernel running
right now?* It says nothing about how much of the card's actual capability
that kernel is using. Ask a sharper question — *what fraction of the GPU's
peak arithmetic throughput did this workload actually use* — and the answer
for that same moment was 0.47%. Ask a third question — *what fraction of its
peak memory bandwidth* — and the answer was 73%.

Three numbers, one GPU, one instant in time, three completely different
pictures. None of them is lying. `nvidia-smi` is telling the truth about
occupancy. The 0.47% is telling the truth about compute. The 73% is telling
the truth about bandwidth. The card was almost perfectly saturated — just not
on the axis "100% utilization" suggests.

That gap turned out to be the thread that ties together everything else in
this piece. Once you can name the two axes an inference workload lives on —
how much compute it uses, and how much memory traffic it uses — a dozen
techniques that look unrelated (batching, caching, speculative decoding,
splitting a model across GPUs, quantization) turn out to be the same handful
of moves, applied on one axis or the other. Some of them help. Some of them
help until they suddenly don't. A couple of them looked like they should
obviously help and then, measured, did almost nothing at all.

This is the story of chasing that gap down, one measurement at a time.

---

## The two phases

Before any of the rest makes sense, it helps to notice that serving one
request isn't one kind of work — it's two, with opposite personalities.

The first phase processes the entire prompt at once. A hundred and fifty
tokens go in, and the GPU does one large matrix multiplication covering all
of them simultaneously. This is *prefill*, and it is compute-heavy in the
most literal sense: the arithmetic is the bottleneck, and the tensor cores
are busy.

The second phase generates one token at a time. Each step reads the entire
set of model weights from memory, does a comparatively tiny amount of
arithmetic with them, and produces exactly one token. This is *decode*, and
it is almost the opposite of prefill: the weights have to be pulled from HBM
into the compute units fresh on every single step, and for a 7-billion
parameter model in fp16, that's roughly 14 gigabytes moving every few
milliseconds. On an A100, moving 14 GB at ~2 TB/s takes about 7 ms — and the
measured time per token was about 12 ms. Well over half of every step is
spent waiting on memory, not computing.

This split — compute-bound prefill, bandwidth-bound decode — turned out to be
the single fact that explains almost every surprising result that follows.
It's also where a slightly odd early observation stopped being odd: run the
same decode workload on a 7-billion-parameter model and a much smaller
500-million-parameter model, and the *bigger* model comes out faster. That
shouldn't happen if compute were the bottleneck — a bigger model does more
arithmetic. It happens because neither model is remotely close to saturating
the GPU's compute, and the smaller model doesn't even give the tensor cores
enough work per step to use them efficiently. Size wasn't the variable that
mattered. Which side of the compute/bandwidth line the workload sat on was.

With that vocabulary in hand — prefill vs. decode, compute-bound vs.
bandwidth-bound — the rest of this piece is a sequence of questions, each one
growing out of the answer to the last.

---

## If decode spends most of its time waiting, why wait for anything else?

If a single decode step is dominated by the time it takes to move weights
through memory, and that same weight-movement has to happen whether the step
serves one request or several, an obvious question follows: what if many
requests shared that same step?

The naive version of this is easy to build: let requests queue up, and once
enough have arrived, submit them together as one batch. It's a reasonable
first guess, and it's almost entirely wrong. Measured against the same
workload — dozens of requests arriving over a few seconds — batching-by-waiting
produced almost the same total throughput as processing them one engine cycle
at a time as they arrived (717 vs. 705 tokens/sec, a wash). What it did *not*
produce was the same latency: the median request waited more than twice as
long before its first output arrived, because it was sitting in a queue for
requests that hadn't shown up yet.

The GPU was never short on capacity. The problem was that requests were being
made to wait for each other for no reason related to the hardware at all. A
scheduler that lets requests join and leave a running batch continuously —
adding a new request the moment it arrives, retiring a finished one the
moment it's done — fixes exactly that, and nothing about the underlying
compute changes. This is the mechanism vLLM calls continuous batching, and
the lesson underneath it is worth holding onto, because it recurs: **the
first fix usually isn't about giving the GPU more work. It's about removing
work that shouldn't have been waiting in the first place.**

---

## If you want more requests in flight, where does the memory go?

Serving more concurrent requests continuously means holding more requests'
intermediate state in memory at once — specifically the KV cache, the running
memory of everything each sequence has generated so far. And that immediately
raises an allocation problem: nobody knows in advance how long a given
sequence will run. Reserve too little space and a long response gets cut off.
Reserve too much — reserve for the worst case, every time — and the GPU wastes
an enormous amount of memory holding space that will probably never be used.
Measured directly, that waste sat between 60% and 80% of allocated cache
memory, doing nothing.

The fix borrows an idea that's older than any of this: instead of reserving
one contiguous block sized for the worst case, allocate memory in small fixed
blocks, hand them out on demand as a sequence grows, and keep a table
mapping each sequence's logical position to wherever its blocks physically
live. This is exactly how an operating system pages virtual memory, applied
to a GPU's KV cache instead of a process's address space (which is why the
technique is called PagedAttention). Fragmentation dropped from 60–80% down
to under 4%.

The direct payoff of getting that memory back is more room for concurrent
requests — and because decode is bandwidth-bound, not compute-bound, adding
more concurrent sequences turned out to be close to free. Sweeping batch size
from 1 to 32 produced throughput scaling that was almost perfectly linear —
80 tokens/sec at batch 1, 2,409 at batch 32 — while wall-clock latency for the
whole batch barely moved (1.25s to 1.33s). Once the memory bottleneck was
gone, the GPU happily did 32x the useful work in essentially the same amount
of time it had been spending on one request, because the weights it already
had to move for one request could now serve thirty-two.

---

## What if many of those requests are computing the same thing?

A batch of concurrent requests in a real deployment rarely looks like thirty-two
unrelated prompts. It usually looks like the same long system prompt, or the
same retrieved document, prepended to thirty-two different questions. Every
one of those requests is redoing the exact same prefill work on the exact
same few hundred tokens of shared prefix before it ever gets to the part
that's actually different.

If the KV cache is already organized into content-addressable blocks — which
it now is, as a side effect of paging it — there's a natural next move:
identify when a new request's prefix matches blocks that are already computed
and cached from an earlier request, and skip recomputing them entirely.
Measured on thirty-two requests sharing a roughly 500-token prefix, turning
this on took time-to-first-token from 829 ms down to 156 ms — a 5.3x
reduction — and total throughput actually *rose* as a side effect, because
all the compute that used to go into re-running an identical prefill got
redirected into generating more tokens instead.

The catch is that the match has to be exact, token for token. Two requests
whose shared context diverges by even one token get no benefit past that
point, which is why serving this well at scale means routing requests with
matching prefixes to the same machine on purpose rather than hoping they land
there.

---

## If decode leaves compute mostly idle, can that idle compute be spent on something?

Return to the fact that started all of this: during decode, the GPU is
mostly waiting on memory, and its compute units sit largely unused. That
raises a genuinely aggressive idea. What if, instead of generating one token
per step, a small and cheap model guessed several tokens ahead, and the real
model checked all of those guesses in a single forward pass instead of
generating them one at a time?

The reason this can work without changing the output at all is a property of
how attention is masked: because each position can only attend to positions
before it, running several candidate tokens through the model in one pass
produces, at every position, the exact same prediction the model would have
produced generating token-by-token from scratch. Verifying five guessed
tokens costs almost the same as generating one real token, because the
bottleneck was never the arithmetic — it was reading the weights, and reading
the weights happens once per forward pass regardless of how many positions
that pass covers. Accept the guesses that match, discard everything after the
first one that doesn't, and the output distribution is mathematically
identical to plain decoding. This is speculative decoding, and the cheapest
version of the "small model" doesn't even need a model — it looks for
repeated sequences already sitting in the prompt or in what's been generated
so far and guesses those.

The first measurement was disappointing: an 8% speedup, nowhere near the
2–3x these techniques are supposed to deliver. That gap became the next
question rather than the end of the story. The task in that first test was
free-form explanation — the model was composing new sentences, and a lookup
against the existing text rarely found anything worth guessing. Switching to
a task where the correct continuation genuinely does repeat the input — copy
this passage verbatim, with a small edit — told a completely different story:
2.78x at low concurrency, 2.38x even at higher concurrency, on the exact same
engine with the exact same settings. The only thing that changed was whether
the guesses were usually right.

That comparison also overturned an assumption that seemed obvious going in:
that this technique should simply stop working once the GPU gets busy with
many concurrent requests, because now the "idle compute" it relies on isn't
idle anymore. The real pattern turned out to be more precise. Under load, a
wrong guess isn't free anymore — verifying it costs real arithmetic that
would otherwise have gone toward useful work — so a low hit rate does flip
into a net loss under concurrency. But a high hit rate keeps winning, loss and
all, because the guesses it wastes are outweighed by the ones it gets right.
Concurrency doesn't decide whether this technique wins. It decides how much
whatever the hit rate already was gets amplified, in either direction.

---

## What happens when the model itself doesn't fit on one GPU?

Every technique so far assumed the model fits in one GPU's memory. That
assumption breaks the moment the model is large enough, and it raises a
question none of the single-GPU tricks can answer: how do you split the model
itself across multiple GPUs, and what does that cost?

**Splitting every layer.** The most direct approach slices the weight
matrices inside every layer — one GPU holds a fraction of every attention
head, every feed-forward weight — and reassembles the full result with a
communication step after each layer. Because decode is bandwidth-bound, this
has an immediate and intuitive payoff: each GPU now only has to read its
*share* of the weights per step, so two GPUs splitting the model in half saw
per-step latency drop from 10.3 ms to 6.6 ms. But the speedup measured lower
than the naive 2x — 1.40–1.55x at two GPUs, 2.13–2.20x at four — and the gap
is the communication cost of stitching the partial results back together at
the end of every layer: 56 collective operations per forward pass on a
28-layer model. That overhead turned out to be a remarkably stable tax,
sitting at close to the same *percentage* of total time regardless of how
large the batch was.

The number that mattered most, though, wasn't the speedup — it was what
happened to throughput *per GPU*. It fell. Two GPUs delivered less than twice
the throughput of one; four delivered barely twice the throughput of one,
spread across four times the hardware. Splitting a model this way buys lower
latency and — because freed-up weight memory becomes room for more
concurrent sequences — dramatically more serving capacity (max cache capacity
scaled from about a million tokens on one GPU to over five million on four).
It does not buy better cost efficiency: four independent single-GPU replicas
would out-throughput one four-way split of the same size, for the same
hardware bill. Splitting the model is for when the model doesn't fit, or when
latency is the thing being optimized for — not for squeezing more tokens per
dollar out of a model that already fits.

**Splitting by expert instead of by layer.** A different kind of model raises
the same "doesn't fit on one GPU" question from an entirely different angle.
A mixture-of-experts model doesn't run every parameter for every token — a
router sends each token to only a handful of specialized sub-networks out of
many, and the ones not selected sit idle for that step. The strange
implication is that a model can have an enormous total parameter count while
only touching a small fraction of it per token. Since a GPU can't know in
advance which experts the next token will need, every expert still has to sit
resident in memory — but the amount of memory *traffic* per step should track
the small active fraction, not the large total. On paper, that looks like
getting the memory footprint of a huge model and the decode speed of a small
one for free.

Measured against two matched reference models — a dense model sized to the
active parameter count, and a dense model sized to the total parameter count
— the mixture-of-experts model landed 67% of the way toward the small
model's speed at low concurrency. That's a real win, just short of the ideal.
At higher concurrency, the same comparison put it at only 6% of the way
there — almost no benefit left at all. The reason comes back to the shape of
the underlying matrix multiplication. A dense model serving many concurrent
requests turns them into one large, efficient matrix multiply. A
mixture-of-experts model scatters those same requests across dozens of
experts, and each expert ends up multiplying a matrix with only a handful of
rows — dozens of small, inefficient multiplies instead of one large one.
Batching, the technique that made every dense-model result in this piece
better, is precisely the one lever this architecture can't pull, because
concurrency that should sharpen the arithmetic instead gets diluted across
however many experts are active.

That also predicted, wrongly, what would happen when this model was split
across two GPUs by dividing whole experts between them rather than slicing
inside each expert. The expectation was that this should help at higher
concurrency, since each GPU would now be responsible for fewer experts and
therefore a larger effective batch per expert. Measured, both ways of
splitting this particular model across two GPUs made it slower, not faster —
dividing whole experts made it slower than slicing inside each expert did.
The arithmetic each GPU was doing per expert was already too small to survive
being split further, and dividing experts between GPUs adds a
round-trip of network communication to route each token to the GPU holding
its assigned expert, on top of that. Splitting by expert only starts paying
off once a model has enough experts that slicing inside each one would have
shredded it beyond usefulness anyway — hundreds of experts, not dozens.

---

## Is there a ceiling to all of this — one you could calculate in advance?

Enough of the results above amount to some version of "throughput scales with
batch size until it doesn't" that the natural next move is to stop
re-discovering that empirically every time and instead work out, from the
hardware's own numbers, exactly where the "until it doesn't" should sit.

The ratio of a GPU's peak arithmetic throughput to its peak memory bandwidth
defines a threshold, conventionally called the ridge point: below it, a
workload can't possibly be using the GPU's full compute capability, because
it isn't feeding the tensor cores enough work per byte moved; above it, the
workload can't possibly be bandwidth-limited anymore, because there's more
arithmetic to do per byte than the hardware could ever move fast enough to
starve it. For the decode step examined throughout this piece, the batch
size *is* the relevant "work per byte" figure, so this ratio predicts a
specific batch size where the transition should happen. For an A100, that
arithmetic puts the ridge at roughly 153.

The measured transition landed almost exactly where predicted — throughput
was still 93% of ideal-linear at batch 32, down to 65% by batch 128, and down
to 43% by batch 256, bracketing 153 cleanly. A cleaner confirmation of having
crossed it: pushing batch size from 256 to 512 produced no further gain at
all — throughput was flat, even fractionally lower — because there was
nothing left to buy. Compute utilization on this measured ceiling topped out
around 51–52%, not the textbook 100%: real attention, normalization, and
sampling overhead, plus real tensor cores never hitting their theoretical
peak, eat the other half. That 51–52% figure turned out to be a genuinely
useful number to carry forward — not a hypothetical ceiling, but the one this
particular stack actually achieves.

One assumption baked into every measurement up to this point got tested
directly here, too: the engine captures the exact sequence of GPU operations
a decode step requires and replays that captured sequence instead of issuing
each operation freshly every time, specifically to avoid paying dispatch
overhead per operation on every single step. A back-of-envelope estimate of
what that saves — a few hundred operations, several microseconds of raw
launch overhead each — suggested something like a 20% speedup. Measured
directly, by turning the optimization off and re-running the same sweep, the
real number was 2.69x at low batch size, decaying as batch size grew and that
fixed overhead got amortized over more and more actual work per step. The
back-of-envelope estimate had only counted the literal cost of issuing a GPU
instruction; it missed a second, larger cost sitting underneath it — the
Python-side bookkeeping the framework does on every operation before that
instruction ever reaches the GPU. Both costs disappear together when the
sequence is captured once and replayed, which is most of why the real number
came in so much higher than the estimate.

---

## The last lever: make the numbers themselves smaller

Every technique up to this point changed how weights were used, moved, or
split. One obvious lever hadn't been tried yet: shrink the weights
themselves. Halve the bytes each parameter takes up, and — on a
bandwidth-bound step — halving the bytes moved should roughly halve the time
spent moving them.

The first attempt at this came back with almost nothing. Storing weights in
half the precision produced throughput within a percent or two of full
precision, at every batch size tested — not a modest win, essentially zero
measurable difference at all, including at the smallest batch size, exactly
where a bandwidth-bound step should have benefited the most. That null
result was more useful than a positive one would have been, because it broke
an assumption implicit in every earlier win in this piece: that moving fewer
bytes automatically becomes moving faster. It doesn't, automatically. Shrinking
a number on disk is only half the story — before the actual multiplication
can happen, that smaller number has to be expanded back to full precision,
and if that expansion step isn't fused efficiently into the same pass that
does the multiplication, the round trip can cost back everything the smaller
storage format saved.

A different quantization format, using a purpose-built kernel specifically
optimized to fuse that expansion step directly into the multiplication
instead of doing it as a separate pass, told the opposite story — and did so
at a far more aggressive compression ratio, a quarter of the original size
rather than half. At low concurrency it delivered 2.2x, not the modest 8%
result of the first attempt, and not even the smaller compression ratio's own
best showing on a different, more efficient kernel path (a format compressing
to half size, run through its own well-optimized kernel, topped out at 1.56x
over the same range). The quarter-size format, on the *properly fused* path,
beat the half-size format outright.

Working out exactly why clarified what had actually gone wrong the first
time. Each format has its own theoretical ceiling — the ratio of full
precision size to compressed size, 2x for the format that halves storage,
roughly 2.7x for the format that quarters it (a bit under 4x, because
several layers are conventionally left at full precision rather than
compressed at all). Measured against those two different ceilings, both
formats realized almost exactly the same fraction of their own — somewhere
around 78–81%. Neither kernel was fundamentally better at converting
theoretical savings into real speed than the other; they were, if anything,
about equally good at it. The quarter-size format won on the absolute number
purely because its ceiling started higher. What sank the very first attempt,
in other words, was never the format — it was that the execution path
underneath it never delivered on the format's own promise in the first
place. Compressing the numbers only pays off if something is built to
actually take advantage of it.

One result here still doesn't have a fully satisfying explanation. The
expectation going in was that the more aggressively compressed format should
pay a *larger* penalty once the workload moved past the ridge point into the
compute-bound region — more compression should mean more expansion work per
step, and that expansion is exactly the cost that becomes a pure liability
once bandwidth stops being the bottleneck. Measured, it was the opposite: the
quarter-size format lost *less* in the compute-bound region than the
half-size one did. That's recorded honestly as an open question rather than
a resolved one — a plausible guess is that a sufficiently well-fused kernel
keeps the expansion cost small enough, relative to everything else in the
step, that it barely shows up either way — but nothing in these measurements
actually confirms that mechanism.

---

## The whole story on one axis

Line up everything above, and each technique answers exactly one of two
questions: does it let a fixed amount of memory traffic cover more useful
work, or does it reduce how much memory traffic there is to begin with.

| Technique | What it moves | Measured result |
|---|---|---|
| Continuous batching | Removes artificial queueing delay | P50 latency 6.36s → 2.41s; throughput essentially unchanged |
| Paged memory allocation | Frees wasted memory into more concurrent work | Batch 1→32 throughput scales near-linearly (80 → 2,409 tok/s) |
| Prefix caching | Skips recomputing shared context | Time-to-first-token 829ms → 156ms |
| Speculative decoding | Spends otherwise-idle compute | 2.78x at high hit rate, net loss under load at low hit rate |
| Splitting layers across GPUs | Shrinks bytes moved per GPU, adds communication | 1.4–2.2x speedup, but *lower* throughput per GPU |
| Mixture-of-experts | Shrinks bytes moved per token, at the cost of GEMM shape | 67% of dense-small speed at low concurrency, 6% at high |
| Quantization | Shrinks the bytes themselves | 1.5–2.2x where bandwidth-bound, break-even or a loss where compute-bound, entirely dependent on whether the kernel underneath actually exploits it |

None of these are free. Every one of them buys a win on one axis by spending
something on the other, or simply doesn't apply once the workload has moved
to the other side of the ridge. The GPU that opened this piece, showing 100%
utilization and 0.47% compute usage, wasn't an anomaly to be fixed. It was
the starting clue that these two axes exist at all — and once they're visible,
almost nothing else in this piece is really surprising in hindsight, even the
parts that were surprising the first time they were measured.

What's still open is what any of this looks like from the other side — not
serving a model that already exists, but training one, where the
communication patterns and the bottlenecks are a genuinely different shape.
That's the next thing to go measure.

---

*Everything above is reproducible. The code, the raw results, and an
interactive walkthrough of each measurement live at
[github.com/schemmy/mle-transition](https://github.com/schemmy/mle-transition).*
