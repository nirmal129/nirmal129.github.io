---
layout: page
title: Parallel Differentiable Sorting and Ranking with CUDA
description: A parallel CUDA implementation of the Pool-Adjacent-Violators (PAV) algorithm for differentiable sorting and ranking, achieving 64x–105x speedups for long sequences.
img: assets/img/PAV.png
importance: 1
category: work
---

Differentiable sorting and ranking are useful primitives for learning problems involving order statistics, ranking losses, monotonicity constraints, and structured prediction. [Blondel et al.](https://arxiv.org/abs/2002.08871) introduced an exact O(n log n) differentiable sorting and ranking algorithm that improves upon earlier all-pairs or approximate differentiable sorting methods. Their approach reduces differentiable sorting and ranking to isotonic optimization, with the Pool-Adjacent-Violators (PAV) algorithm serving as the main computational primitive.

PAV solves isotonic regression by scanning a sequence, maintaining contiguous monotone blocks, and merging adjacent blocks whenever the monotonicity constraint is violated. While elegant and efficient on a single sequence, the classical PAV algorithm is sequential in nature: each block merge can depend on previous merges, and local violations may propagate backward through the sequence. This dependency structure makes it difficult to fully utilize GPU parallelism, especially for very long sequences or large batched workloads.

In this project, I explored a parallel CUDA implementation of PAV for differentiable sorting and ranking, building on the `torchsort` framework. The main idea is to divide the input sequence into *p* chunks, run PAV independently on each chunk to obtain local PAV blocks, and then merge these block summaries through a binary-tree reduction to recover the final global PAV solution. This exposes parallelism across both the sequence and batch dimensions while preserving the exact structure required by the underlying differentiable sorting algorithm.

Empirically, this parallelization yielded substantial speedups in the regimes where sequential PAV becomes the primary bottleneck. The custom CUDA kernels achieved approximately **64x–105x speedups** for sequence lengths of 100K, and around **14x speedup** for batch size 4096, compared to standard open-source sequential GPU implementations. These results suggest that careful algorithmic restructuring—even for seemingly sequential primitives like PAV—can make differentiable sorting and ranking significantly more practical for long-sequence and large-batch settings.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Parallel-Sort-Result.png" title="Speedup results for parallel PAV" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Empirical speedup results comparing the parallel CUDA PAV implementation against standard sequential GPU baselines across varying sequence lengths and batch sizes.
</div>

The code is available at [Parallel-torchsort](https://github.com/nirmal129/Parallel-torchsort). The repository also includes a project report discussing possible future directions, including a ROC convex hull analogy and a saddle-point optimization perspective for further improving the merge step.
