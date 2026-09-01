# AI_USE.md

## DATA-266 HW1 — AI Use Disclosure

### 1. Which parts used an assistant, and which parts were written by me?

I completed the assignment implementation and experiments independently. The neural-network code, preprocessing, model configurations, training runs, CUDA source code, benchmark runs, profiler work, figures, and reported numerical results were completed by me.

After completing the assignment, I used an AI assistant only to help organize the repository documentation requested by the course, including Markdown files such as `METRICS.md`, `README.md`, `REPORT.md`, and this disclosure file. The assistant was given my completed assignment material and reported outputs as the source for the documentation.

I did not use the assistant to generate the implementation that produced the submitted experimental results.

### 2. One specific incorrect assistant output

No assistant-generated implementation output was used for this assignment, so I do not have a genuine incorrect model output, tensor-shape error, deprecated API result, or incorrectly computed metric from an assistant-generated implementation to report.

I am intentionally not inventing a failure that did not occur. The course instructions state that if nothing produced by an assistant was wrong, the student should say so and describe what was verified instead.

### 3. How did I verify the work?

I verified the implementation by running the completed notebooks and checking the actual outputs:

- The neural-network experiments produced the three-seed accuracy values reported in `METRICS.md`.
- The fixed 70/15/15 split produced 531 training, 114 validation, and 114 test examples.
- The CUDA program compiled successfully with `nvcc`.
- CPU and GPU matrix-multiplication results were compared using maximum absolute difference.
- Nsight Systems was checked and was unavailable in the environment.
- Nsight Compute was available and successfully generated a profiler report.
- The benchmark results in `METRICS.md` come from the unprofiled benchmark runs, not the instrumented profiler run.

### 4. What was changed and why?

Because the implementation was completed independently, there was no assistant-generated implementation to repair. The only AI-assisted portion was documentation organization after the experiments were already complete.

The repository documentation preserves the actual configurations, measurements, profiler behavior, and conclusions from the completed work rather than replacing them with newly generated experimental values.

## Verification Summary

The final repository should contain the executed notebooks with outputs intact, the CUDA source, `RUN_LOG.txt`, `METRICS.md`, and `AI_USE.md`, as required by the assignment. fileciteturn1file3L143-L152
