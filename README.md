# Recursive Language Models

A small-scale reproduction of "Recursive Language Models" by Zhang, Kraska and Khattab (2026), arxiv:2512.24601. Implements the paper's inference scaffold (a Python REPL augmented with a recursive `llm_query` function) and evaluates it against a direct-LLM baseline and a no-sub-call ablation on two benchmarks of differing information complexity.

## Background

Modern LLMs degrade as inputs get longer, even within their nominal context window. The usual workarounds (RAG, summarisation, compaction) lose information by construction, and they fail on tasks that require dense access to every part of the input.

The paper proposes a different scaffold. Given a prompt $P$ and a base language model $\mathcal{M}$ with context size $K$, a Recursive Language Model (RLM) initializes a persistent REPL environment $\mathcal{E}$ in which $P$ is bound to a variable. The root model $\mathcal{M}$ sees only constant-size metadata about $P$ (its length, a short prefix, chunk lengths). It writes Python that runs in $\mathcal{E}$, observes truncated stdout, and may call a sub-LLM via `llm_query(prompt: str) -> str` over arbitrary slices of $P$. The loop terminates when $\mathcal{M}$ emits `FINAL(answer)` or `FINAL_VAR(name)`.

This sidesteps three structural failures of conventional agent scaffolds:

1. $P$ never enters $\mathcal{M}$'s context window directly, so $|P| \gg K$ is allowed.
2. The final answer is read from a variable rather than autoregressively generated, so output length is unbounded.
3. Sub-calls are programmatic, so they can be invoked inside loops over slices of $P$. This enables $\Omega(|P|)$ or $\Omega(|P|^2)$ semantic work over the input.

## Algorithm 1 (paper)

```
function RLM(prompt P, model M):
    state   <- InitREPL(context = P)
    state   <- AddFunction(state, sub_RLM_M)
    history <- [Metadata(state)]
    loop:
        code              <- M(history)
        (state, stdout)   <- REPL(state, code)
        history           <- history ++ code ++ Metadata(stdout)
        if state[Final] is set:
            return state[Final]
```

## Repository layout

| Path | Purpose |
|------|---------|
| `rlm/client.py`             | OpenRouter chat client (OpenAI SDK with overridden `base_url`) |
| `rlm/repl.py`               | Persistent Python REPL with stdout capture and truncation |
| `rlm/prompts.py`            | System prompts adapted from paper Appendix C |
| `rlm/rlm.py`                | Algorithm 1 loop (parses ` ```repl ` blocks, runs them, detects `FINAL`/`FINAL_VAR`) |
| `rlm/baseline.py`           | Direct LLM call with head/tail truncation if input overflows |
| `benchmarks/sniah.py`       | Single-needle-in-haystack generator and scorer |
| `benchmarks/oolong_lite.py` | Linear-complexity semantic-counting generator and scorer |
| `eval.py`                   | Grid runner over (method, benchmark, length) |
| `REPORT.md`                 | Long-form reproduction notes |
| `RLM.pdf`                   | Source paper |
| `eval.log`                  | Captured stdout from the full eval run |

## Setup

Requires Python 3.9+ and the following packages:

```bash
pip install openai python-dotenv
```

OpenRouter is used for unified access to LLM providers. Create `.env` in the repo root with:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

Optional model overrides (defaults shown):

```
RLM_ROOT_MODEL=anthropic/claude-haiku-4.5
RLM_SUB_MODEL=anthropic/claude-haiku-4.5
```

## Running

Smoke run (3 instances per cell, 2 lengths, approx 4 minutes, approx \$0.30):

```bash
python3 eval.py --quick
```

Full grid (8 instances per cell, 3 lengths, approx 25 minutes, approx \$6):

```bash
python3 eval.py
```

## Experimental setup

| Parameter                  | Value |
|----------------------------|-------|
| Root model                 | `anthropic/claude-haiku-4.5` (via OpenRouter) |
| Sub-call model             | `anthropic/claude-haiku-4.5` (same) |
| Sampling temperature       | 0.0 |
| Max iterations per RLM run | 8 |
| Sub-call char budget       | 200,000 |
| Stdout truncation per turn | 4,000 chars |
| Input lengths              | 4,000, 16,000, 64,000 chars |
| Instances per cell         | $n = 8$ |
| Methods                    | `baseline`, `rlm-no-sub`, `rlm` |
| Benchmarks                 | S-NIAH, OOLONG-lite |

Total task runs in the full eval: $3 \times 3 \times 8 \times 2 = 144$.

## Benchmarks

**S-NIAH** (single needle in a haystack). A unique eight-digit integer $n \in [10^7, 10^8)$ is embedded in a sentence of the form "The special magic number for the experiment is $n$." This sentence is inserted at a random position in a sequence of unrelated filler sentences padded to the target character budget. The model is asked for $n$. Score is exact-substring match: $\mathrm{score}(\hat{y}) = \mathbb{1}[n \in \hat{y}]$. Information complexity is $\mathcal{O}(1)$ in $|P|$.

**OOLONG-lite** (linear semantic counting). The input is a list of $m \approx |P| / 80$ records of the form `user_id | date | question`, where each question belongs to one of four ground-truth categories ($\{\text{location}, \text{numeric}, \text{human}, \text{entity}\}$). The category labels are not present in the data; the model must infer them from the semantics of each question. The query asks for the count of records matching one randomly selected target category. Score is exact-integer match. Information complexity is $\Theta(|P|)$ since every record matters and substring matching is insufficient.

## Results

### S-NIAH (constant-complexity retrieval)

| Method     | 4K chars     | 16K chars    | 64K chars    | Mean       |
|------------|--------------|--------------|--------------|------------|
| baseline   | 100.0% (8/8) | 100.0% (8/8) | 100.0% (8/8) | **100.0%** |
| rlm-no-sub | 100.0% (8/8) | 100.0% (8/8) | 100.0% (8/8) | **100.0%** |
| rlm        | 100.0% (8/8) | 100.0% (8/8) | 100.0% (8/8) | **100.0%** |

All methods saturate. Standard error per cell is zero since $p = 1$.

### OOLONG-lite (linear-complexity semantic counting)

| Method     | 4K chars   | 16K chars  | 64K chars  | Mean      |
|------------|------------|------------|------------|-----------|
| baseline   |  0.0% (0/8) |  0.0% (0/8) |  0.0% (0/8) |  **0.0%** |
| rlm-no-sub | 50.0% (4/8) | 50.0% (4/8) | 75.0% (6/8) | **58.3%** |
| rlm        | 87.5% (7/8) | 62.5% (5/8) | 25.0% (2/8) | **58.3%** |

Standard error of a sample proportion at $n = 8$:

$$\mathrm{SE}_p = \sqrt{\frac{p(1-p)}{n}}$$

Per-cell SE for OOLONG-lite (in percentage points):

| Method     | 4K SE | 16K SE | 64K SE |
|------------|-------|--------|--------|
| baseline   |  0.0  |  0.0   |  0.0   |
| rlm-no-sub | 17.7  | 17.7   | 15.3   |
| rlm        | 11.7  | 17.1   | 15.3   |

These error bars are large by design ($n = 8$). The qualitative ordering is robust; precise within-cell percentages are not.

### Aggregate gain over baseline

Mean OOLONG-lite accuracy averaged across the three input lengths:

- baseline:    $\bar{p} = 0.000$
- rlm-no-sub:  $\bar{p} = 0.583$
- rlm:         $\bar{p} = 0.583$

Both RLM variants improve over the baseline by an absolute **58.3 percentage points** on average. On the smallest input the full RLM scores 87.5% against a baseline of 0.0%, an absolute gain of **87.5 pp**.

### Significance check on the 64K regression

The largest within-method gap is `rlm-no-sub` (75.0%) vs `rlm` (25.0%) at 64K. Pooled-variance two-proportion z-test:

$$
\hat{p} = \frac{x_1 + x_2}{n_1 + n_2} = \frac{6 + 2}{16} = 0.5, \qquad
\mathrm{SE} = \sqrt{\hat{p}(1 - \hat{p})\!\left(\tfrac{1}{n_1} + \tfrac{1}{n_2}\right)} = \sqrt{0.25 \cdot 0.25} = 0.25
$$

$$
z = \frac{p_1 - p_2}{\mathrm{SE}} = \frac{0.75 - 0.25}{0.25} = 2.0, \qquad p_{\text{two-tailed}} \approx 0.046
$$

Significant at $\alpha = 0.05$ but on a single-cell sample of 16. Worth investigating, not worth treating as definitive.

### Cost and time

| Metric                       | Value             |
|------------------------------|-------------------|
| Total task runs              | 144               |
| Wall clock                   | 1,468.5 s (24 min 28 s) |
| OpenRouter spend             | \$5.93            |
| Mean cost per task run       | \$0.041           |
| Mean wall time per task run  | 10.2 s            |

Approximate per-method costs (inferred from the spend distribution across cells):

| Method     | Mean cost per run |
|------------|-------------------|
| baseline   | $\sim$\$0.005     |
| rlm-no-sub | $\sim$\$0.025     |
| rlm        | $\sim$\$0.080     |

## Discussion

**1. Needle benchmarks understate the long-context problem.** All three methods, including the cheapest baseline, hit 100% on S-NIAH at every length tested. The benchmark is too easy at this scale to discriminate between approaches. This matches the paper's discussion in section 4: needle tasks have $\mathcal{O}(1)$ information complexity and modern LLMs handle them comfortably, even though they otherwise struggle with long contexts.

**2. Direct prompting collapses on linear-complexity tasks.** The baseline scores 0/8 on every OOLONG-lite cell, including the smallest (4K chars, well within the model's nominal window). The failure mode here is not context overflow but the model's inability to reliably perform per-record semantic classification followed by aggregation in a single forward pass.

**3. The REPL alone closes most of the gap.** The `rlm-no-sub` variant has access only to a Python REPL, no `llm_query`. It scores 50% to 75% on OOLONG-lite versus 0% for the baseline. On tasks where a per-record predicate can be approximated by code (regex, keyword counts), the REPL alone provides most of the benefit. This matches paper Observation 2.

**4. Sub-calls help on small inputs but degrade on larger ones in this setup.** Full `rlm` beats `rlm-no-sub` at 4K (87.5% vs 50.0%, +37.5 pp), ties at 16K (62.5% vs 50.0%, +12.5 pp), and loses at 64K (25.0% vs 75.0%, $-$50.0 pp). The likely cause is per-call classification noise compounding as the number of sub-calls grows. At 64K chars there are roughly 750 records; if each sub-call call classifies a record at $\sim$95% accuracy, the probability of a fully-correct aggregate count is $0.95^{750} \approx 10^{-17}$. The paper observes the same failure mode for Qwen3-Coder in Appendix E.3 ("hundreds to thousands of recursive sub-calls for a single simple task") and recommends adding a batching warning to the system prompt.

## Implementation note

During smoke testing the model was seen emitting four ` ```repl ` code blocks and a `FINAL(42)` line in the same turn, where the `42` was a guess made before any code had run. Accepting that `FINAL` discarded the correct answer found by the code. The fix in `rlm/rlm.py` is a single branch: when a turn contains both code and a `FINAL`, run the code, ignore the `FINAL`, and prompt the model to look at the output before finalizing. The system prompt was updated correspondingly. This kind of guard does not appear in the paper's pseudocode but is necessary in practice; Appendix E describes several adjacent failure modes (the model returning a verbal guess that contradicts a correctly-built variable, or repeatedly verifying its answer and never returning).

## Limitations

This reproduction does not cover:

- **The 1M+ token regime.** The paper evaluates up to $2^{18} \approx 262{,}144$ tokens. The largest input here is roughly 16,000 tokens.
- **The full benchmark suite.** The paper evaluates on S-NIAH, OOLONG, OOLONG-Pairs, BrowseComp+ and CodeQA. This reproduction implements miniature versions of S-NIAH and OOLONG only.
- **The fine-tuned RLM-Qwen3-8B.** Out of scope for an inference-only reproduction.
- **A real cost analysis.** \$5.93 of spend is too small to extrapolate cost curves with any confidence.

## Future work

1. Add the Qwen-style "batch sub-calls aggressively" line from paper Appendix C.1b to the system prompt and re-test the 64K regression.
2. Increase $n$ to 16 or 32 per cell to bring SE below 10 pp.
3. Implement OOLONG-Pairs (quadratic complexity) since the paper reports the largest gaps there.
4. Try a stronger root model (e.g. Sonnet 4.5) with cheap sub-calls (Haiku 4.5), mirroring the paper's GPT-5 + GPT-5-mini setup.
5. Push input lengths to $2^{15}$ and $2^{17}$ tokens to test the long-context claim directly.

## Citation

```bibtex
@article{zhang2026rlm,
  title   = {Recursive Language Models},
  author  = {Zhang, Alex L. and Kraska, Tim and Khattab, Omar},
  year    = {2026},
  journal = {arxiv preprint arXiv:2512.24601}
}
```

See [REPORT.md](REPORT.md) for additional commentary on the implementation, including the speculative-`FINAL` parsing bug discovered during smoke testing.
