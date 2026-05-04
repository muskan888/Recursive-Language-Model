# A small-scale reproduction of Recursive Language Models

## What the paper proposes

The Zhang, Kraska and Khattab paper "Recursive Language Models" (arxiv 2512.24601) is about a fairly specific failure of LLMs. Once a prompt gets long enough, accuracy drops off a cliff, and beyond the model's context window the prompt does not fit in the first place. The usual workarounds (RAG, summary compaction, retrieval agents) all lose information, or quietly miss anything that depends on every part of the input.

Their idea is straightforward. Instead of pasting the prompt into the model, they put it inside a Python REPL as a variable called `context`. The model itself only sees metadata about the prompt: its length, a short prefix, how it is chunked. It then writes Python code that the harness runs, and the printed output gets sent back on the next turn. The REPL also exposes a function `llm_query(prompt)` that calls a sub-LLM, so the root model can fire off recursive calls in a loop, say one per chunk of `context`. When it has the answer, it emits `FINAL(answer)` or `FINAL_VAR(name)` and the loop terminates.

The whole thing fits in roughly thirty lines of pseudocode (Algorithm 1 in the paper), but it sidesteps three different limitations that most agent scaffolds run into.

## What I built

A small Python package, no framework, eight files. The pieces are:

`rlm/client.py` wraps the OpenRouter chat API using the OpenAI SDK, since OpenRouter speaks the same protocol. `rlm/repl.py` is a persistent REPL backed by Python's `exec()` against a single globals dict. It captures stdout and truncates it before handing it back to the model, so a stray `print(context)` cannot blow up the conversation window. `rlm/prompts.py` holds the system prompt, lifted close to verbatim from Appendix C of the paper, with one extra clause forbidding speculative `FINAL(...)` in the same turn as a code block (a bug I hit during smoke testing, more on that below). `rlm/rlm.py` is the loop itself: parse the model's reply for `repl` code blocks, run them, look for `FINAL`, append the output to the history, repeat up to a cap. `rlm/baseline.py` is the comparison method, a single LLM call with the entire prompt jammed into the user message and head/tail truncation if it overflows. The two benchmark generators live under `benchmarks/`, and `eval.py` walks the (method, benchmark, length) grid and prints accuracy tables.

Both the root model and the sub-call model are Anthropic's `claude-haiku-4.5` through OpenRouter. Configurable via env vars if you want to swap something heavier in for the root.

## How I tested

The paper evaluates on four benchmarks (S-NIAH, BrowseComp+, OOLONG, OOLONG-Pairs), several of them in the 6 to 11 million token range. That budget was not on the table, so I wrote down two miniature versions of the spirit of these tasks:

**S-NIAH** is the single-needle-in-a-haystack task. A unique 8-digit number is hidden inside a sea of unrelated filler sentences and the model has to repeat it back. This is the easy benchmark, the kind LLMs are already good at.

**OOLONG-lite** is a counting task. The input is a list of records like `user_0034 | 2023-10-16 | What is the capital of France?`, hundreds of them. The query asks how many of these questions are about a particular semantic category (places, people, numeric values, organizations). The labels are not in the data, so the model has to infer them from the meaning of each question. This is the harder benchmark, because every record matters and substring matching is not enough.

The grid was three lengths (4,000 characters, 16,000, and 64,000), three methods (baseline, RLM without sub-calls, full RLM with sub-calls), and eight independent instances per cell. 144 task runs in total.

## Results

### S-NIAH

| method      | 4K chars | 16K chars | 64K chars |
|-------------|----------|-----------|-----------|
| baseline    | 100.0%   | 100.0%    | 100.0%    |
| rlm-no-sub  | 100.0%   | 100.0%    | 100.0%    |
| rlm         | 100.0%   | 100.0%    | 100.0%    |

All three methods score perfectly. This is consistent with the paper, and the paper itself flags in §4 that needle benchmarks make the model look better than it actually is on long context. The whole point of running it was to confirm that the harness was not silently broken on the easy case before reading too much into the harder one.

### OOLONG-lite

| method      | 4K chars | 16K chars | 64K chars |
|-------------|----------|-----------|-----------|
| baseline    |   0.0%   |   0.0%    |   0.0%    |
| rlm-no-sub  |  50.0%   |  50.0%    |  75.0%    |
| rlm         |  87.5%   |  62.5%    |  25.0%    |

A few things to read off this.

The baseline scores zero on every cell. Direct prompting just cannot do this task at any of the three sizes. That tracks with the paper's full OOLONG numbers, where GPT-5 got 44% and base Qwen3-Coder got 36%. With a smaller model (Haiku) and a slightly fuzzier "infer the category" twist, the floor drops all the way to zero.

Both RLM variants beat the baseline, but their tradeoff is messier than I expected. The full RLM with sub-calls wins big on the smallest input (87.5% vs 0%), but on the longest input the no-subcall variant actually does better (75% vs 25%). My best read on this is that at 64K characters there are around 750 records, and the full RLM ends up firing one sub-LLM call per chunk (or worse, per line). The per-call noise compounds. The no-subcall variant falls back to simpler code that counts deterministically once it has worked out the rule, and that turns out to be a more reliable strategy for this particular task at this particular size.

The paper actually warns about this exact failure mode in Appendix E.3, where Qwen-Coder makes thousands of sub-calls and degrades. Adding their "batch sub-calls aggressively" line to the system prompt would probably close most of the gap. So would using a smarter root model (Sonnet rather than Haiku) so the planning step picks the right strategy in the first place. There is also genuine variance in these numbers; with only eight instances per cell, a 25% number sits on a small enough sample that I would not bet a lot on the exact figure.

## Cost and time

The full eval ran in 24 minutes 28 seconds. OpenRouter spend was \$5.93, working out to about four cents per task run on average. The earlier smoke run was around \$0.30. Both fit comfortably inside the original \$20 credit, with \$14.07 still on the account when the run finished.

## What this reproduces and what it does not

The qualitative findings carry across cleanly. RLM beats the baseline on information-dense long-context tasks. The REPL alone (without sub-calls) is enough to get most of the gain on tasks where code can do the work, which matches Observation 2 in §4. LLM accuracy degrades with both input length and task complexity, and the rate of degradation is much milder for the RLM than for the baseline.

What this does not reproduce: the 1M+ token regime, the fine-tuned Qwen3-8B variant from Appendix A, and any kind of real cost curve (the total spend is too small to extrapolate). And of course the original benchmarks themselves, since I wrote miniature stand-ins rather than running BrowseComp+ or the real OOLONG split.

## What I would do next

If I had another evening on this, I would tighten the system prompt with the Qwen-style batching warning from Appendix C.1b and see whether that fixes the 64K regression. I would push instances per cell up to 16 or 32 so the variance shrinks. A real OOLONG-Pairs equivalent (the quadratic version) would be a useful third benchmark, since the paper found the most dramatic gaps there. And it would be worth trying Sonnet as the root model with Haiku as the sub-call model, mirroring the paper's GPT-5 plus GPT-5-mini setup.

None of these is hard. The implementation is small enough that each one is a half-hour change.

## A note on the implementation bug

Worth flagging because it surprised me. On the very first end-to-end run, the model emitted four `repl` code blocks and a `FINAL(42)` line all in the same turn. My loop accepted the FINAL because it was syntactically valid, even though the `42` was a wild guess made before any code had actually run. The actual answer (which the code did find on the second-to-last block) got thrown away.

The fix is one branch in `rlm/rlm.py`: if a turn contains both code blocks and a `FINAL`, run the code, ignore the FINAL, and prompt the model to look at the output and try again. I also added a sentence to the system prompt telling the model not to do this. After that change, the same instance solved correctly with the right answer.

This is the kind of thing that does not show up in the paper's pseudocode but matters quite a lot in practice. The paper's Appendix E examples mention several adjacent failure modes (Qwen-Coder repeatedly verifying its answer and never returning, or returning a verbal guess that contradicts a correctly-built variable). Frontier-model RLM trajectories in the wild have a lot of these footguns, and a real implementation needs guards for most of them.
