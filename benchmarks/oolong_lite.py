"""OOLONG-lite: linear-complexity aggregation over per-line records.

Each line is a record like:
    user_42 | 2023-05-12 | The capital of France is Paris.

The query asks the model to count records that satisfy a semantic predicate
that is HARDER than substring search (e.g., "questions about a city/place").
This mimics OOLONG's flavor: needs to look at every line, can't be solved by
keyword grep alone, scales linearly with input length.

Score = exact-int match (within tolerance of 0; this is harsh on purpose).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

# (text, label) pairs. Label is GROUND TRUTH, not shown to the model.
# Labels: 'location', 'numeric', 'human', 'entity'
SEED_LINES: list[tuple[str, str]] = [
    ("What is the capital of France?", "location"),
    ("Who painted the Mona Lisa?", "human"),
    ("How many planets are in our solar system?", "numeric"),
    ("What company makes the iPhone?", "entity"),
    ("Where was the Eiffel Tower built?", "location"),
    ("Who wrote the play Hamlet?", "human"),
    ("What is the speed of light in m/s?", "numeric"),
    ("Which brand owns Lamborghini?", "entity"),
    ("What river runs through Egypt?", "location"),
    ("Who discovered penicillin?", "human"),
    ("How tall is Mount Everest in meters?", "numeric"),
    ("Which company developed Windows?", "entity"),
    ("Where is the Great Barrier Reef located?", "location"),
    ("Who composed the Ninth Symphony?", "human"),
    ("What is the population of Tokyo?", "numeric"),
    ("Which automaker produces the Model S?", "entity"),
    ("What desert covers most of northern Africa?", "location"),
    ("Who invented the telephone?", "human"),
    ("How many bones are in the adult human body?", "numeric"),
    ("Which firm owns Instagram?", "entity"),
]


@dataclass
class OolongInstance:
    query: str
    context: str
    gold: int  # the count
    target_chars: int
    target_label: str


def make_instance(target_chars: int, seed: int) -> OolongInstance:
    rng = random.Random(seed)
    target_label = rng.choice(["location", "numeric", "human", "entity"])

    lines: list[str] = []
    label_counts = {"location": 0, "numeric": 0, "human": 0, "entity": 0}
    uid = 0
    while sum(len(l) + 1 for l in lines) < target_chars:
        text, label = rng.choice(SEED_LINES)
        uid += 1
        lines.append(f"user_{uid:04d} | 2023-{rng.randint(1,12):02d}-{rng.randint(1,28):02d} | {text}")
        label_counts[label] += 1

    rng.shuffle(lines)
    context = "\n".join(lines)
    gold = label_counts[target_label]

    label_human = {
        "location": "places, cities, rivers, mountains, or geographic locations",
        "numeric": "numeric values, counts, measurements, or quantities",
        "human": "specific people (asking 'who')",
        "entity": "companies, brands, or organizations",
    }[target_label]

    query = (
        "Below is a list of general-knowledge questions, one per line, in the "
        f"format `user_id | date | question`.\n\n"
        f"Count how many of these questions are about {label_human}. "
        f"Each question can be classified into exactly one category — you must infer "
        f"the category from the semantics of the question. "
        f"Reply with exactly one integer (the count) and nothing else."
    )
    return OolongInstance(
        query=query,
        context=context,
        gold=gold,
        target_chars=target_chars,
        target_label=target_label,
    )


def score(prediction: str, gold: int) -> float:
    """Exact-match on the integer. Tolerant to surrounding text."""
    import re
    nums = re.findall(r"-?\d+", prediction)
    if not nums:
        return 0.0
    # take the LAST number — models often print working then a final answer
    pred = int(nums[-1])
    return 1.0 if pred == gold else 0.0
