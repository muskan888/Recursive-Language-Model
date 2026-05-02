"""S-NIAH (single-needle-in-a-haystack), faithful to RULER's setup.

Plant a unique 'magic-number' phrase at a random position in a sea of filler.
Score = exact-substring match for the number in the model's answer.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

FILLER_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Scientists in 1923 first described the phenomenon in detail.",
    "Pasta dishes from northern Italy often use butter rather than olive oil.",
    "Migratory birds rely on magnetic fields for long-distance navigation.",
    "The library on Elm Street stays open until midnight on weekends.",
    "Volcanic soil tends to be unusually rich in mineral nutrients.",
    "Several cities in Patagonia experience strong westerly winds year round.",
    "Old steam locomotives required regular replenishment of water and coal.",
    "Honeybees communicate the location of food sources through dance.",
    "Cathedrals built in the gothic style emphasize verticality and light.",
    "Coffee beans were originally chewed before they were brewed as a drink.",
    "The bridge over the gorge sways noticeably during high winds.",
    "Mosses can survive in extremely dry conditions and revive rapidly when wet.",
    "Ancient Roman concrete used volcanic ash for added strength.",
    "Some species of octopus can solve simple puzzles to obtain food.",
]


@dataclass
class SNiahInstance:
    query: str
    context: str
    gold: str  # the magic number as a string
    target_chars: int


def make_instance(target_chars: int, seed: int) -> SNiahInstance:
    rng = random.Random(seed)
    magic = str(rng.randint(10_000_000, 99_999_999))  # 8-digit
    needle = f"The special magic number for the experiment is {magic}."

    sentences: list[str] = []
    while sum(len(s) + 1 for s in sentences) < target_chars:
        sentences.append(rng.choice(FILLER_SENTENCES))
    insert_at = rng.randint(1, max(1, len(sentences) - 1))
    sentences.insert(insert_at, needle)
    context = " ".join(sentences)

    query = (
        "What is the special magic number for the experiment that is mentioned "
        "in the context? Reply with just the number, nothing else."
    )
    return SNiahInstance(query=query, context=context, gold=magic, target_chars=target_chars)


def score(prediction: str, gold: str) -> float:
    return 1.0 if gold in prediction else 0.0
