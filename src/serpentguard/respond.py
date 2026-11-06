"""The app's response: WHO SEARO 2016 guideline passages, cut from the local PDF by the committed index
(data/m3/kb/who_passages.csv: page, `pdftotext -raw` offsets and a hash, never the text), and one species card per
class. R0 looks passages up by verdict and species, R1 ranks them with BM25. Notebook 16 builds the index and the
cards and runs Q1."""
import functools
import hashlib
import json
import os
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from serpentguard.decision import NOT_LISTED_TEXT, REASON_TEXT, UNKNOWN_MESSAGE
from serpentguard.medical_importance import GUIDELINE, MEDICAL_CSV, pages
from serpentguard.metrics import DANGEROUS, NOT_LISTED, VERDICTS

REPO_ROOT = Path(__file__).resolve().parents[2]
KB_DIR = REPO_ROOT / "data" / "m3" / "kb"
INDEX_CSV, CARDS_CSV = KB_DIR / "who_passages.csv", KB_DIR / "species_cards.csv"
WHO_PDF_ENV = "SERPENTGUARD_WHO_PDF"
WHO_PDF_CANDIDATES = (REPO_ROOT / "refs" / "who_searo_2016.pdf", REPO_ROOT.parent / "refs" / "who_searo_2016.pdf")
PDFTOTEXT = ("pdftotext", "-raw", "-enc", "UTF-8", "-eol", "unix")
WHO_SOURCE, CARD_SOURCE, NOTE_SOURCE = GUIDELINE, "SerpentGuard species card", "SerpentGuard"
HASH_CHARS = 16

TOP_K = 3
FIRST_AID_IDS = ("who-110-01", "who-119-04", "who-118-04", "who-119-03", "who-120-02", "who-120-05")
NOT_LISTED_NOTE_ID, UNKNOWN_NOTE_ID = "note-not-listed", "note-unknown"
UNKNOWN_NOTE = UNKNOWN_MESSAGE + "; give first aid as if the snake were dangerous"

STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between both but
by can could did do does doing down during each few for from further had has have having he her here hers him his
how i if in into is it its itself just me more most my no nor not now of off on once only or other our ours out over
own same she should so some such than that the their theirs them then there these they this those through to too
under until up very was we were what when where which while who whom why will with would you your yours
im ive dont didnt doesnt isnt cant got get gets also okay ok please yes one ones
around along across onto upon toward towards within without
""".split())
PLURAL_MIN = 4                      # strip a final "s" from longer words, but not "ss"
BM25_K1, BM25_B = 1.5, 0.75         # rank_bm25's defaults


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    source: str
    page: int | None = None
    section: str = ""
    heading: str = ""
    topic: str = ""
    species: str = ""

    @property
    def citation(self):
        return f"{self.source}, {pages(self.page)}" if self.page else self.source


@dataclass(frozen=True)
class Question:
    """A typed question; Q1's questions also carry a style, a topic and their gold passage ids."""
    id: str
    text: str
    style: str
    verdict: str
    species: str
    reasons: tuple
    topic: str
    gold: tuple


def find_who_pdf():
    """The guideline PDF: $SERPENTGUARD_WHO_PDF, else refs/ in or beside the repo; None when absent."""
    candidates = [Path(os.environ[WHO_PDF_ENV])] if os.environ.get(WHO_PDF_ENV) else WHO_PDF_CANDIDATES
    return next((path for path in candidates if path.is_file()), None)


@functools.cache
def page_text(pdf, page=None):
    """One page (or with no page the whole PDF) as `pdftotext -raw` gives it; every passage's offsets index a page."""
    page_args = () if page is None else ("-f", str(page), "-l", str(page))
    return subprocess.run([*PDFTOTEXT, *page_args, str(pdf), "-"], capture_output=True, text=True, encoding="utf-8",
                          check=True).stdout


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:HASH_CHARS]


def read_index(path=INDEX_CSV):
    return pd.read_csv(path, dtype={"section": str, "id": str, "sha256": str})


def who_passages(pdf, index=None):
    """The indexed passages cut from the PDF; a passage whose hash differs is an error."""
    index = read_index() if index is None else index
    passages = []
    for row in index.itertuples(index=False):
        text = page_text(pdf, int(row.page))[int(row.start):int(row.end)]
        if text_hash(text) != row.sha256:
            raise ValueError(f"{row.id}: the text cut from {pdf} does not match its hash; check the PDF and pdftotext")
        passages.append(Passage(row.id, text, WHO_SOURCE, int(row.page), row.section, row.heading, row.topic))
    return passages


def read_cards(path=CARDS_CSV):
    return pd.read_csv(path, keep_default_na=False, dtype={"who_category": str})


def card_passages(cards):
    return [Passage(row.id, row.text, CARD_SOURCE, species=row.species, topic="species")
            for row in cards.itertuples(index=False)]


def build_kb(who_pdf):
    """The guideline's indexed passages, cut from who_pdf, then the committed species cards."""
    return who_passages(who_pdf) + card_passages(read_cards())


def tokenize(text):
    """Lower-case words without stopwords, a final plural "s" stripped."""
    words = "".join(char.lower() if char.isalnum() else " " for char in text.replace("'", "").replace("’", "")).split()
    return [word[:-1] if len(word) > PLURAL_MIN and word.endswith("s") and not word.endswith("ss") else word
            for word in words if word not in STOPWORDS]


def query_text(query):
    return query.text if isinstance(query, Question) else str(query)


def text_bytes(passages):
    return sum(len(p.text.encode()) for p in passages)


@functools.cache
def dangerous_species(medical_csv=MEDICAL_CSV):
    table = pd.read_csv(medical_csv, keep_default_na=False)
    return frozenset(table.loc[table["dangerous"].astype(str) == "True", "species"])


@functools.cache
def _default_cards():
    return {p.species: p for p in card_passages(read_cards())}


def species_card(verdict, species=None, cards=None, dangerous=None):
    """Dangerous: the highest-ranked dangerous species' card (none if no top-3 species is); not listed: the top one's."""
    label, ranked, _ = _verdict_parts(verdict, species, ())
    cards = _default_cards() if cards is None else cards
    dangerous = dangerous_species() if dangerous is None else dangerous
    if label == DANGEROUS:
        name = next((name for name in ranked if name in dangerous), None)
    elif label == NOT_LISTED:
        name = ranked[0] if ranked else None
    else:
        name = None
    return cards.get(name)


class LookupRetriever:
    """R0: fixed passages per verdict: first aid and the card, or a note, the card and first aid."""

    def __init__(self, kb, medical_csv=MEDICAL_CSV):
        by_id = {p.id: p for p in kb}
        self.first_aid = [by_id[pid] for pid in FIRST_AID_IDS]
        self.cards = {p.species: p for p in kb if p.source == CARD_SOURCE}
        self.dangerous = dangerous_species(medical_csv)

    def search(self, query_or_verdict, k=TOP_K, species=None, reasons=()):
        label, ranked_species, reasons = _verdict_parts(query_or_verdict, species, reasons)
        shown = species_card(label, ranked_species, self.cards, self.dangerous)
        card = [shown] if shown else []
        if label == DANGEROUS:
            ranked = self.first_aid + card
        elif label == NOT_LISTED:
            ranked = [Passage(NOT_LISTED_NOTE_ID, NOT_LISTED_TEXT, NOTE_SOURCE)] + card + self.first_aid
        else:
            why = ", ".join(REASON_TEXT[r] for r in reasons if r in REASON_TEXT) or "no reason recorded"
            ranked = [Passage(UNKNOWN_NOTE_ID, UNKNOWN_NOTE.format(why), NOTE_SOURCE)] + self.first_aid
        return ranked[:k]

    def size_bytes(self):
        table = {"first_aid": FIRST_AID_IDS, "cards": {name: p.id for name, p in self.cards.items()}}
        notes = NOT_LISTED_TEXT + UNKNOWN_NOTE + "".join(REASON_TEXT.values())
        return (text_bytes(self.first_aid) + text_bytes(self.cards.values()) + len(notes.encode())
                + len(json.dumps(table).encode()))


def _verdict_parts(query, species, reasons):
    """(label, ranked species names, reasons) from a decision.Verdict, a Question, or a verdict label."""
    if isinstance(query, Question):
        return query.verdict, (query.species,) if query.species else (), query.reasons
    if hasattr(query, "label") and hasattr(query, "top3"):
        return query.label, tuple(entry[0] for entry in query.top3), tuple(query.reason)
    if query not in VERDICTS:
        raise ValueError(f"R0 looks up a verdict ({', '.join(VERDICTS)}), not free text: {query!r}")
    named = () if species is None else (species,) if isinstance(species, str) else tuple(species)
    return query, named, tuple(reasons)


class BM25Retriever:
    """R1: Okapi BM25 over every passage's words."""

    def __init__(self, kb, k1=BM25_K1, b=BM25_B):
        self.passages = list(kb)
        self.bm25 = BM25Okapi([tokenize(p.text) for p in self.passages], k1=k1, b=b)

    def search(self, query, k=TOP_K):
        scores = self.bm25.get_scores(tokenize(query_text(query)))
        return [self.passages[i] for i in np.argsort(-scores, kind="stable")[:k]]

    def size_bytes(self):
        return text_bytes(self.passages) + len(pickle.dumps(self.bm25))
