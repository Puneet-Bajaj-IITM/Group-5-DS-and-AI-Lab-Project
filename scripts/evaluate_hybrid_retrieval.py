"""Evaluate BM25, MiniLM and RRF on the versioned Q1 query set."""

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort
from rank_bm25 import BM25Okapi
from tokenizers import Tokenizer

from serpentguard.retrieval import reciprocal_rank_fusion


ROOT = Path(__file__).resolve().parents[1]
STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has
have having he her here hers him his how i if in into is it its itself just me more most my no
nor not now of off on once only or other our ours out over own same she should so some such than
that the their theirs them then there these they this those through to too under until up very
was we were what when where which while who whom why will with would you your yours im ive dont
didnt doesnt isnt cant got get gets also okay ok please yes one ones around along across onto
upon toward towards within without
""".split())


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def extract_corpus(pdf, pdftotext, *, exclude_unmatched=False):
    """Recreate the notebook corpus and verify each WHO passage against its hash."""
    kb = ROOT / "data" / "m3" / "kb"
    rows = csv_rows(kb / "who_passages.csv")
    pages = {}
    verified, unmatched = [], []
    for row in rows:
        page = row["page"]
        if page not in pages:
            output = subprocess.check_output([
                pdftotext, "-raw", "-enc", "UTF-8", "-eol", "unix",
                "-f", page, "-l", page, str(pdf), "-",
            ])
            pages[page] = output.decode("utf-8")
        passage = pages[page][int(row["start"]):int(row["end"])]
        if hashlib.sha256(passage.encode()).hexdigest()[:16] != row["sha256"]:
            if not exclude_unmatched:
                raise ValueError(f"Source passage does not match the repository hash: {row['id']}")
            unmatched.append(row["id"])
            continue
        row["text"] = passage
        verified.append(row)
    if not verified:
        raise ValueError("No WHO passages matched the repository hashes")
    return verified + csv_rows(kb / "species_cards.csv"), unmatched


def tokenize(text):
    text = text.replace("'", "").replace("’", "")
    words = "".join(c.lower() if c.isalnum() else " " for c in text).split()
    return [
        word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith("ss") else word
        for word in words if word not in STOPWORDS
    ]


def encoder(model, tokenizer_path, max_length):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length)
    tokenizer.no_padding()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    input_names = [item.name for item in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    def encode(text):
        token = tokenizer.encode(text)
        inputs = {
            "input_ids": token.ids, "attention_mask": token.attention_mask,
            "token_type_ids": token.type_ids,
        }
        values = {name: np.array([inputs[name]], dtype=np.int64) for name in input_names}
        hidden = session.run([output_name], values)[0][0]
        mask = np.asarray(token.attention_mask, dtype=np.float32)
        pooled = (hidden * mask[:, None]).sum(axis=0) / mask.sum()
        norm = np.linalg.norm(pooled)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("MiniLM returned a zero or non-finite embedding")
        return pooled / norm

    return encode


def first_hit(ranking, gold):
    return next((rank for rank, passage in enumerate(ranking, 1) if passage in gold), None)


def metrics(ranks):
    n = len(ranks)
    return {
        "questions": n,
        "recall_at_1": sum(rank == 1 for rank in ranks) / n,
        "recall_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / n,
        "mrr_at_10": sum(1 / rank for rank in ranks if rank is not None and rank <= 10) / n,
    }


def evaluate(args):
    corpus, unmatched = extract_corpus(
        args.who_pdf, args.pdftotext, exclude_unmatched=args.exclude_unmatched_passages,
    )
    questions = csv_rows(ROOT / "data" / "m3" / "response_questions.csv")
    passages = [document["id"] for document in corpus]
    if len(passages) != len(set(passages)):
        raise ValueError("Duplicate passage IDs in the corpus")
    passage_set = set(passages)
    bm25 = BM25Okapi([tokenize(document["text"]) for document in corpus])
    models = {"int4": args.int4_model}
    if args.fp32_model:
        models["fp32"] = args.fp32_model
    encoders = {
        name: encoder(model, args.tokenizer, args.max_length) for name, model in models.items()
    }
    vectors = {
        name: np.stack([encode(document["text"]) for document in corpus])
        for name, encode in encoders.items()
    }
    cosine = (
        (vectors["int4"] * vectors["fp32"]).sum(axis=1) if "fp32" in vectors else None
    )
    rows, excluded_questions = [], []
    for question in questions:
        text = question["question"]
        gold = set(question["gold_ids"].split(";"))
        if gold.intersection(unmatched):
            excluded_questions.append(question["id"])
            continue
        if not gold <= passage_set:
            raise ValueError(f"Question {question['id']} references unknown passage IDs")
        lexical = [passages[i] for i in np.argsort(-bm25.get_scores(tokenize(text)), kind="stable")]
        ranks = {"bm25": first_hit(lexical, gold)}
        for name, encode in encoders.items():
            scores = vectors[name] @ encode(text)
            dense = [passages[i] for i in np.argsort(-scores, kind="stable")]
            fused = reciprocal_rank_fusion([lexical, dense])
            ranks.update({f"minilm_{name}": first_hit(dense, gold), f"rrf_{name}": first_hit(fused, gold)})
        rows.append({"id": question["id"], "style": question["style"], **ranks})

    if not rows:
        raise ValueError("No evaluation questions have complete gold coverage")
    summary = {
        style: {
            name: metrics([row[name] for row in rows if style == "all" or row["style"] == style])
            for name in ranks
        }
        for style in ("all", *sorted({row["style"] for row in rows}))
    }
    result = {
        "passages": len(passages), "query_set": "M3 Q1 draft queries",
        "unmatched_passage_ids": unmatched, "excluded_query_ids": excluded_questions,
        "rrf_k": 60, "fusion_candidate_depth": len(passages),
        "model_bytes": {name: path.stat().st_size for name, path in models.items()},
        "tokenizer_bytes": args.tokenizer.stat().st_size,
        "float32_dense_index_bytes": vectors["int4"].nbytes,
        "corpus_int4_fp32_cosine": None if cosine is None else {
            "minimum": float(cosine.min()), "median": float(np.median(cosine)),
        },
        "metrics": summary, "rank_by_question": rows,
    }
    if args.dense_index:
        with args.dense_index.open("xb") as target:
            np.savez(target, passage_ids=np.asarray(passages), vectors=vectors["int4"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--who-pdf", type=Path, required=True)
    parser.add_argument("--pdftotext", default="pdftotext")
    parser.add_argument("--int4-model", type=Path, required=True)
    parser.add_argument("--fp32-model", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--exclude-unmatched-passages", action="store_true")
    parser.add_argument("--dense-index", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
