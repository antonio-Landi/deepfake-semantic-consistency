#!/usr/bin/env python3
"""Calcola BERTScore e metriche semantiche a partire da annotazioni VLM gia generate.

Questo script NON usa Qwen. Legge i file JSONL di cache gia esistenti nelle due
cartelle (originali fake e attacked), abbina le coppie per rel_path, e calcola:
  - BERTScore F1 (roberta-large)
  - Semantic Consistency (attributi VLM)
  - Concept Retention (sovrapposizione lessicale)

Output nella cartella --output-dir:
  pairs.csv    : metriche per ogni coppia
  summary.json : medie aggregate

Uso rapido (senza argomenti CLI - usa la QUICK CONFIG sotto):
  python compute_bertscore_from_cache.py

Uso con argomenti:
  python compute_bertscore_from_cache.py --original-cache path/orig.jsonl --attacked-cache path/atk.jsonl --output-dir path/out
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# ---------------------------------------------------------------------------
# QUICK CONFIG - modifica questi path se vuoi avviare senza argomenti CLI
# ---------------------------------------------------------------------------
DEFAULT_ORIGINAL_CACHE = r"C:\Users\Antonio\Documents\GitHub\AADD_Challenge\NLP\nlp_semantic_results-test\annotations_v2_neutral_caption.jsonl"
DEFAULT_ATTACKED_CACHE = r"C:\Users\Antonio\Documents\GitHub\AADD_Challenge\NLP\nlp_semantic_results-attacked\annotations_v2_neutral_caption.jsonl"
DEFAULT_OUTPUT_DIR     = r"C:\Users\Antonio\Documents\GitHub\AADD_Challenge\NLP\nlp_semantic_results-paired"

# BERTScore config
BERTSCORE_MODEL            = "roberta-large"
BERTSCORE_LANG             = "en"
BERTSCORE_USE_IDF          = False
BERTSCORE_RESCALE_BASELINE = False

PROMPT_VERSION = "v2_neutral_caption"

ATTRIBUTES: Dict[str, set] = {
    "visible_smile":   {"yes", "no", "unknown"},
    "eyeglasses":      {"yes", "no", "unknown"},
    "head_pose":       {"frontal", "three_quarter", "profile", "unknown"},
    "headwear":        {"yes", "no", "unknown"},
    "background_type": {"indoor", "outdoor", "plain", "other", "unknown"},
}

STOPWORDS = {
    "a","an","and","are","as","at","be","by","close","closeup","for","from",
    "has","in","into","is","it","its","of","on","or","person","portrait",
    "shows","the","this","to","up","wearing","with",
}
SENSITIVE_TERMS = {
    "adolescent","adult","age","baby","boy","child","children","elderly",
    "ethnicity","female","gender","girl","identity","lady","male","man","men",
    "middle_aged","old","older","race","teen","teenage","teens","youth",
    "woman","women","young",
}


@dataclass
class AnnotationRow:
    role: str
    rel_path: str
    caption: str
    visible_smile: str
    eyeglasses: str
    head_pose: str
    headwear: str
    background_type: str
    caption_status: str

    def attr_value(self, name: str) -> str:
        return getattr(self, name)


@dataclass
class PairMetrics:
    pair_id: str
    rel_path: str
    caption_original: str
    caption_attacked: str
    caption_status_original: str
    caption_status_attacked: str
    visible_smile_original: str
    visible_smile_attacked: str
    eyeglasses_original: str
    eyeglasses_attacked: str
    head_pose_original: str
    head_pose_attacked: str
    headwear_original: str
    headwear_attacked: str
    background_type_original: str
    background_type_attacked: str
    evaluable_attrs: int
    matched_attrs: int
    semantic_consistency: Optional[float]
    concept_retention: Optional[float]
    bertscore_f1: Optional[float]
    text_metrics_eligible: bool
    status: str


def normalize_value(name: str, value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in ATTRIBUTES[name]:
        return "unknown"
    return normalized


def load_jsonl(path: Path) -> Dict[str, AnnotationRow]:
    rows: Dict[str, AnnotationRow] = {}
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("prompt_version") != PROMPT_VERSION:
                print(f"[WARN] Prompt version diversa ({data.get('prompt_version')}), riga saltata.", file=sys.stderr)
                continue
            rel = str(data["rel_path"])
            rows[rel] = AnnotationRow(
                role=str(data.get("role", "")),
                rel_path=rel,
                caption=str(data.get("caption", "")),
                visible_smile=normalize_value("visible_smile", data.get("visible_smile", "unknown")),
                eyeglasses=normalize_value("eyeglasses", data.get("eyeglasses", "unknown")),
                head_pose=normalize_value("head_pose", data.get("head_pose", "unknown")),
                headwear=normalize_value("headwear", data.get("headwear", "unknown")),
                background_type=normalize_value("background_type", data.get("background_type", "unknown")),
                caption_status=str(data.get("caption_status", "accepted")),
            )
    return rows


def tokenize_caption(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z_'-]*", text.lower())
    cleaned = []
    for token in tokens:
        token = token.strip("_'-").replace("-", "_")
        if len(token) < 3:
            continue
        if token in STOPWORDS or token in SENSITIVE_TERMS:
            continue
        cleaned.append(token)
    return cleaned


def extract_concepts(text: str) -> set:
    tokens = tokenize_caption(text)
    concepts = set(tokens)
    for idx in range(len(tokens) - 1):
        concepts.add(f"{tokens[idx]} {tokens[idx + 1]}")
    return concepts


def concept_retention_score(orig: str, atk: str) -> Optional[float]:
    o = extract_concepts(orig)
    if not o:
        return None
    a = extract_concepts(atk)
    return len(o & a) / len(o)


def score_pair(idx: int, rel_path: str, orig: AnnotationRow, atk: AnnotationRow, min_attrs: int) -> PairMetrics:
    evaluable = [name for name in ATTRIBUTES if orig.attr_value(name) != "unknown"]
    matched   = sum(1 for name in evaluable if orig.attr_value(name) == atk.attr_value(name))

    if len(evaluable) >= min_attrs:
        sem_cons = matched / len(evaluable)
        status   = "evaluated"
    else:
        sem_cons = None
        status   = "excluded_few_clear_attributes"

    text_eligible = (
        orig.caption_status.startswith("accepted")
        and atk.caption_status.startswith("accepted")
    )

    return PairMetrics(
        pair_id=f"{idx:06d}",
        rel_path=rel_path,
        caption_original=orig.caption,
        caption_attacked=atk.caption,
        caption_status_original=orig.caption_status,
        caption_status_attacked=atk.caption_status,
        visible_smile_original=orig.visible_smile,
        visible_smile_attacked=atk.visible_smile,
        eyeglasses_original=orig.eyeglasses,
        eyeglasses_attacked=atk.eyeglasses,
        head_pose_original=orig.head_pose,
        head_pose_attacked=atk.head_pose,
        headwear_original=orig.headwear,
        headwear_attacked=atk.headwear,
        background_type_original=orig.background_type,
        background_type_attacked=atk.background_type,
        evaluable_attrs=len(evaluable),
        matched_attrs=matched,
        semantic_consistency=sem_cons,
        concept_retention=(
            concept_retention_score(orig.caption, atk.caption) if text_eligible else None
        ),
        bertscore_f1=None,
        text_metrics_eligible=text_eligible,
        status=status,
    )


def compute_bertscore(rows: List[PairMetrics]) -> str:
    target = [r for r in rows if r.text_metrics_eligible and r.caption_original and r.caption_attacked]
    if not target:
        return "skipped_no_eligible_captions"
    try:
        from bert_score import score as bs_score
    except ImportError:
        print("\n[ERRORE] bert-score non installato. Esegui:\n  pip install bert-score", file=sys.stderr)
        return "skipped_missing_bert_score_package"

    candidates = [r.caption_attacked for r in target]
    references = [r.caption_original for r in target]

    print(f"\nCalcolo BERTScore su {len(candidates)} coppie eligible (modello: {BERTSCORE_MODEL})...")
    _, _, f1 = bs_score(
        candidates, references,
        lang=BERTSCORE_LANG,
        model_type=BERTSCORE_MODEL,
        idf=BERTSCORE_USE_IDF,
        rescale_with_baseline=BERTSCORE_RESCALE_BASELINE,
        verbose=True,
    )
    for row, val in zip(target, f1.tolist()):
        row.bertscore_f1 = float(val)
    return "computed"


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def count_values(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts


def write_pairs_csv(path: Path, rows: Sequence[PairMetrics]) -> None:
    fieldnames = list(PairMetrics.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})


def build_summary(rows: Sequence[PairMetrics], bs_status: str) -> Dict[str, Any]:
    evaluated = [r for r in rows if r.semantic_consistency is not None]
    per_attr: Dict[str, Any] = {}
    for name in ATTRIBUTES:
        total, matched = 0, 0
        for r in rows:
            if getattr(r, f"{name}_original") == "unknown":
                continue
            total += 1
            if getattr(r, f"{name}_original") == getattr(r, f"{name}_attacked"):
                matched += 1
        per_attr[name] = {
            "evaluable": total,
            "matched": matched,
            "retention": (matched / total) if total else None,
        }

    return {
        "prompt_version": PROMPT_VERSION,
        "bertscore_model": BERTSCORE_MODEL,
        "bertscore_lang": BERTSCORE_LANG,
        "total_pairs": len(rows),
        "evaluated_pairs": len(evaluated),
        "excluded_pairs": len(rows) - len(evaluated),
        "mean_evaluable_attrs": mean(float(r.evaluable_attrs) for r in rows),
        "semantic_consistency_mean": mean(r.semantic_consistency for r in rows),
        "concept_retention_mean": mean(r.concept_retention for r in rows),
        "bertscore_f1_mean": mean(r.bertscore_f1 for r in rows),
        "bertscore_status": bs_status,
        "text_metrics_eligible_pairs": sum(1 for r in rows if r.text_metrics_eligible),
        "caption_status_original": count_values(r.caption_status_original for r in rows),
        "caption_status_attacked": count_values(r.caption_status_attacked for r in rows),
        "per_attribute": per_attr,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcola BERTScore da annotazioni VLM gia generate (senza rieseguire Qwen)."
    )
    parser.add_argument("--original-cache", type=Path, default=DEFAULT_ORIGINAL_CACHE,
                        help="JSONL delle annotazioni delle fake originali.")
    parser.add_argument("--attacked-cache", type=Path, default=DEFAULT_ATTACKED_CACHE,
                        help="JSONL delle annotazioni delle immagini attacked.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Cartella dove salvare pairs.csv e summary.json.")
    parser.add_argument("--min-evaluable-attrs", type=int, default=3,
                        help="Numero minimo di attributi chiari per includere una coppia.")
    parser.add_argument("--skip-bertscore", action="store_true",
                        help="Salta il calcolo di BERTScore.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    print(f"Carico annotazioni originali da: {args.original_cache}")
    originals = load_jsonl(Path(args.original_cache))
    print(f"  -> {len(originals)} immagini caricate")

    print(f"Carico annotazioni attacked da:  {args.attacked_cache}")
    attacked = load_jsonl(Path(args.attacked_cache))
    print(f"  -> {len(attacked)} immagini caricate")

    common    = sorted(set(originals) & set(attacked))
    only_orig = set(originals) - set(attacked)
    only_atk  = set(attacked)  - set(originals)

    if only_orig:
        print(f"[WARN] {len(only_orig)} solo in originali (nessun attacked): {sorted(only_orig)}", file=sys.stderr)
    if only_atk:
        print(f"[WARN] {len(only_atk)} solo in attacked (nessun originale): {sorted(only_atk)}", file=sys.stderr)

    if not common:
        print("[ERRORE] Nessuna coppia trovata. Controlla i rel_path nei JSONL.", file=sys.stderr)
        return 1

    print(f"\nCoppie abbinate: {len(common)}")

    rows: List[PairMetrics] = []
    for idx, rel_path in enumerate(common, start=1):
        rows.append(score_pair(idx, rel_path, originals[rel_path], attacked[rel_path], args.min_evaluable_attrs))

    bs_status = "skipped_by_user"
    if not args.skip_bertscore:
        bs_status = compute_bertscore(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_csv    = output_dir / "pairs.csv"
    summary_json = output_dir / "summary.json"

    write_pairs_csv(pairs_csv, rows)
    summary = build_summary(rows, bs_status)
    with summary_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print("\n" + "="*65)
    print("RISULTATI")
    print("="*65)
    print(f"  Coppie totali             : {summary['total_pairs']}")
    print(f"  Coppie valutate           : {summary['evaluated_pairs']}")
    sc = summary['semantic_consistency_mean']
    cr = summary['concept_retention_mean']
    bf = summary['bertscore_f1_mean']
    print(f"  Semantic Consistency mean : {sc:.4f}" if sc is not None else "  Semantic Consistency mean : N/A")
    print(f"  Concept Retention mean    : {cr:.4f}" if cr is not None else "  Concept Retention mean    : N/A")
    if bf is not None:
        print(f"  BERTScore F1 mean         : {bf:.4f}  <-- principale metrica NLP")
        print()
        print(f"  {'rel_path':<12} {'BERTScore F1':>13}  caption_original[:40]")
        print(f"  {'-'*12} {'-'*13}  {'-'*40}")
        for row in rows:
            bs = f"{row.bertscore_f1:.4f}" if row.bertscore_f1 is not None else "       N/A"
            print(f"  {row.rel_path:<12} {bs:>13}  {row.caption_original[:60]}")
    else:
        print(f"  BERTScore F1              : {bs_status}")
    print("="*65)
    print(f"\nSalvato pairs.csv    -> {pairs_csv}")
    print(f"Salvato summary.json -> {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
