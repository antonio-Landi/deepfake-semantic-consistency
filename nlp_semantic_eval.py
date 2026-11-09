#!/usr/bin/env python3
"""Annotate and compare images for the lightweight NLP extension of DAELTA.

With both source folders configured, the script compares original fake images
with their attacked versions. With just one folder, it produces captions and
controlled visual attributes for that folder, which is useful for a small pilot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


# ---------------------------------------------------------------------------
# QUICK CONFIGURATION
# ---------------------------------------------------------------------------
# Edit these values for a local run.  Use None (or leave a value empty) when a
# folder is not needed.  Command-line options, if provided, override them.
#
# Examples:
# DEFAULT_ORIGINAL_DIR = "C:\Users\Antonio\Documents\GitHub\AADD_Challenge\AADD_2026_Test"
# DEFAULT_ATTACKED_DIR = "C:\Users\Antonio\Documents\GitHub\AADD_Challenge\attackeds"
# DEFAULT_IMAGE_LIMIT = 20       # pilot on 20 images
# DEFAULT_IMAGE_LIMIT = None     # process every available image
DEFAULT_ORIGINAL_DIR: Optional[str] = r"C:\Users\Antonio\Documents\GitHub\AADD_Challenge\AADD_2026_Test"
DEFAULT_ATTACKED_DIR: Optional[str] = r"C:\Users\Antonio\Documents\GitHub\AADD_Challenge\attacked"
DEFAULT_OUTPUT_DIR = r"C:\Users\Antonio\Documents\GitHub\AADD_Challenge\NLP\nlp_semantic_results-All"
DEFAULT_IMAGE_LIMIT: Optional[int] = None
DEFAULT_MODEL_REVISION = "main"
# Keep None for the ordinary protocol.  A recovery run can cap only oversized
# inputs in memory (for example 1792 px -> 1024 px) without changing files.
DEFAULT_MAX_IMAGE_SIDE: Optional[int] = None

# The standard annotation is always greedy.  This optional repeated audit is
# disabled for ordinary pilot/final runs because it multiplies VLM inference.
# Set it to 3 only for a small pilot when checking technical repeatability.
DEFAULT_GREEDY_STABILITY_RUNS = 0


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

ATTRIBUTES: Dict[str, set[str]] = {
    "visible_smile": {"yes", "no", "unknown"},
    "eyeglasses": {"yes", "no", "unknown"},
    "head_pose": {"frontal", "three_quarter", "profile", "unknown"},
    "headwear": {"yes", "no", "unknown"},
    "background_type": {"indoor", "outdoor", "plain", "other", "unknown"},
}

PROMPT_VERSION = "v2_neutral_caption"

# Caption-similarity protocol. These values are saved in summary.json so a
# BERTScore result can always be reproduced from the output alone.
DEFAULT_BERTSCORE_MODEL = "roberta-large"
BERTSCORE_USE_IDF = False
BERTSCORE_RESCALE_WITH_BASELINE = False

PROMPT = """Look only at directly visible, non-sensitive facts.
Return only valid JSON, without markdown or extra text.

Required JSON keys:
caption, visible_smile, eyeglasses, head_pose, headwear, background_type.

Allowed values:
- visible_smile: yes, no, unknown
- eyeglasses: yes, no, unknown
- head_pose: frontal, three_quarter, profile, unknown
- headwear: yes, no, unknown
- background_type: indoor, outdoor, plain, other, unknown

Caption rules (mandatory):
- Write exactly one short English sentence starting with "A person".
- Describe only visible clothing, accessories, pose, or background.
- Do not use age, gender, identity, ethnicity, emotion, or real/fake status.
- Never use demographic words such as man, woman, boy, girl, child, young,
  old, older, elderly, male, or female.

Use unknown when an attribute is unclear."""

RETRY_PROMPT = """Return the required JSON again for this image. The previous
caption was invalid because it did not follow the neutrality rules.

The caption must be exactly one short English sentence starting with "A person"
and must describe only visible clothing, accessories, pose, or background.
Do not mention age, gender, identity, ethnicity, emotion, or real/fake status.
Never use: man, woman, boy, girl, child, young, old, older, elderly, male,
female. Return only valid JSON, without markdown or extra text.

Required JSON keys:
caption, visible_smile, eyeglasses, head_pose, headwear, background_type.

Allowed values:
- visible_smile: yes, no, unknown
- eyeglasses: yes, no, unknown
- head_pose: frontal, three_quarter, profile, unknown
- headwear: yes, no, unknown
- background_type: indoor, outdoor, plain, other, unknown"""

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "close",
    "closeup",
    "for",
    "from",
    "has",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "person",
    "portrait",
    "shows",
    "the",
    "this",
    "to",
    "up",
    "wearing",
    "with",
}

SENSITIVE_TERMS = {
    "adolescent",
    "adult",
    "age",
    "baby",
    "boy",
    "child",
    "children",
    "elderly",
    "ethnicity",
    "female",
    "gender",
    "girl",
    "identity",
    "lady",
    "male",
    "man",
    "men",
    "middle_aged",
    "old",
    "older",
    "race",
    "teen",
    "teenage",
    "teens",
    "youth",
    "woman",
    "women",
    "young",
}

FORBIDDEN_CAPTION_TERMS = SENSITIVE_TERMS


@dataclass
class ImageAnnotation:
    caption: str
    visible_smile: str
    eyeglasses: str
    head_pose: str
    headwear: str
    background_type: str
    raw_text: str
    error: str = ""
    caption_status: str = "accepted"
    caption_violations: str = ""

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


@dataclass
class AnnotationRow:
    """One VLM description, used when only one image directory is supplied."""

    image_id: str
    rel_path: str
    role: str
    caption: str
    visible_smile: str
    eyeglasses: str
    head_pose: str
    headwear: str
    background_type: str
    caption_status: str
    caption_violations: str
    error: str


@dataclass
class GreedyStabilityRow:
    """Agreement of repeated fresh greedy annotations for one image."""

    rel_path: str
    role: str
    requested_runs: int
    successful_runs: int
    failed_runs: int
    vector_mode: str
    vector_mode_agreement: Optional[float]
    visible_smile_mode: str
    visible_smile_agreement: Optional[float]
    eyeglasses_mode: str
    eyeglasses_agreement: Optional[float]
    head_pose_mode: str
    head_pose_agreement: Optional[float]
    headwear_mode: str
    headwear_agreement: Optional[float]
    background_type_mode: str
    background_type_agreement: Optional[float]
    errors: str


def image_key(path: Path, folder: Path, match_mode: str) -> str:
    rel = path.relative_to(folder)
    if match_mode == "relative_path":
        return rel.as_posix()
    if match_mode == "stem":
        return rel.with_suffix("").as_posix()
    raise ValueError(f"Unknown match mode: {match_mode}")


def collect_images(folder: Path, match_mode: str) -> Dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")

    images: Dict[str, Path] = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            key = image_key(path, folder, match_mode)
            if key in images:
                raise RuntimeError(
                    f"Duplicate image key '{key}' in {folder}. "
                    "Use --match-mode relative_path or remove duplicates."
                )
            images[key] = path
    return images


def normalize_value(name: str, value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in ATTRIBUTES[name]:
        return "unknown"
    return normalized


def caption_violations(caption: str) -> List[str]:
    """Return deterministic violations of the neutral-caption policy."""
    normalized = caption.strip()
    if not normalized:
        return ["empty_caption"]

    violations: List[str] = []
    if not re.match(r"^a (person\b|portrait of a person\b)", normalized, flags=re.IGNORECASE):
        violations.append("non_neutral_opening")

    tokens = {
        token.replace("-", "_")
        for token in re.findall(r"[a-z]+(?:[-_][a-z]+)?", normalized.lower())
    }
    forbidden = sorted(tokens & FORBIDDEN_CAPTION_TERMS)
    if forbidden:
        violations.append("forbidden_terms=" + ",".join(forbidden))
    return violations


def fallback_caption(annotation: "ImageAnnotation") -> str:
    """Create a neutral traceable fallback without inventing new visual facts."""
    phrases: List[str] = []
    if annotation.visible_smile == "yes":
        phrases.append("has a visible smile")
    if annotation.eyeglasses == "yes":
        phrases.append("wears eyeglasses")
    if annotation.headwear == "yes":
        phrases.append("wears headwear")
    if annotation.head_pose == "frontal":
        phrases.append("is facing forward")
    elif annotation.head_pose == "three_quarter":
        phrases.append("is in a three-quarter pose")
    elif annotation.head_pose == "profile":
        phrases.append("is in profile")
    if annotation.background_type != "unknown":
        phrases.append(f"is against a {annotation.background_type} background")

    if not phrases:
        return "A person is visible."
    return "A person " + " and ".join(phrases) + "."


def extract_json_object(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            payload, _ = decoder.raw_decode(text[start:])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    raise ValueError("No JSON object found in model output")


def annotation_from_text(text: str) -> ImageAnnotation:
    payload = extract_json_object(text)
    caption = str(payload.get("caption", "")).strip()
    return ImageAnnotation(
        caption=caption,
        visible_smile=normalize_value("visible_smile", payload.get("visible_smile", "unknown")),
        eyeglasses=normalize_value("eyeglasses", payload.get("eyeglasses", "unknown")),
        head_pose=normalize_value("head_pose", payload.get("head_pose", "unknown")),
        headwear=normalize_value("headwear", payload.get("headwear", "unknown")),
        background_type=normalize_value("background_type", payload.get("background_type", "unknown")),
        raw_text=text,
    )


def annotation_to_dict(
    role: str,
    rel_path: str,
    annotation: ImageAnnotation,
    inference_fingerprint: str,
) -> Dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "inference_fingerprint": inference_fingerprint,
        "role": role,
        "rel_path": rel_path,
        "caption": annotation.caption,
        "visible_smile": annotation.visible_smile,
        "eyeglasses": annotation.eyeglasses,
        "head_pose": annotation.head_pose,
        "headwear": annotation.headwear,
        "background_type": annotation.background_type,
        "raw_text": annotation.raw_text,
        "error": annotation.error,
        "caption_status": annotation.caption_status,
        "caption_violations": annotation.caption_violations,
    }


def annotation_from_dict(row: Dict[str, Any]) -> ImageAnnotation:
    return ImageAnnotation(
        caption=str(row.get("caption", "")),
        visible_smile=normalize_value("visible_smile", row.get("visible_smile", "unknown")),
        eyeglasses=normalize_value("eyeglasses", row.get("eyeglasses", "unknown")),
        head_pose=normalize_value("head_pose", row.get("head_pose", "unknown")),
        headwear=normalize_value("headwear", row.get("headwear", "unknown")),
        background_type=normalize_value("background_type", row.get("background_type", "unknown")),
        raw_text=str(row.get("raw_text", "")),
        error=str(row.get("error", "")),
        caption_status=str(row.get("caption_status", "accepted")),
        caption_violations=str(row.get("caption_violations", "")),
    )


def load_annotation_cache(
    path: Path,
    inference_fingerprint: str,
) -> Dict[Tuple[str, str], ImageAnnotation]:
    cache: Dict[Tuple[str, str], ImageAnnotation] = {}
    if not path.exists():
        return cache

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if (
                row.get("prompt_version") != PROMPT_VERSION
                or row.get("inference_fingerprint") != inference_fingerprint
            ):
                continue
            key = (str(row["role"]), str(row["rel_path"]))
            cache[key] = annotation_from_dict(row)
    return cache


def append_annotation(
    path: Path,
    role: str,
    rel_path: str,
    annotation: ImageAnnotation,
    inference_fingerprint: str,
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                annotation_to_dict(role, rel_path, annotation, inference_fingerprint),
                ensure_ascii=False,
            )
        )
        handle.write("\n")


def package_version(package_name: str) -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(package_name)
    except (ImportError, PackageNotFoundError):
        return "not_installed"


def configuration_fingerprint(configuration: Dict[str, Any]) -> str:
    """Return a stable short identifier for caches created with one VLM setup."""
    payload = json.dumps(configuration, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class QwenAnnotator:
    def __init__(
        self,
        model_id: str,
        model_revision: str,
        device: str,
        max_new_tokens: int,
        max_image_side: Optional[int],
    ):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.model_id = model_id
        self.model_revision = model_revision
        self.device_requested = device
        self.max_new_tokens = max_new_tokens
        self.max_image_side = max_image_side
        self.processor = AutoProcessor.from_pretrained(model_id, revision=model_revision)

        model_kwargs: Dict[str, Any] = {"torch_dtype": "auto", "revision": model_revision}
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        if device != "auto":
            self.model.to(device)
        self.model.eval()

        resolved_revision = getattr(self.model.config, "_commit_hash", None) or "not_available"
        resolved_device = str(getattr(self.model, "device", "not_available"))
        # The cache fingerprint intentionally covers the VLM, prompt, and
        # decoding protocol. Image resizing is recorded separately because a
        # recovery may resize only previously failed images while safely
        # reusing successful annotations produced with the same VLM protocol.
        fingerprint_configuration = {
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
            "model_id": self.model_id,
            "model_revision_requested": self.model_revision,
            "model_revision_resolved": resolved_revision,
            "transformers_version": package_version("transformers"),
            "torch_version": self.torch.__version__,
            "device_requested": self.device_requested,
            "device_resolved": resolved_device,
            "model_dtype": str(getattr(self.model, "dtype", "not_available")),
            "generation": {
                "strategy": "greedy",
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": self.max_new_tokens,
            },
        }
        self.inference_fingerprint = configuration_fingerprint(fingerprint_configuration)
        self.inference_configuration = {
            **fingerprint_configuration,
            "image_preprocessing": {
                "max_image_side": self.max_image_side,
                "applies_only_when_source_is_larger": self.max_image_side is not None,
                "resample": "lanczos" if self.max_image_side is not None else "not_applied",
                "writes_source_files": False,
            },
        }

    def prepare_image(self, image: Image.Image) -> Image.Image:
        """Return an in-memory RGB image, shrinking only oversized inputs."""
        if self.max_image_side is None or max(image.size) <= self.max_image_side:
            return image
        scale = self.max_image_side / max(image.size)
        resized_size = tuple(max(1, round(dimension * scale)) for dimension in image.size)
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        return image.resize(resized_size, resample=resampling)

    def generate(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        try:
            inputs = inputs.to(self.model.device)
        except AttributeError:
            pass

        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
            )

        generated_ids = [
            output_ids_row[len(input_ids) :]
            for input_ids, output_ids_row in zip(inputs.input_ids, output_ids)
        ]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def annotate(self, image_path: Path) -> ImageAnnotation:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.prepare_image(image)
            first_text = self.generate(image, PROMPT)
            annotation = annotation_from_text(first_text)
            violations = caption_violations(annotation.caption)
            if not violations:
                return annotation

            try:
                retry_text = self.generate(image, RETRY_PROMPT)
                retry_annotation = annotation_from_text(retry_text)
                retry_violations = caption_violations(retry_annotation.caption)
            except Exception as exc:
                retry_text = f"[retry failed: {exc}]"
                retry_annotation = annotation
                retry_violations = ["retry_generation_failed"]

        all_violations = "; ".join(violations + retry_violations)
        raw_text = f"[first response]\n{first_text}\n[retry response]\n{retry_text}"
        if not retry_violations:
            retry_annotation.raw_text = raw_text
            retry_annotation.caption_status = "accepted_after_retry"
            retry_annotation.caption_violations = "; ".join(violations)
            return retry_annotation

        retry_annotation.caption = fallback_caption(retry_annotation)
        retry_annotation.raw_text = raw_text
        retry_annotation.caption_status = "fallback_after_invalid_retry"
        retry_annotation.caption_violations = all_violations
        return retry_annotation


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


def extract_concepts(text: str) -> set[str]:
    tokens = tokenize_caption(text)
    concepts = set(tokens)
    for idx in range(len(tokens) - 1):
        concepts.add(f"{tokens[idx]} {tokens[idx + 1]}")
    return concepts


def concept_retention(original_caption: str, attacked_caption: str) -> Optional[float]:
    original = extract_concepts(original_caption)
    if not original:
        return None
    attacked = extract_concepts(attacked_caption)
    return len(original & attacked) / len(original)


def score_pair(
    pair_id: str,
    rel_path: str,
    original: ImageAnnotation,
    attacked: ImageAnnotation,
    min_evaluable_attrs: int,
) -> PairMetrics:
    evaluable = [name for name in ATTRIBUTES if original.attr_value(name) != "unknown"]
    matched = sum(1 for name in evaluable if original.attr_value(name) == attacked.attr_value(name))
    if len(evaluable) >= min_evaluable_attrs:
        semantic_consistency = matched / len(evaluable)
        status = "evaluated"
    else:
        semantic_consistency = None
        status = "excluded_few_clear_attributes"

    text_metrics_eligible = (
        original.caption_status.startswith("accepted")
        and attacked.caption_status.startswith("accepted")
        and bool(original.caption)
        and bool(attacked.caption)
        and not original.error
        and not attacked.error
    )

    return PairMetrics(
        pair_id=pair_id,
        rel_path=rel_path,
        caption_original=original.caption,
        caption_attacked=attacked.caption,
        caption_status_original=original.caption_status,
        caption_status_attacked=attacked.caption_status,
        visible_smile_original=original.visible_smile,
        visible_smile_attacked=attacked.visible_smile,
        eyeglasses_original=original.eyeglasses,
        eyeglasses_attacked=attacked.eyeglasses,
        head_pose_original=original.head_pose,
        head_pose_attacked=attacked.head_pose,
        headwear_original=original.headwear,
        headwear_attacked=attacked.headwear,
        background_type_original=original.background_type,
        background_type_attacked=attacked.background_type,
        evaluable_attrs=len(evaluable),
        matched_attrs=matched,
        semantic_consistency=semantic_consistency,
        concept_retention=(
            concept_retention(original.caption, attacked.caption) if text_metrics_eligible else None
        ),
        bertscore_f1=None,
        text_metrics_eligible=text_metrics_eligible,
        status=status,
    )


def compute_bertscore(
    rows: List[PairMetrics],
    lang: str,
    model_type: str,
) -> str:
    target_rows = [
        row
        for row in rows
        if row.text_metrics_eligible and row.caption_original and row.caption_attacked
    ]
    candidates = [row.caption_attacked for row in target_rows]
    references = [row.caption_original for row in target_rows]
    if not candidates:
        return "skipped_no_captions"

    try:
        from bert_score import score
    except ImportError:
        return "skipped_missing_bert_score_package"

    kwargs: Dict[str, Any] = {
        "lang": lang,
        "model_type": model_type,
        "idf": BERTSCORE_USE_IDF,
        "rescale_with_baseline": BERTSCORE_RESCALE_WITH_BASELINE,
        "verbose": False,
    }
    _, _, f1 = score(candidates, references, **kwargs)
    for row, value in zip(target_rows, f1.tolist()):
        row.bertscore_f1 = float(value)
    return "computed"


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def count_values(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def mode_and_agreement(values: Sequence[str]) -> Tuple[str, Optional[float]]:
    """Return the deterministic mode and its proportion among non-failed runs."""
    if not values:
        return "unknown", None
    counts = Counter(values)
    mode = min(counts, key=lambda value: (-counts[value], value))
    return mode, counts[mode] / len(values)


def annotation_vector(annotation: ImageAnnotation) -> Tuple[str, ...]:
    return tuple(annotation.attr_value(name) for name in ATTRIBUTES)


def build_greedy_stability_row(
    rel_path: str,
    role: str,
    annotations: Sequence[ImageAnnotation],
) -> GreedyStabilityRow:
    successful = [annotation for annotation in annotations if not annotation.error]
    errors = [annotation.error for annotation in annotations if annotation.error]
    vector_values = ["|".join(annotation_vector(annotation)) for annotation in successful]
    vector_mode, vector_agreement = mode_and_agreement(vector_values)

    attribute_results: Dict[str, Tuple[str, Optional[float]]] = {}
    for name in ATTRIBUTES:
        attribute_results[name] = mode_and_agreement(
            [annotation.attr_value(name) for annotation in successful]
        )

    return GreedyStabilityRow(
        rel_path=rel_path,
        role=role,
        requested_runs=len(annotations),
        successful_runs=len(successful),
        failed_runs=len(annotations) - len(successful),
        vector_mode=vector_mode,
        vector_mode_agreement=vector_agreement,
        visible_smile_mode=attribute_results["visible_smile"][0],
        visible_smile_agreement=attribute_results["visible_smile"][1],
        eyeglasses_mode=attribute_results["eyeglasses"][0],
        eyeglasses_agreement=attribute_results["eyeglasses"][1],
        head_pose_mode=attribute_results["head_pose"][0],
        head_pose_agreement=attribute_results["head_pose"][1],
        headwear_mode=attribute_results["headwear"][0],
        headwear_agreement=attribute_results["headwear"][1],
        background_type_mode=attribute_results["background_type"][0],
        background_type_agreement=attribute_results["background_type"][1],
        errors="; ".join(errors),
    )


def build_greedy_stability_summary(
    rows: Sequence[GreedyStabilityRow],
    runs: int,
    inference_configuration: Dict[str, Any],
) -> Dict[str, Any]:
    per_attribute = {
        name: mean(getattr(row, f"{name}_agreement") for row in rows)
        for name in ATTRIBUTES
    }
    return {
        "audit": "repeated_greedy_annotation",
        "purpose": "technical_repeatability_only",
        "requested_runs_per_image": runs,
        "images_audited": len(rows),
        "successful_annotation_runs": sum(row.successful_runs for row in rows),
        "failed_annotation_runs": sum(row.failed_runs for row in rows),
        "full_attribute_vector_agreement_mean": mean(
            row.vector_mode_agreement for row in rows
        ),
        "per_attribute_mode_agreement_mean": per_attribute,
        "inference_configuration": inference_configuration,
    }


def write_greedy_stability_csv(path: Path, rows: Sequence[GreedyStabilityRow]) -> None:
    fieldnames = list(GreedyStabilityRow.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})


def bertscore_configuration(lang: str, model_type: str) -> Dict[str, Any]:
    """Describe the fixed caption-similarity protocol saved with each run."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        package_version = version("bert-score")
    except (ImportError, PackageNotFoundError):
        package_version = "not_installed"

    return {
        "metric": "BERTScore_F1",
        "candidate": "caption_attacked",
        "reference": "caption_original",
        "language": lang,
        "model_type": model_type,
        "idf": BERTSCORE_USE_IDF,
        "rescale_with_baseline": BERTSCORE_RESCALE_WITH_BASELINE,
        "package_version": package_version,
    }


def build_summary(
    rows: Sequence[PairMetrics],
    bertscore_status: str,
    bertscore_config: Dict[str, Any],
    inference_configuration: Dict[str, Any],
) -> Dict[str, Any]:
    evaluated = [row for row in rows if row.semantic_consistency is not None]
    per_attribute: Dict[str, Dict[str, Any]] = {}
    for name in ATTRIBUTES:
        total = 0
        matched = 0
        for row in rows:
            original_value = getattr(row, f"{name}_original")
            attacked_value = getattr(row, f"{name}_attacked")
            if original_value == "unknown":
                continue
            total += 1
            if original_value == attacked_value:
                matched += 1
        per_attribute[name] = {
            "evaluable": total,
            "matched": matched,
            "retention": (matched / total) if total else None,
        }

    return {
        "prompt_version": PROMPT_VERSION,
        "total_pairs": len(rows),
        "evaluated_pairs": len(evaluated),
        "excluded_pairs": len(rows) - len(evaluated),
        "mean_evaluable_attrs": mean(float(row.evaluable_attrs) for row in rows),
        "semantic_consistency_mean": mean(row.semantic_consistency for row in rows),
        "concept_retention_mean": mean(row.concept_retention for row in rows),
        "bertscore_f1_mean": mean(row.bertscore_f1 for row in rows),
        "bertscore_status": bertscore_status,
        "bertscore_configuration": bertscore_config,
        "inference_configuration": inference_configuration,
        "text_metrics_eligible_pairs": sum(1 for row in rows if row.text_metrics_eligible),
        "caption_status_original": count_values(row.caption_status_original for row in rows),
        "caption_status_attacked": count_values(row.caption_status_attacked for row in rows),
        "per_attribute": per_attribute,
    }


def write_pairs_csv(path: Path, rows: Sequence[PairMetrics]) -> None:
    fieldnames = list(PairMetrics.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})


def write_annotations_csv(path: Path, rows: Sequence[AnnotationRow]) -> None:
    """Save standalone descriptions when no original/attacked comparison is requested."""
    fieldnames = list(AnnotationRow.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})


def write_summary(path: Path, summary: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def make_annotator(args: argparse.Namespace) -> QwenAnnotator:
    return QwenAnnotator(
        model_id=args.model_id,
        model_revision=args.model_revision,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        max_image_side=args.max_image_side,
    )


def annotate_fresh(annotator: QwenAnnotator, image_path: Path) -> ImageAnnotation:
    """Generate one new annotation without consulting or updating a cache."""
    try:
        return annotator.annotate(image_path)
    except Exception as exc:
        return ImageAnnotation(
            caption="",
            visible_smile="unknown",
            eyeglasses="unknown",
            head_pose="unknown",
            headwear="unknown",
            background_type="unknown",
            raw_text="",
            error=str(exc),
        )


def annotate_or_load(
    cache: Dict[Tuple[str, str], ImageAnnotation],
    cache_path: Path,
    annotator: QwenAnnotator,
    role: str,
    rel_path: str,
    image_path: Path,
    overwrite: bool,
    retry_failed_cache: bool,
) -> ImageAnnotation:
    key = (role, rel_path)
    if (
        not overwrite
        and key in cache
        and not (retry_failed_cache and cache[key].error)
    ):
        return cache[key]

    annotation = annotate_fresh(annotator, image_path)
    cache[key] = annotation
    append_annotation(
        cache_path,
        role,
        rel_path,
        annotation,
        annotator.inference_fingerprint,
    )
    return annotation


def backup_recovery_artifacts(output_dir: Path, cache_path: Path) -> Path:
    """Copy current artifacts before a failed-cache recovery changes derived files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = output_dir / f"recovery_backup_{timestamp}"
    suffix = 1
    while backup_dir.exists():
        backup_dir = output_dir / f"recovery_backup_{timestamp}_{suffix}"
        suffix += 1
    backup_dir.mkdir()

    for artifact in (cache_path, output_dir / "pairs.csv", output_dir / "summary.json"):
        if artifact.exists():
            shutil.copy2(artifact, backup_dir / artifact.name)
    return backup_dir


def run_greedy_stability_audit(
    annotator: QwenAnnotator,
    image_sets: Sequence[Tuple[str, Dict[str, Path]]],
    rel_paths: Sequence[str],
    output_dir: Path,
    runs: int,
) -> Dict[str, Path]:
    """Repeat fresh greedy inference; this is intentionally independent of the cache."""
    raw_path = output_dir / f"greedy_stability_annotations_{annotator.inference_fingerprint}.jsonl"
    csv_path = output_dir / "greedy_stability.csv"
    summary_path = output_dir / "greedy_stability_summary.json"
    rows: List[GreedyStabilityRow] = []
    total_images = len(image_sets) * len(rel_paths)
    image_index = 0

    with raw_path.open("w", encoding="utf-8") as handle:
        for role, images in image_sets:
            for rel_path in rel_paths:
                image_index += 1
                print(
                    f"[stability {image_index:>4}/{total_images}] {role}: {rel_path}",
                    flush=True,
                )
                annotations: List[ImageAnnotation] = []
                for run_index in range(1, runs + 1):
                    annotation = annotate_fresh(annotator, images[rel_path])
                    annotations.append(annotation)
                    raw_row = annotation_to_dict(
                        role,
                        rel_path,
                        annotation,
                        annotator.inference_fingerprint,
                    )
                    raw_row.update(
                        {
                            "audit": "repeated_greedy_annotation",
                            "run_index": run_index,
                            "requested_runs_per_image": runs,
                        }
                    )
                    handle.write(json.dumps(raw_row, ensure_ascii=False))
                    handle.write("\n")
                rows.append(build_greedy_stability_row(rel_path, role, annotations))

    write_greedy_stability_csv(csv_path, rows)
    write_summary(
        summary_path,
        build_greedy_stability_summary(rows, runs, annotator.inference_configuration),
    )
    return {"raw_annotations": raw_path, "csv": csv_path, "summary": summary_path}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate images or compute DAELTA NLP semantic consistency for image pairs."
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=DEFAULT_ORIGINAL_DIR,
        help="Folder with original fake images. Optional when --attacked-dir or the quick config is set.",
    )
    parser.add_argument(
        "--attacked-dir",
        type=Path,
        default=DEFAULT_ATTACKED_DIR,
        help="Folder with DAELTA attacked images. Optional for description-only runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder for annotations and metrics (default: %(default)s).",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Hugging Face VLM checkpoint used for captions and attributes.",
    )
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_MODEL_REVISION,
        help=(
            "Hugging Face model revision. Use an immutable commit hash for a final run "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for the VLM. Use auto, cuda, cuda:0, mps, or cpu.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_IMAGE_LIMIT,
        help="Maximum number of images/pairs. Set None in QUICK CONFIG for all images.",
    )
    parser.add_argument(
        "--match-mode",
        choices=("stem", "relative_path"),
        default="stem",
        help="Pair images by same relative stem by default, so 002.jpg matches 002.png.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=220, help="Maximum VLM response length.")
    parser.add_argument(
        "--min-evaluable-attrs",
        type=int,
        default=3,
        help="Minimum clear attributes in the original image for a pair to enter the mean.",
    )
    parser.add_argument("--overwrite-cache", action="store_true", help="Regenerate annotations even if cached.")
    parser.add_argument(
        "--retry-failed-cache",
        action="store_true",
        help=(
            "Retry only cached annotations whose previous run recorded an error; successful cache "
            "entries are reused and existing artifacts are backed up first."
        ),
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=DEFAULT_MAX_IMAGE_SIDE,
        help=(
            "Maximum longest image side passed to the VLM. Oversized inputs are resized only in "
            "memory with Lanczos; source image files are never changed. Default: no cap."
        ),
    )
    parser.add_argument(
        "--greedy-stability-runs",
        type=int,
        default=DEFAULT_GREEDY_STABILITY_RUNS,
        help=(
            "Repeat fresh greedy VLM annotations this many times per selected image to audit "
            "technical repeatability; 0 disables the audit (default: %(default)s)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report matching pairs without loading the VLM.")
    parser.add_argument("--skip-bertscore", action="store_true", help="Skip BERTScore even if installed.")
    parser.add_argument("--bertscore-lang", default="en", help="Language passed to bert-score.")
    parser.add_argument(
        "--bertscore-model",
        default=DEFAULT_BERTSCORE_MODEL,
        help="BERTScore model_type recorded in summary.json (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer or None in QUICK CONFIG.")
    if args.greedy_stability_runs < 0 or args.greedy_stability_runs == 1:
        raise ValueError("--greedy-stability-runs must be 0 or an integer of at least 2.")
    if args.max_image_side is not None and args.max_image_side <= 0:
        raise ValueError("--max-image-side must be a positive integer or omitted.")
    if args.overwrite_cache and args.retry_failed_cache:
        raise ValueError("--overwrite-cache and --retry-failed-cache cannot be used together.")
    if not args.original_dir and not args.attacked_dir:
        raise ValueError(
            "Set DEFAULT_ORIGINAL_DIR and/or DEFAULT_ATTACKED_DIR in QUICK CONFIG, "
            "or provide --original-dir and/or --attacked-dir."
        )

    originals = collect_images(args.original_dir, args.match_mode) if args.original_dir else {}
    attacked = collect_images(args.attacked_dir, args.match_mode) if args.attacked_dir else {}
    paired_mode = args.original_dir is not None and args.attacked_dir is not None

    if paired_mode:
        rel_paths = sorted(set(originals) & set(attacked))
        if args.limit is not None:
            rel_paths = rel_paths[: args.limit]
        if not rel_paths:
            raise RuntimeError("No matching image pairs found by relative path.")

        missing_attacked = len(set(originals) - set(attacked))
        missing_original = len(set(attacked) - set(originals))
        if missing_attacked or missing_original:
            print(
                f"[WARN] Unmatched files: {missing_attacked} without attacked copy, "
                f"{missing_original} without original copy.",
                file=sys.stderr,
            )

        if args.dry_run:
            print(f"Mode: pair comparison")
            print(f"Matched pairs: {len(rel_paths)}")
            print(f"Match mode: {args.match_mode}")
            print("First pairs:")
            for rel_path in rel_paths[:10]:
                print(f"  {rel_path}: {originals[rel_path].name} <-> {attacked[rel_path].name}")
            return 0
    else:
        role = "original" if args.original_dir is not None else "attacked"
        images = originals if role == "original" else attacked
        rel_paths = sorted(images)
        if args.limit is not None:
            rel_paths = rel_paths[: args.limit]
        if not rel_paths:
            raise RuntimeError("No images found in the supplied directory.")

        if args.dry_run:
            print("Mode: description only")
            print(f"Role: {role}")
            print(f"Images to annotate: {len(rel_paths)}")
            print("First images:")
            for rel_path in rel_paths[:10]:
                print(f"  {rel_path}: {images[rel_path].name}")
            return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotator = make_annotator(args)
    cache_path = args.output_dir / (
        f"annotations_{PROMPT_VERSION}_{annotator.inference_fingerprint}.jsonl"
    )
    if args.overwrite_cache and cache_path.exists():
        cache_path.unlink()
    cache = load_annotation_cache(cache_path, annotator.inference_fingerprint)
    if args.retry_failed_cache:
        failed_cache_entries = [key for key, annotation in cache.items() if annotation.error]
        if not failed_cache_entries:
            raise RuntimeError(
                "No failed annotations were found in the matching cache. Recovery stopped to avoid "
                "an unintended full regeneration."
            )
        backup_dir = backup_recovery_artifacts(args.output_dir, cache_path)
        print(
            f"Recovery mode: retrying {len(failed_cache_entries)} failed cached annotations; "
            f"successful entries will be reused. Backup: {backup_dir}",
            flush=True,
        )

    if not paired_mode:
        role = "original" if args.original_dir is not None else "attacked"
        images = originals if role == "original" else attacked
        annotation_rows: List[AnnotationRow] = []
        total = len(rel_paths)
        for index, rel_path in enumerate(rel_paths, start=1):
            print(f"[{index:>4}/{total}] {rel_path}", flush=True)
            annotation = annotate_or_load(
                cache,
                cache_path,
                annotator,
                role,
                rel_path,
                images[rel_path],
                args.overwrite_cache,
                args.retry_failed_cache,
            )
            annotation_rows.append(
                AnnotationRow(
                    image_id=f"{index:06d}",
                    rel_path=rel_path,
                    role=role,
                    caption=annotation.caption,
                    visible_smile=annotation.visible_smile,
                    eyeglasses=annotation.eyeglasses,
                    head_pose=annotation.head_pose,
                    headwear=annotation.headwear,
                    background_type=annotation.background_type,
                    caption_status=annotation.caption_status,
                    caption_violations=annotation.caption_violations,
                    error=annotation.error,
                )
            )

        annotations_csv = args.output_dir / "image_annotations.csv"
        summary_json = args.output_dir / "summary.json"
        write_annotations_csv(annotations_csv, annotation_rows)
        write_summary(
            summary_json,
            {
                "mode": "description_only",
                "prompt_version": PROMPT_VERSION,
                "role": role,
                "source_dir": str(args.original_dir if role == "original" else args.attacked_dir),
                "total_images": len(annotation_rows),
                "failed_annotations": sum(1 for row in annotation_rows if row.error),
                "caption_status_counts": count_values(row.caption_status for row in annotation_rows),
                "bertscore_status": "not_applicable_description_only",
                "inference_configuration": annotator.inference_configuration,
            },
        )
        stability_outputs: Optional[Dict[str, Path]] = None
        if args.greedy_stability_runs:
            stability_outputs = run_greedy_stability_audit(
                annotator,
                [(role, images)],
                rel_paths,
                args.output_dir,
                args.greedy_stability_runs,
            )
        print(f"\nSaved annotations: {cache_path}")
        print(f"Saved image descriptions: {annotations_csv}")
        print(f"Saved summary: {summary_json}")
        if stability_outputs:
            print(f"Saved greedy stability CSV: {stability_outputs['csv']}")
            print(f"Saved greedy stability summary: {stability_outputs['summary']}")
        return 0

    rows: List[PairMetrics] = []
    total = len(rel_paths)
    for index, rel_path in enumerate(rel_paths, start=1):
        print(f"[{index:>4}/{total}] {rel_path}", flush=True)
        original = annotate_or_load(
            cache,
            cache_path,
            annotator,
            "original",
            rel_path,
            originals[rel_path],
            args.overwrite_cache,
            args.retry_failed_cache,
        )
        attacked_annotation = annotate_or_load(
            cache,
            cache_path,
            annotator,
            "attacked",
            rel_path,
            attacked[rel_path],
            args.overwrite_cache,
            args.retry_failed_cache,
        )
        rows.append(
            score_pair(
                pair_id=f"{index:06d}",
                rel_path=rel_path,
                original=original,
                attacked=attacked_annotation,
                min_evaluable_attrs=args.min_evaluable_attrs,
            )
        )

    bertscore_status = "skipped_by_user"
    if not args.skip_bertscore:
        bertscore_status = compute_bertscore(rows, args.bertscore_lang, args.bertscore_model)

    pairs_csv = args.output_dir / "pairs.csv"
    summary_json = args.output_dir / "summary.json"
    write_pairs_csv(pairs_csv, rows)
    write_summary(
        summary_json,
        build_summary(
            rows,
            bertscore_status,
            bertscore_configuration(args.bertscore_lang, args.bertscore_model),
            annotator.inference_configuration,
        ),
    )

    stability_outputs: Optional[Dict[str, Path]] = None
    if args.greedy_stability_runs:
        stability_outputs = run_greedy_stability_audit(
            annotator,
            [("original", originals), ("attacked", attacked)],
            rel_paths,
            args.output_dir,
            args.greedy_stability_runs,
        )

    print(f"\nSaved annotations: {cache_path}")
    print(f"Saved pair metrics: {pairs_csv}")
    print(f"Saved summary: {summary_json}")
    if stability_outputs:
        print(f"Saved greedy stability CSV: {stability_outputs['csv']}")
        print(f"Saved greedy stability summary: {stability_outputs['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
