#!/usr/bin/env python3
"""Precision / recall of the scorer against hand labels (Q4).

  # score and evaluate in one go
  python evaluate_pipeline.py --videos data/synthetic --truth data/synthetic/ground_truth.json

  # evaluate existing scores
  python evaluate_pipeline.py --scores out.json --truth labels.csv

  # make a blank labelling sheet for your own clips, fill it in, then evaluate
  python evaluate_pipeline.py --make-template out.json --template-out labels.csv

The positive class is the defect the assignment cares about ("precision/recall on bad
videos"): a 1-5 dimension is "bad" at 2 or below, egocentric framing is bad at 0, duplicate
risk is positive at 0.5 or above. Tiers get a 3x3 confusion matrix, reject-detection P/R and
auto-accept precision.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from src import config
from src.cli import (
    add_decode_option,
    add_download_options,
    exits_cleanly,
    load_records,
    prepare_output,
    read_json,
    write_json,
)
from src.config import get_settings
from src.schemas.scoring_schema import DIMENSIONS, UNIT_DIMENSIONS, Scores
from src.scorer.prompts import PROMPT_VERSION, VERSIONS
from src.scorer.provider_errors import VLMError
from src.scorer.quality_scorer import (
    QualityScorer,
    composite_score,
    hard_gate_failures,
    iter_inputs,
    tier_confidence,
)

TIERS = ("usable", "borderline", "reject")

Pair = tuple[str, dict[str, Any], dict[str, Any]]  # clip name, scored record, hand labels


def clip_name(record: dict[str, Any]) -> str:
    return Path((record.get("meta") or {}).get("source") or record.get("source", "")).name


def is_bad(dim: str, value: float) -> bool:
    if dim == "egocentric_framing":
        return value < 0.5
    if dim == "duplicate_risk":
        return value >= 0.5
    return value <= config.BAD_MAX_1_TO_5


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p is not None and r is not None and p + r else None
    return {"precision": _rounded(p), "recall": _rounded(r), "f1": _rounded(f)}


def _check_truth(path: Path, name: str, entry: Any) -> None:
    if not isinstance(entry, dict) or not isinstance(entry.get("scores"), dict):
        raise ValueError(f"{path}: {name}: expected an object with tier and scores")  # noqa: TRY004 - bad input file
    if entry.get("tier") not in TIERS:
        raise ValueError(
            f"{path}: {name}: tier must be one of {', '.join(TIERS)}, got {entry.get('tier')!r}"
        )


def _load_truth_csv(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            tier = (row.get("tier") or "").strip()
            if not row.get("file") or not tier:
                continue
            name = Path(row["file"]).name
            try:
                scores = {d: float(row[d]) for d in DIMENSIONS if (row.get(d) or "").strip()}
            except ValueError as exc:
                raise ValueError(f"{path}: {name}: {exc}") from exc
            out[name] = {"tier": tier, "scores": scores}
    return out


def load_truth(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        truth = _load_truth_csv(path)
    else:
        data = read_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected an object keyed by file name")
        truth = {Path(k).name: v for k, v in data.items()}
    for name, entry in truth.items():
        _check_truth(path, name, entry)
    return truth


def _failure_type(predicted_bad: bool, truly_bad: bool) -> str:
    if predicted_bad and not truly_bad:
        return "false_positive"
    if truly_bad and not predicted_bad:
        return "false_negative"
    return "off_by_2+"


def _dimension_metrics(dim: str, pairs: list[Pair]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Confusion counts and agreement for one dimension, plus the clips it got wrong."""
    tp = fp = fn = tn = 0
    exact = within1 = n = 0
    abs_err: list[float] = []
    failures = []
    for name, record, labels in pairs:
        if dim not in labels["scores"]:
            continue
        n += 1
        pred = record["scores"][dim]["score"]
        true = labels["scores"][dim]
        pred_bad, true_bad = is_bad(dim, pred), is_bad(dim, true)
        tp += pred_bad and true_bad
        fp += pred_bad and not true_bad
        fn += true_bad and not pred_bad
        tn += not pred_bad and not true_bad
        if dim in UNIT_DIMENSIONS:
            exact += (pred >= 0.5) == (true >= 0.5)
        else:
            exact += pred == true
            within1 += abs(pred - true) <= 1
            abs_err.append(abs(pred - true))
        if pred_bad != true_bad or (dim not in UNIT_DIMENSIONS and abs(pred - true) >= 2):
            failures.append(
                {
                    "clip": name,
                    "dimension": dim,
                    "predicted": pred,
                    "truth": true,
                    "type": _failure_type(pred_bad, true_bad),
                    "reasoning": record["scores"][dim]["reasoning"],
                }
            )
    entry = {
        "n": n,
        "positives_in_truth": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        **_prf(tp, fp, fn),
        "exact_match": round(exact / n, 3) if n else None,
    }
    if dim not in UNIT_DIMENSIONS and n:
        entry["within_1"] = round(within1 / n, 3)
        entry["mae"] = round(sum(abs_err) / n, 3)
    return entry, failures


def _tier_metrics(pairs: list[Pair]) -> dict[str, Any]:
    matrix: dict[str, Counter[str]] = {t: Counter() for t in TIERS}
    for _, record, labels in pairs:
        matrix[labels["tier"]][record["tier"]] += 1
    auto = sum(matrix[t]["usable"] for t in TIERS)
    return {
        "accuracy": round(sum(matrix[t][t] for t in TIERS) / len(pairs), 3),
        "confusion_matrix_truth_rows": {t: {p: matrix[t][p] for p in TIERS} for t in TIERS},
        "reject_detection": _prf(
            matrix["reject"]["reject"],
            sum(matrix[t]["reject"] for t in TIERS if t != "reject"),
            sum(matrix["reject"][t] for t in TIERS if t != "reject"),
        ),
        "auto_accept_precision": round(matrix["usable"]["usable"] / auto, 3) if auto else None,
        "auto_accepted": auto,
        "sent_to_review": sum(matrix[t]["borderline"] for t in TIERS),
    }


def evaluate(scored: list[dict[str, Any]], truth: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairs = [
        (clip_name(record), record, truth[clip_name(record)])
        for record in scored
        if clip_name(record) in truth and "error" not in record
    ]
    if not pairs:
        raise ValueError("no scored videos match the ground truth file names")

    per_dim, failures = {}, []
    for dim in DIMENSIONS:
        per_dim[dim], dim_failures = _dimension_metrics(dim, pairs)
        failures.extend(dim_failures)
    providers = Counter(
        f"{r['meta']['provider']}:{r['meta']['model']}" for _, r, _ in pairs if r.get("meta")
    )
    return {
        "evaluated_clips": len(pairs),
        "scorer": dict(providers),
        "positive_class": "defect (1-5 dims <= 2; framing = 0; duplicate >= 0.5)",
        "per_dimension": per_dim,
        "tier": _tier_metrics(pairs),
        "tier_errors": [
            {
                "clip": name,
                "predicted": record["tier"],
                "truth": labels["tier"],
                "composite": record.get("meta", {}).get("composite_score"),
                "hard_gates_failed": record.get("meta", {}).get("hard_gates_failed"),
            }
            for name, record, labels in pairs
            if record["tier"] != labels["tier"]
        ],
        "failures": failures,
    }


def threshold_sweep(
    scored: list[dict[str, Any]], truth: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Q5 calibration: auto-accept precision and review share for each threshold pair.

    Uses the stored dimension scores and confidences, so no model calls are repeated. Pick the
    loosest pair whose auto-accept precision meets the target (e.g. >= 0.95) on real labels.
    """
    rows = []
    pairs = [
        (record, truth[clip_name(record)])
        for record in scored
        if "error" not in record and clip_name(record) in truth
    ]
    for comp_t in (3.25, 3.5, 3.75, 4.0, 4.25):
        for conf_t in (0.5, 0.6, 0.7, 0.8, 0.9):
            acc = tp = rev = 0
            for record, labels in pairs:
                scores = Scores.model_validate(record["scores"])
                comp = composite_score(scores)
                if hard_gate_failures(scores) or comp < config.BORDERLINE_MIN_COMPOSITE:
                    continue
                disagreements = len((record.get("meta") or {}).get("cv_disagreements") or [])
                if comp >= comp_t and tier_confidence(scores, comp, disagreements) >= conf_t:
                    acc += 1
                    tp += labels["tier"] == "usable"
                else:
                    rev += 1
            rows.append(
                {
                    "composite_min": comp_t,
                    "confidence_min": conf_t,
                    "auto_accepted": acc,
                    "auto_accept_precision": round(tp / acc, 3) if acc else None,
                    "review_share": round(rev / len(pairs), 3) if pairs else None,
                }
            )
    return rows


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def to_markdown(rep: dict[str, Any]) -> str:
    lines = [
        (
            f"Scorer: {', '.join(rep['scorer'])}; clips: {rep['evaluated_clips']}; "
            f"positive class: {rep['positive_class']}"
        ),
        "",
        (
            "| Dimension | Defects in truth | TP | FP | FN "
            "| Precision | Recall | F1 | Exact | Within 1 |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d, e in rep["per_dimension"].items():
        lines.append(
            f"| {d} | {e['positives_in_truth']} | {e['tp']} | {e['fp']} | {e['fn']} | "
            f"{fmt(e['precision'])} | {fmt(e['recall'])} | {fmt(e['f1'])} | "
            f"{fmt(e['exact_match'])} | {fmt(e.get('within_1'))} |"
        )
    t = rep["tier"]
    rejects = t["reject_detection"]
    lines += [
        "",
        (
            f"Tier accuracy {fmt(t['accuracy'])}; "
            f"reject detection P {fmt(rejects['precision'])} R {fmt(rejects['recall'])}; "
            f"auto-accept precision {fmt(t['auto_accept_precision'])} "
            f"({t['auto_accepted']} auto-accepted, {t['sent_to_review']} sent to review)"
        ),
        "",
        "| Truth \\ Predicted | usable | borderline | reject |",
        "|---|---|---|---|",
    ]
    for tr, row in t["confusion_matrix_truth_rows"].items():
        lines.append(f"| {tr} | {row['usable']} | {row['borderline']} | {row['reject']} |")
    return "\n".join(lines) + "\n"


def make_template(scores_path: Path, out: Path) -> None:
    records = load_records(scores_path)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", *DIMENSIONS, "tier", "notes"])
        for r in records:
            w.writerow([clip_name(r), *[""] * len(DIMENSIONS), "", ""])
    print(
        f"wrote {out}: fill one row per clip without looking at the model's scores", file=sys.stderr
    )


@exits_cleanly
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="precision/recall of the scorer against hand labels"
    )
    parser.add_argument(
        "--truth", help="ground_truth.json (synthetic) or labels.csv (your own clips)"
    )
    parser.add_argument("--scores", help="JSON from score_videos.py")
    parser.add_argument(
        "--videos", nargs="+", help="score these first (same inputs as score_videos.py)"
    )
    parser.add_argument("--provider", choices=["gemini", "openai", "vllm", "offline"])
    parser.add_argument("--model")
    parser.add_argument("--prompt-version", default=PROMPT_VERSION, choices=sorted(VERSIONS))
    parser.add_argument("--workers", type=int, default=4)
    add_download_options(parser)
    add_decode_option(parser)
    parser.add_argument("--scores-out", help="where to save scores when --videos is used")
    parser.add_argument("--out", default="sample_outputs/precision_recall_report.json")
    parser.add_argument("--markdown", help="also write the tables as Markdown")
    parser.add_argument("--make-template", help="scores JSON to build a blank labels.csv from")
    parser.add_argument("--template-out", default="labels.csv")
    args = parser.parse_args(argv)

    if args.make_template:
        prepare_output(args.template_out)
        make_template(Path(args.make_template), Path(args.template_out))
        return 0
    if not args.truth:
        parser.error("--truth is required")
    if not args.videos and not args.scores:
        parser.error("give --scores or --videos")

    # Check everything cheap before --videos spends money on model calls.
    prepare_output(args.out, args.markdown, args.scores_out)
    truth = load_truth(Path(args.truth))
    if args.videos:
        settings = get_settings(args.provider, args.model)
        scorer = QualityScorer(
            settings,
            args.prompt_version,
            workers=args.workers,
            download_workers=args.download_workers,
            keep_cache=args.keep_cache,
            decode_timeout=args.decode_timeout,
        )
        scored = scorer.score_batch(iter_inputs(args.videos, settings))
        if args.scores_out:
            write_json(args.scores_out, scored)
        if scorer.auth_failure or scorer.input_failure:
            raise VLMError(scorer.auth_failure or scorer.input_failure)
    else:
        scored = load_records(args.scores)

    report = evaluate(scored, truth)
    report["threshold_sweep"] = threshold_sweep(scored, truth)
    write_json(args.out, report)
    md = to_markdown(report)
    if args.markdown:
        Path(args.markdown).write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
