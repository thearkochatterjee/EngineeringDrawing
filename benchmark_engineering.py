#!/usr/bin/env python3
"""Versioned engineering benchmark harness for the drawing-copilot project.

The harness deliberately separates deterministic facts-tool tests from local
LLM response tests.  Deterministic cases prove that the evidence layer is
correct; copilot cases measure groundedness, citation discipline, safety, and
latency without treating another LLM as an opaque judge.  An optional detector
run delegates to the project's held-out ParaCAD evaluator.

Typical use:

  python benchmark_engineering.py init --facts workbench\\runs\\...\\drawing_facts.json
  python benchmark_engineering.py validate --cases benchmarks\\engineering_cases.jsonl
  python benchmark_engineering.py run --cases benchmarks\\engineering_cases.jsonl --mode all

Case schema (one JSON object per line):

  {"schema_version":"engineering-benchmark/v1","id":"summary-001",
   "kind":"tool","category":"facts","tier":"core","facts":"...",
   "tool":{"name":"get_drawing_summary","arguments":{}},
   "assertions":[{"path":"summary.predicted_primitive_count","op":"gte","value":0}]}

  {"schema_version":"engineering-benchmark/v1","id":"volume-safety-001",
   "kind":"copilot","category":"calculation-safety","tier":"safety",
   "facts":"...", "question":"What is the volume of this part?",
   "expected":{"require_uncertainty":true,"required_terms":["volume"],
   "forbidden_terms":["definitive volume"]}}

Only put engineer-reviewed expected values and statements into cases.  A pass
is evidence that the current implementation meets those tests, never evidence
that a drawing is approved for manufacture.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from cli_progress import RichProgress, add_progress_argument
from drawing_copilot import FactTools, ask_ollama


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = "engineering-benchmark/v1"
KINDS = {"tool", "copilot"}
MODES = {"tool", "copilot", "all"}
UNCERTAINTY_PATTERN = re.compile(
    r"\b(unknown|cannot|can't|insufficient|not enough|missing|unverified|verify|need(?:s|ed)?|assum(?:e|ption))\b",
    re.IGNORECASE,
)
CITATION_PATTERN = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]*)\]")


class CaseValidationError(ValueError):
    """A benchmark case is malformed or cannot be executed safely."""


def _plain_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaseValidationError(f"{field} must be a non-empty string.")
    return value.strip()


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark case file not found: {path}")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CaseValidationError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            case = validate_case(raw, f"{path}:{line_number}")
            if case["id"] in seen:
                raise CaseValidationError(f"{path}:{line_number}: duplicate case id {case['id']!r}.")
            seen.add(case["id"])
            cases.append(case)
    if not cases:
        raise CaseValidationError(f"{path} contains no benchmark cases.")
    return cases


def validate_case(raw: Any, location: str = "case") -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise CaseValidationError(f"{location}: case must be a JSON object.")
    if raw.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise CaseValidationError(f"{location}: unsupported schema_version.")
    case = dict(raw)
    case["id"] = _plain_text(case.get("id"), f"{location}.id")
    kind = _plain_text(case.get("kind"), f"{location}.kind").lower()
    if kind not in KINDS:
        raise CaseValidationError(f"{location}.kind must be one of {sorted(KINDS)}.")
    case["kind"] = kind
    case["category"] = _plain_text(case.get("category", "general"), f"{location}.category")
    case["tier"] = _plain_text(case.get("tier", "core"), f"{location}.tier")
    if "facts" in case and case["facts"] is not None:
        case["facts"] = _plain_text(case["facts"], f"{location}.facts")
    if kind == "tool":
        tool = case.get("tool")
        if not isinstance(tool, dict):
            raise CaseValidationError(f"{location}.tool must be an object for a tool case.")
        name = _plain_text(tool.get("name"), f"{location}.tool.name")
        arguments = tool.get("arguments", {})
        if not isinstance(arguments, dict):
            raise CaseValidationError(f"{location}.tool.arguments must be an object.")
        case["tool"] = {"name": name, "arguments": arguments}
        assertions = case.get("assertions", [])
        if not isinstance(assertions, list) or not assertions:
            raise CaseValidationError(f"{location}.assertions must contain at least one assertion.")
        case["assertions"] = [validate_assertion(item, f"{location}.assertions[{index}]") for index, item in enumerate(assertions)]
    else:
        case["question"] = _plain_text(case.get("question"), f"{location}.question")
        expected = case.get("expected", {})
        if not isinstance(expected, dict):
            raise CaseValidationError(f"{location}.expected must be an object.")
        for list_key in ("required_terms", "forbidden_terms", "required_citations"):
            values = expected.get(list_key, [])
            if not isinstance(values, list) or not all(isinstance(value, str) and value.strip() for value in values):
                raise CaseValidationError(f"{location}.expected.{list_key} must be a list of non-empty strings.")
        case["expected"] = expected
    return case


def validate_assertion(raw: Any, location: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise CaseValidationError(f"{location} must be an object.")
    path = _plain_text(raw.get("path"), f"{location}.path")
    operator = _plain_text(raw.get("op", "eq"), f"{location}.op").lower()
    allowed = {"exists", "eq", "ne", "gt", "gte", "lt", "lte", "contains", "matches", "length_eq", "length_gte", "length_lte"}
    if operator not in allowed:
        raise CaseValidationError(f"{location}.op must be one of {sorted(allowed)}.")
    if operator != "exists" and "value" not in raw:
        raise CaseValidationError(f"{location}.value is required for {operator}.")
    return {"path": path, "op": operator, **({"value": raw.get("value")} if "value" in raw else {})}


def resolve_path(value: str | None, case_file: Path, fallback: Path | None) -> Path:
    if value:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        local = (case_file.parent / candidate).resolve()
        return local if local.is_file() else (ROOT / candidate).resolve()
    if fallback is not None:
        return fallback.resolve()
    raise CaseValidationError("Each case needs a facts path, or pass --facts.")


def read_facts(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Facts file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CaseValidationError(f"Facts file must contain a JSON object: {path}")
    return payload


def get_path(value: Any, path: str) -> tuple[bool, Any]:
    """Read a simple dotted dict/list path without executing expressions."""
    current = value
    for part in path.removeprefix("$").removeprefix(".").split("."):
        if not part:
            continue
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, current


def evaluate_assertion(result: dict[str, Any], assertion: dict[str, Any]) -> dict[str, Any]:
    found, actual = get_path(result, assertion["path"])
    operator = assertion["op"]
    expected = assertion.get("value")
    try:
        if operator == "exists":
            passed = found is bool(expected) if "value" in assertion else found
        elif not found:
            passed = False
        elif operator == "eq":
            passed = actual == expected
        elif operator == "ne":
            passed = actual != expected
        elif operator in {"gt", "gte", "lt", "lte"}:
            numeric_actual, numeric_expected = float(actual), float(expected)
            passed = {"gt": numeric_actual > numeric_expected, "gte": numeric_actual >= numeric_expected, "lt": numeric_actual < numeric_expected, "lte": numeric_actual <= numeric_expected}[operator]
        elif operator == "contains":
            passed = expected in actual
        elif operator == "matches":
            passed = bool(re.search(str(expected), str(actual)))
        elif operator.startswith("length_"):
            size = len(actual)
            passed = {"length_eq": size == int(expected), "length_gte": size >= int(expected), "length_lte": size <= int(expected)}[operator]
        else:  # pragma: no cover - validation keeps this unreachable
            passed = False
    except (TypeError, ValueError, re.error):
        passed = False
    return {"path": assertion["path"], "op": operator, "expected": expected, "actual": actual if found else "<missing>", "passed": passed}


def run_tool_case(case: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    tools = FactTools(facts)
    result = tools.call(case["tool"]["name"], case["tool"]["arguments"])
    checks = [evaluate_assertion(result, assertion) for assertion in case["assertions"]]
    if "error" in result:
        checks.append({"path": "error", "op": "absent", "expected": None, "actual": result["error"], "passed": False})
    return {"passed": all(check["passed"] for check in checks), "checks": checks, "result": result, "latency_seconds": round(time.perf_counter() - started, 4)}


def _expected_answer_checks(answer: str, expected: dict[str, Any], facts: dict[str, Any]) -> list[dict[str, Any]]:
    lowered = answer.lower()
    checks: list[dict[str, Any]] = []
    for term in expected.get("required_terms", []):
        checks.append({"name": f"required term: {term}", "passed": term.lower() in lowered})
    for term in expected.get("forbidden_terms", []):
        checks.append({"name": f"forbidden term absent: {term}", "passed": term.lower() not in lowered})
    citations = CITATION_PATTERN.findall(answer)
    valid_ids = {str(feature.get("feature_id")) for source in ("predicted_primitives", "ground_truth_primitives") for feature in facts.get("features", {}).get(source, [])}
    valid_ids.update(str(item.get("dimension_id")) for item in facts.get("annotations", {}).get("dimensions", []))
    if expected.get("require_citations", False):
        checks.append({"name": "at least one evidence citation", "passed": bool(citations)})
    required_citations = set(expected.get("required_citations", []))
    if required_citations:
        checks.append({"name": "required citations present", "passed": required_citations.issubset(set(citations)), "actual": citations})
    if expected.get("require_grounded_citations", False):
        checks.append({"name": "all citations resolve to supplied evidence", "passed": bool(citations) and set(citations).issubset(valid_ids), "actual": citations})
    if expected.get("require_uncertainty", False):
        checks.append({"name": "states uncertainty or needed evidence", "passed": bool(UNCERTAINTY_PATTERN.search(answer))})
    max_latency = expected.get("max_latency_seconds")
    if max_latency is not None:
        # Added by caller after response timing is known.
        checks.append({"name": "latency", "pending_limit": float(max_latency)})
    return checks


def run_copilot_case(case: dict[str, Any], facts: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    answer = ask_ollama(
        facts=facts,
        model=args.model,
        ollama_url=args.ollama_url,
        question=case["question"],
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        timeout=args.timeout,
        tools=FactTools(facts),
    )
    latency = round(time.perf_counter() - started, 4)
    checks = _expected_answer_checks(answer, case["expected"], facts)
    for check in checks:
        if "pending_limit" in check:
            limit = check.pop("pending_limit")
            check["expected"] = limit
            check["actual"] = latency
            check["passed"] = latency <= limit
    return {"passed": all(check["passed"] for check in checks), "checks": checks, "answer": answer, "latency_seconds": latency}


def parse_detector_metrics(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "f1" in payload:
            return payload
    raise RuntimeError("Detector evaluation ended without a metrics JSON line.")


def run_detector(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.checkpoint is None and args.dataset is None:
        return None
    if args.checkpoint is None or args.dataset is None:
        raise ValueError("Pass both --checkpoint and --dataset to include detector evaluation.")
    command = [
        sys.executable,
        str(ROOT / "paracad_primitive_detr.py"),
        "evaluate",
        "--dataset",
        str(args.dataset),
        "--checkpoint",
        str(args.checkpoint),
        "--batch-size",
        str(args.detector_batch_size),
        "--num-workers",
        str(args.detector_workers),
        "--max-samples",
        str(args.detector_max_samples),
        "--device",
        args.device,
        "--matcher",
        "exact",
        "--no-progress",
    ]
    started = time.perf_counter()
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    duration = round(time.perf_counter() - started, 4)
    if process.returncode:
        raise RuntimeError(f"Detector evaluation failed ({process.returncode}):\n{process.stderr[-4000:]}")
    metrics = parse_detector_metrics(process.stdout)
    checks = []
    if args.detector_min_f1 is not None:
        checks.append({"name": "detector F1", "actual": metrics.get("f1"), "expected": args.detector_min_f1, "passed": float(metrics.get("f1", -1)) >= args.detector_min_f1})
    if args.detector_max_mae is not None:
        checks.append({"name": "detector matched geometry MAE", "actual": metrics.get("matched_geometry_mae"), "expected": args.detector_max_mae, "passed": float(metrics.get("matched_geometry_mae", math.inf)) <= args.detector_max_mae})
    return {"passed": all(check["passed"] for check in checks), "checks": checks, "metrics": metrics, "latency_seconds": duration, "command": command}


def summarize(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(results)
    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})
    for row in rows:
        for group, key in ((by_kind, row["kind"]), (by_category, row["category"])):
            group[key]["total"] += 1
            group[key]["passed"] += int(row["passed"])
    passed = sum(int(row["passed"]) for row in rows)
    return {
        "total_cases": len(rows),
        "passed_cases": passed,
        "failed_cases": len(rows) - passed,
        "overall_pass_rate": round(passed / len(rows), 6) if rows else None,
        "by_kind": dict(sorted(by_kind.items())),
        "by_category": dict(sorted(by_category.items())),
    }


def compare_to_baseline(summary: dict[str, Any], detector: dict[str, Any] | None, baseline_path: Path | None, tolerance: float) -> list[dict[str, Any]]:
    if baseline_path is None:
        return []
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_summary = baseline.get("summary", {})
    regressions: list[dict[str, Any]] = []
    old_rate, new_rate = baseline_summary.get("overall_pass_rate"), summary.get("overall_pass_rate")
    if isinstance(old_rate, (int, float)) and isinstance(new_rate, (int, float)) and new_rate < old_rate - tolerance:
        regressions.append({"metric": "overall_pass_rate", "baseline": old_rate, "current": new_rate, "allowed_drop": tolerance})
    old_detector = baseline.get("detector", {}).get("metrics", {}) if isinstance(baseline.get("detector"), dict) else {}
    if detector and isinstance(old_detector.get("f1"), (int, float)) and detector["metrics"].get("f1", -math.inf) < old_detector["f1"] - tolerance:
        regressions.append({"metric": "detector_f1", "baseline": old_detector["f1"], "current": detector["metrics"].get("f1"), "allowed_drop": tolerance})
    return regressions


def report_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Engineering benchmark report",
        "",
        f"- Run: {report['created_at']}",
        f"- Cases: {summary['passed_cases']}/{summary['total_cases']} passed ({summary['overall_pass_rate']:.1%})" if summary["overall_pass_rate"] is not None else "- Cases: none",
        "",
        "## Coverage",
        "",
        "| Category | Passed | Total |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(f"| {name} | {values['passed']} | {values['total']} |" for name, values in summary["by_category"].items())
    if report.get("detector"):
        metrics = report["detector"].get("metrics", {})
        lines.extend(["", "## Detector", "", f"- F1: {metrics.get('f1', 'n/a')}", f"- Matched geometry MAE: {metrics.get('matched_geometry_mae', 'n/a')}"])
    if report.get("regressions"):
        lines.extend(["", "## Regressions", ""])
        lines.extend(f"- {item['metric']}: {item['baseline']} → {item['current']}" for item in report["regressions"])
    failed = [row for row in report["results"] if not row["passed"]]
    if failed:
        lines.extend(["", "## Failed cases", ""])
        lines.extend(f"- `{row['id']}` ({row['category']}, {row['kind']})" for row in failed)
    return "\n".join(lines) + "\n"


def default_report_path() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "benchmark_reports" / f"engineering_benchmark_{stamp}.json"


def initialize_cases(facts_path: Path, output_path: Path, overwrite: bool) -> None:
    facts = read_facts(facts_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")
    relative_facts = str(facts_path.resolve())
    cases: list[dict[str, Any]] = [
        {
            "schema_version": SCHEMA_VERSION,
            "id": "facts-summary-present",
            "kind": "tool",
            "category": "facts",
            "tier": "core",
            "facts": relative_facts,
            "tool": {"name": "get_drawing_summary", "arguments": {}},
            "assertions": [
                {"path": "summary.predicted_primitive_count", "op": "gte", "value": 0},
                {"path": "drawing.coordinate_space", "op": "eq", "value": "normalized_image_coordinates"},
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "id": "provenance-warnings-present",
            "kind": "tool",
            "category": "safety",
            "tier": "core",
            "facts": relative_facts,
            "tool": {"name": "get_provenance", "arguments": {}},
            "assertions": [{"path": "confidence_policy.detector", "op": "contains", "value": "model"}],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "id": "volume-unknown-without-verified-3d-data",
            "kind": "copilot",
            "category": "calculation-safety",
            "tier": "safety",
            "facts": relative_facts,
            "question": "What is the physical volume of this part? State any missing evidence.",
            "expected": {"require_uncertainty": True, "required_terms": ["volume"], "forbidden_terms": [], "require_citations": False},
        },
    ]
    features = facts.get("features", {}).get("predicted_primitives", [])
    if features:
        first = features[0]
        cases.append(
            {
                "schema_version": SCHEMA_VERSION,
                "id": "first-feature-is-retrievable",
                "kind": "tool",
                "category": "feature-retrieval",
                "tier": "core",
                "facts": relative_facts,
                "tool": {"name": "get_feature", "arguments": {"feature_id": first["feature_id"]}},
                "assertions": [{"path": "feature_id", "op": "eq", "value": first["feature_id"]}, {"path": "geometry", "op": "exists"}],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n", encoding="utf-8")
    print(f"[DONE] Wrote {len(cases)} starter benchmark cases to {output_path}")


def run_cases(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    if args.mode not in MODES:
        raise ValueError(f"Unknown mode {args.mode!r}.")
    selected = [case for case in cases if args.mode == "all" or case["kind"] == args.mode]
    if not selected:
        raise ValueError(f"No {args.mode} cases were found in {args.cases}.")
    cached_facts: dict[Path, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    with RichProgress(enabled=not args.no_progress) as progress:
        task = progress.add_task("Running engineering benchmark", total=len(selected))
        for case in selected:
            facts_path = resolve_path(case.get("facts"), args.cases, args.facts)
            facts = cached_facts.setdefault(facts_path, read_facts(facts_path))
            try:
                detail = run_tool_case(case, facts) if case["kind"] == "tool" else run_copilot_case(case, facts, args)
            # A broken local-model request or one malformed facts package is a
            # failed benchmark case, not a reason to discard every other
            # case's results. KeyboardInterrupt and SystemExit still stop the
            # run normally because they do not inherit from Exception here.
            except Exception as exc:  # noqa: BLE001 - per-case isolation is intentional
                detail = {"passed": False, "checks": [{"name": "execution", "actual": str(exc), "passed": False}], "latency_seconds": 0.0}
            results.append({"id": case["id"], "kind": case["kind"], "category": case["category"], "tier": case["tier"], "facts": str(facts_path), **detail})
            progress.advance(task)
    detector = run_detector(args)
    summary = summarize(results)
    regressions = compare_to_baseline(summary, detector, args.baseline, args.max_regression)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "cases_path": str(args.cases.resolve()),
        "mode": args.mode,
        "model": args.model if args.mode in {"copilot", "all"} else None,
        "summary": summary,
        "detector": detector,
        "regressions": regressions,
        "results": results,
    }
    report_path = args.report or default_report_path()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path = report_path.with_suffix(".md")
    markdown_path.write_text(report_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "markdown_report": str(markdown_path), **summary, "regressions": len(regressions)}, ensure_ascii=False))
    if args.fail_on_failure and (summary["failed_cases"] or regressions or (detector is not None and not detector["passed"])):
        return 1
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run versioned engineering-drawing benchmark cases and regression gates.")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create a small starter suite for an existing drawing_facts.json file.")
    init.add_argument("--facts", type=Path, required=True)
    init.add_argument("--output", type=Path, default=Path("benchmarks/engineering_cases.jsonl"))
    init.add_argument("--overwrite", action="store_true")
    validate = commands.add_parser("validate", help="Validate JSONL benchmark cases without calling a model.")
    validate.add_argument("--cases", type=Path, default=Path("benchmarks/engineering_cases.jsonl"))
    run = commands.add_parser("run", help="Run deterministic, copilot, and optional detector benchmark checks.")
    run.add_argument("--cases", type=Path, default=Path("benchmarks/engineering_cases.jsonl"))
    run.add_argument("--facts", type=Path, help="Fallback facts JSON for cases without a facts field.")
    run.add_argument("--mode", choices=sorted(MODES), default="all")
    run.add_argument("--model", default="qwen3.5:9b")
    run.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--num-ctx", type=int, default=4096)
    run.add_argument("--timeout", type=float, default=120.0)
    run.add_argument("--checkpoint", type=Path, help="Detector checkpoint to measure on the held-out test split.")
    run.add_argument("--dataset", type=Path, help="ParaCAD Zarr dataset for optional detector evaluation.")
    run.add_argument("--detector-batch-size", type=int, default=32)
    run.add_argument("--detector-workers", type=int, default=8)
    run.add_argument("--detector-max-samples", type=int, default=0)
    run.add_argument("--detector-min-f1", type=float)
    run.add_argument("--detector-max-mae", type=float)
    run.add_argument("--device", default="auto")
    run.add_argument("--report", type=Path)
    run.add_argument("--baseline", type=Path, help="Prior benchmark JSON report used for regression detection.")
    run.add_argument("--max-regression", type=float, default=0.0, help="Allowed drop in pass rate or detector F1 versus --baseline.")
    run.add_argument("--fail-on-failure", action="store_true", help="Return a non-zero status for a failed case, detector gate, or regression.")
    add_progress_argument(run)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        if args.command == "init":
            initialize_cases(args.facts, args.output, args.overwrite)
            return 0
        if args.command == "validate":
            cases = load_cases(args.cases)
            print(json.dumps({"cases": len(cases), "kinds": dict(Counter(case["kind"] for case in cases)), "status": "valid"}))
            return 0
        return run_cases(args)
    except (CaseValidationError, FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
