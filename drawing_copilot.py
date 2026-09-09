#!/usr/bin/env python3
"""Ask a local Ollama model grounded questions about ``drawing_facts.json``.

The model receives a small drawing summary and may call deterministic tools for
features, dimensions, topology, and review checks. It can optionally save a
reviewed declarative workflow only when ``--allow-skill-write`` is supplied;
it never receives shell, arbitrary-file, or drawing-modification access.

Examples:
  python drawing_copilot.py --facts drawing_facts.json --model qwen3.5:9b \
      --question "Which hole patterns should I verify?"

  python drawing_copilot.py --facts drawing_facts.json --model qwen3.5:9b --interactive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cli_progress import RichProgress, add_progress_argument
from drawing_facts import feature_center
from engineering_knowledge import KnowledgeIndex
from engineering_skills import ALLOWED_FACT_TOOLS, SkillLibrary
from vision_evidence import build_vision_evidence


MAX_TOOL_RESULTS = 50
MAX_TOOL_TURNS = 8


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_drawing_summary",
            "description": "Get the detected primitive counts, confidence summary, available evidence, and review-finding count.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_features",
            "description": "List detected or evaluation-label primitives. Use feature_id values in later calls and cite those IDs in the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["line", "circle", "arc"], "description": "Optional primitive type."},
                    "source": {"type": "string", "enum": ["predicted", "ground_truth"], "description": "Defaults to predicted features."},
                    "min_confidence": {"type": "number", "description": "Optional detector-confidence floor for predicted features."},
                    "limit": {"type": "integer", "description": "Maximum result count, capped at 50."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_feature",
            "description": "Get exact stored geometry, confidence, and image-space metrics for one feature ID such as P87 or GT-Line0.",
            "parameters": {"type": "object", "properties": {"feature_id": {"type": "string"}}, "required": ["feature_id"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_holes",
            "description": "List circle features treated as hole candidates. These are geometric candidates, not confirmed drilled holes.",
            "parameters": {"type": "object", "properties": {"source": {"type": "string", "enum": ["predicted", "ground_truth"]}, "limit": {"type": "integer"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "measure_center_distance",
            "description": "Measure center-to-center distance of two stored features in normalized image and pixel coordinates. It is never a physical CAD dimension.",
            "parameters": {
                "type": "object",
                "properties": {"feature_id_a": {"type": "string"}, "feature_id_b": {"type": "string"}},
                "required": ["feature_id_a", "feature_id_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dimensions",
            "description": "List label dimensions (evaluation only) and OCR numeric candidates. OCR candidates are unverified and may not be tied to geometry.",
            "parameters": {
                "type": "object",
                "properties": {"dimension_type": {"type": "string"}, "source": {"type": "string", "enum": ["label", "ocr"]}, "limit": {"type": "integer"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_constraints",
            "description": "Get ParaCAD constraints, when the facts were built with ground-truth JSONL for evaluation. Unavailable for normal unlabeled drawings.",
            "parameters": {"type": "object", "properties": {"constraint_type": {"type": "string"}, "limit": {"type": "integer"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_symmetry",
            "description": "Return image-geometry symmetry proposals and the feature pairs supporting each proposal.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_repeated_hole_patterns",
            "description": "Return same-radius circle groups that may be repeated hole patterns.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_topology",
            "description": "Return inferred connectivity components, junctions, and closed-line-profile count from predicted primitive endpoints.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_rules",
            "description": "Return deterministic review findings such as near-duplicate detections and low confidence. Findings require engineer verification.",
            "parameters": {"type": "object", "properties": {"severity": {"type": "string", "enum": ["info", "review", "warning"]}, "limit": {"type": "integer"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_annotations",
            "description": "Return OCR text and title-block candidates. All OCR is unverified until checked by an engineer.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_provenance",
            "description": "Get evidence sources, confidence policy, warnings, and limits of this analysis.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_SCHEMAS.extend(
    [
        {
            "type": "function",
            "function": {
                "name": "search_engineering_knowledge",
                "description": "Search the optional approved local engineering-reference index. Cite returned [K#] chunks and do not treat a result as applicable until its revision and scope are verified.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "description": "Maximum results, capped at 10."}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_visual_evidence",
                "description": "List the original drawing, detector overlay, and focused crops attached to this vision-enabled conversation. Numeric geometry and feature IDs must still come from facts tools.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_skills",
                "description": "List built-in and user-approved engineering-review skills. Skills are inspectable, fixed plans over safe drawing fact tools.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_skill",
                "description": "Inspect a skill's instructions, declared inputs, and fixed tool plan before running it.",
                "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_skill",
                "description": "Run one inspected skill. It executes only its declared allowlisted facts-tool calls and returns the evidence for synthesis.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "inputs": {"type": "object", "description": "Values for this skill's declared inputs."}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_skill_draft",
                "description": "Create an in-session draft for a reusable engineering-review workflow. It may use only declared safe drawing facts tools; it does not write a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Lowercase hyphenated name, 3-64 characters."},
                        "description": {"type": "string"},
                        "instructions": {"type": "string"},
                        "input_schema": {"type": "object", "description": "Optional inputs keyed by lowercase name, each with description and required."},
                        "tool_plan": {"type": "array", "description": "One to eight steps, each with an allowlisted fact-tool name and JSON arguments. Use $input.name only for declared inputs."},
                    },
                    "required": ["name", "description", "instructions", "tool_plan"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "save_skill",
                "description": "Persist a reviewed in-session skill draft. This is enabled only when the user explicitly starts the copilot with --allow-skill-write.",
                "parameters": {"type": "object", "properties": {"draft_id": {"type": "string"}}, "required": ["draft_id"]},
            },
        },
    ]
)


def _limit(value: Any, default: int = 25) -> int:
    try:
        return max(1, min(MAX_TOOL_RESULTS, int(value)))
    except (TypeError, ValueError):
        return default


def _brief_feature(feature: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_id": feature["feature_id"],
        "type": feature["type"],
        "source": feature["source"],
        "confidence": feature["confidence"],
        "geometry": feature["geometry"],
        "metrics": feature["metrics"],
    }


class FactTools:
    """Allowlisted, deterministic lookups over one facts document."""

    def __init__(
        self,
        facts: dict[str, Any],
        skills_dir: Path = Path("engineering_skills"),
        allow_skill_write: bool = False,
        allow_skill_replace: bool = False,
        knowledge_index: KnowledgeIndex | None = None,
        vision_evidence: list[dict[str, str]] | None = None,
    ) -> None:
        self.facts = facts
        features = facts.get("features", {})
        self.predicted = list(features.get("predicted_primitives", []))
        self.ground_truth = list(features.get("ground_truth_primitives", []))
        self.by_id = {feature["feature_id"]: feature for feature in self.predicted + self.ground_truth}
        self.skills = SkillLibrary(skills_dir, allow_write=allow_skill_write, allow_replace=allow_skill_replace)
        self.knowledge_index = knowledge_index
        self.vision_evidence = vision_evidence or []

    def _features(self, source: str | None) -> list[dict[str, Any]]:
        return self.ground_truth if source == "ground_truth" else self.predicted

    def get_drawing_summary(self, **_: Any) -> dict[str, Any]:
        return {
            "drawing": self.facts.get("drawing", {}),
            "summary": self.facts.get("summary", {}),
            "available_evidence": {
                "predicted_primitives": len(self.predicted),
                "ground_truth_primitives": len(self.ground_truth),
                "dimensions": len(self.facts.get("annotations", {}).get("dimensions", [])),
                "constraints": len(self.facts.get("annotations", {}).get("constraints", [])),
                "ocr_annotations": len(self.facts.get("annotations", {}).get("ocr_text", [])),
            },
        }

    def list_features(self, kind: str | None = None, source: str | None = None, min_confidence: float | None = None, limit: int = 25, **_: Any) -> dict[str, Any]:
        rows = self._features(source)
        if kind:
            rows = [feature for feature in rows if feature["type"] == kind]
        if min_confidence is not None and source != "ground_truth":
            try:
                floor = float(min_confidence)
                rows = [feature for feature in rows if feature.get("confidence") is not None and feature["confidence"] >= floor]
            except (TypeError, ValueError):
                pass
        limited = rows[: _limit(limit)]
        return {"count": len(rows), "returned": len(limited), "features": [_brief_feature(feature) for feature in limited]}

    def get_feature(self, feature_id: str, **_: Any) -> dict[str, Any]:
        feature = self.by_id.get(feature_id)
        return _brief_feature(feature) if feature else {"error": f"Unknown feature_id {feature_id!r}. Use list_features first."}

    def list_holes(self, source: str | None = None, limit: int = 25, **_: Any) -> dict[str, Any]:
        rows = [feature for feature in self._features(source) if feature["type"] == "circle"]
        limited = rows[: _limit(limit)]
        return {
            "interpretation": "Circles are geometric hole candidates only; confirm hole callouts and manufacturing intent.",
            "count": len(rows),
            "holes": [_brief_feature(feature) for feature in limited],
        }

    def measure_center_distance(self, feature_id_a: str, feature_id_b: str, **_: Any) -> dict[str, Any]:
        first, second = self.by_id.get(feature_id_a), self.by_id.get(feature_id_b)
        if not first or not second:
            missing = [feature_id for feature_id, feature in ((feature_id_a, first), (feature_id_b, second)) if feature is None]
            return {"error": f"Unknown feature ID(s): {', '.join(missing)}"}
        center_a, center_b = feature_center(first), feature_center(second)
        normalized = ((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2) ** 0.5
        image = self.facts["drawing"]["image_size_pixels"]
        pixels = (((center_a[0] - center_b[0]) * image["width"]) ** 2 + ((center_a[1] - center_b[1]) * image["height"]) ** 2) ** 0.5
        return {
            "feature_ids": [feature_id_a, feature_id_b],
            "center_distance_normalized": round(normalized, 6),
            "center_distance_pixels_approx": round(pixels, 2),
            "warning": "Image-space measurement only. Physical units and CAD dimensions are unknown unless verified by a dimension annotation.",
        }

    def list_dimensions(self, dimension_type: str | None = None, source: str | None = None, limit: int = 25, **_: Any) -> dict[str, Any]:
        rows = self.facts.get("annotations", {}).get("dimensions", [])
        if dimension_type:
            rows = [row for row in rows if row.get("type") == dimension_type]
        if source:
            rows = [row for row in rows if row.get("source") == source]
        limited = rows[: _limit(limit)]
        return {"count": len(rows), "returned": len(limited), "dimensions": limited}

    def get_constraints(self, constraint_type: str | None = None, limit: int = 25, **_: Any) -> dict[str, Any]:
        rows = self.facts.get("annotations", {}).get("constraints", [])
        if constraint_type:
            rows = [row for row in rows if str(row.get("type", "")).lower() == constraint_type.lower()]
        limited = rows[: _limit(limit)]
        return {"count": len(rows), "returned": len(limited), "constraints": limited}

    def find_symmetry(self, **_: Any) -> dict[str, Any]:
        return {"proposals": self.facts.get("features", {}).get("symmetry_proposals", [])}

    def find_repeated_hole_patterns(self, **_: Any) -> dict[str, Any]:
        return {"patterns": self.facts.get("features", {}).get("repeated_hole_patterns", [])}

    def get_topology(self, **_: Any) -> dict[str, Any]:
        return self.facts.get("topology", {})

    def check_rules(self, severity: str | None = None, limit: int = 25, **_: Any) -> dict[str, Any]:
        rows = self.facts.get("checks", {}).get("findings", [])
        if severity:
            rows = [row for row in rows if row.get("severity") == severity]
        limited = rows[: _limit(limit)]
        return {"count": len(rows), "returned": len(limited), "findings": limited}

    def get_annotations(self, query: str | None = None, limit: int = 25, **_: Any) -> dict[str, Any]:
        rows = self.facts.get("annotations", {}).get("ocr_text", [])
        if query:
            query_lower = query.lower()
            rows = [row for row in rows if query_lower in str(row.get("text", "")).lower()]
        limited = rows[: _limit(limit)]
        return {
            "count": len(rows),
            "returned": len(limited),
            "ocr_text": limited,
            "title_block": self.facts.get("annotations", {}).get("title_block", {}),
            "warning": "OCR content is unverified.",
        }

    def get_provenance(self, **_: Any) -> dict[str, Any]:
        return self.facts.get("provenance", {})

    def search_engineering_knowledge(self, query: str, limit: int = 5, **_: Any) -> dict[str, Any]:
        if self.knowledge_index is None:
            return {"error": "No approved knowledge index is loaded. Build one with build_knowledge_index.py and pass --knowledge-index."}
        return self.knowledge_index.search(query, limit)

    def get_visual_evidence(self, **_: Any) -> dict[str, Any]:
        if not self.vision_evidence:
            return {"error": "No vision evidence is attached. Restart with --vision-crops auto (and optionally --vision-image)."}
        return {
            "count": len(self.vision_evidence),
            "evidence": [{"id": item["id"], "description": item["description"]} for item in self.vision_evidence],
            "warning": "Attached images provide visual verification context. Retrieve feature IDs, confidence, dimensions, and measurements from facts tools.",
        }

    def list_skills(self, **_: Any) -> dict[str, Any]:
        return self.skills.list()

    def get_skill(self, name: str, **_: Any) -> dict[str, Any]:
        return self.skills.get(name)

    def run_skill(self, name: str, inputs: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        return self.skills.run(name, inputs or {}, self._call_fact)

    def create_skill_draft(
        self, name: str, description: str, instructions: str, tool_plan: list[dict[str, Any]], input_schema: dict[str, Any] | None = None, **_: Any
    ) -> dict[str, Any]:
        return self.skills.create_draft(
            name=name,
            description=description,
            instructions=instructions,
            tool_plan=tool_plan,
            input_schema=input_schema or {},
        )

    def save_skill(self, draft_id: str, **_: Any) -> dict[str, Any]:
        return self.skills.save_draft(draft_id)

    def _call_fact(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in ALLOWED_FACT_TOOLS:
            return {"error": f"Skill cannot call unsupported tool {name!r}."}
        function: Callable[..., dict[str, Any]] | None = getattr(self, name, None)
        if function is None:
            return {"error": f"Unsupported tool {name!r}."}
        try:
            return function(**arguments)
        except (KeyError, TypeError, ValueError) as exc:
            return {"error": f"Invalid arguments for {name}: {exc}"}

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name in ALLOWED_FACT_TOOLS:
            return self._call_fact(name, arguments)
        functions: dict[str, Callable[..., dict[str, Any]]] = {
            "search_engineering_knowledge": self.search_engineering_knowledge,
            "get_visual_evidence": self.get_visual_evidence,
            "list_skills": self.list_skills,
            "get_skill": self.get_skill,
            "run_skill": self.run_skill,
            "create_skill_draft": self.create_skill_draft,
            "save_skill": self.save_skill,
        }
        function = functions.get(name)
        if function is None:
            return {"error": f"Unsupported tool {name!r}."}
        try:
            return function(**arguments)
        except (KeyError, TypeError, ValueError) as exc:
            return {"error": f"Invalid arguments for {name}: {exc}"}


def _system_prompt(facts: dict[str, Any], vision_evidence: list[dict[str, str]] | None = None) -> str:
    summary = facts.get("summary", {})
    drawing = facts.get("drawing", {})
    return f"""You are an engineering-drawing review copilot. Answer only from the supplied facts and tool results for {drawing.get('image_name', 'this drawing')}.

Rules:
- Before claiming a drawing-specific fact, call a tool that supplies the supporting evidence.
- Cite each drawing-specific assertion with the returned feature ID in square brackets, such as [P87] or [GT-Line0]. Cite a dimension ID where relevant.
- A predicted primitive is model output, not confirmed CAD geometry. State detector confidence for material recommendations based on it.
- Ground-truth labels are evaluation-only and must be named as such; they are not available for a normal customer drawing.
- OCR text is unverified. Do not invent material, scale, tolerances, units, datum schemes, threads, or manufacturing requirements.
- Distances from measure_center_distance are image-space only, never physical dimensions.
- Treat rule findings and symmetry/hole patterns as review prompts, not pass/fail certification. Recommend visual or drawing-standard verification where appropriate.
- If the facts do not support an answer, say what is unknown and name the evidence needed.
- Reusable skills are declared, non-executable workflows over facts tools. Inspect a skill before running it. When asked to create one, first create_skill_draft; explain its fixed plan and request review. Persist it with save_skill only if the user asked to save it and skill writing was explicitly enabled.
- Never represent a skill as a new sensor or source of truth. It only packages existing evidence lookups.

{"Attached visual evidence, in order: " + "; ".join(item['id'] + " (" + item['description'] + ")" for item in vision_evidence) + ". Inspect it for visual verification, but use facts tools for feature IDs, values, confidence, and final claims." if vision_evidence else "No vision evidence is attached for this session."}

Small initial context: {json.dumps({'summary': summary, 'coordinate_space': drawing.get('coordinate_space'), 'physical_scale': drawing.get('physical_scale')}, ensure_ascii=False)}"""


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310 - user selects local/known Ollama URL
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail or exc.reason}") from exc


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:  # nosec B310 - user selects local/known Ollama URL
        return json.loads(response.read().decode("utf-8"))


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls") or []
    return [call for call in calls if isinstance(call, dict) and isinstance(call.get("function"), dict)]


def ask_ollama(
    facts: dict[str, Any], model: str, ollama_url: str, question: str, temperature: float, num_ctx: int, timeout: float,
    tools: FactTools | None = None, messages: list[dict[str, Any]] | None = None, vision_evidence: list[dict[str, str]] | None = None,
) -> str:
    tools = tools or FactTools(facts)
    base_url = ollama_url.rstrip("/")
    messages = messages if messages is not None else [{"role": "system", "content": _system_prompt(facts, vision_evidence)}]
    user_message: dict[str, Any] = {"role": "user", "content": question}
    if vision_evidence and not any("images" in message for message in messages):
        user_message["images"] = [item["base64_jpeg"] for item in vision_evidence]
    messages.append(user_message)
    for _ in range(MAX_TOOL_TURNS):
        response = _post_json(
            f"{base_url}/api/chat",
            {
                "model": model,
                "messages": messages,
                "tools": TOOL_SCHEMAS,
                "stream": False,
                "options": {"temperature": temperature, "num_ctx": num_ctx},
            },
            timeout,
        )
        message = response.get("message")
        if not isinstance(message, dict):
            raise RuntimeError(f"Ollama returned no chat message: {response}")
        calls = _tool_calls(message)
        messages.append(message)
        if not calls:
            content = str(message.get("content", "")).strip()
            return content or "The local model returned no answer."
        for call in calls:
            function = call["function"]
            name = str(function.get("name", ""))
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            result = tools.call(name, arguments)
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result, ensure_ascii=False)})
    return "I reached the maximum grounded-tool turns before the answer was complete. Please ask a narrower question."


def check_ollama(ollama_url: str, timeout: float, selected_model: str | None = None) -> int:
    try:
        payload = _get_json(f"{ollama_url.rstrip('/')}/api/tags", timeout)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Cannot reach Ollama at {ollama_url}: {exc}", file=sys.stderr)
        return 2
    models = [model.get("name", "unknown") for model in payload.get("models", [])]
    print("[DONE] Ollama is reachable. Available models:")
    for model in models:
        print(f"  - {model}")
    if selected_model:
        if selected_model in models:
            print(f"[DONE] Selected copilot model is available: {selected_model}")
        else:
            print(f"[WARN] Selected copilot model is not installed: {selected_model}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Ground a local Ollama model in deterministic engineering-drawing facts.")
    parser.add_argument("--facts", type=Path, default=Path("drawing_facts.json"), help="Facts JSON created by build_drawing_facts.py.")
    parser.add_argument("--model", default="qwen3.5:9b", help="Exact model name reported by `ollama list` or --check-ollama.")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama service base URL.")
    parser.add_argument("--question", help="One grounded engineering question.")
    parser.add_argument("--interactive", action="store_true", help="Continue asking questions until you enter exit.")
    parser.add_argument("--check-ollama", action="store_true", help="List models visible to the configured Ollama service, then exit.")
    parser.add_argument("--skills-dir", type=Path, default=Path("engineering_skills"), help="Directory holding user-approved declarative skill JSON files.")
    parser.add_argument("--list-skills", action="store_true", help="List built-in and local skills, then exit without contacting Ollama.")
    parser.add_argument("--run-skill", help="Run one skill directly against the facts file, then print its evidence JSON.")
    parser.add_argument("--skill-input", default="{}", help="JSON object supplying inputs for --run-skill.")
    parser.add_argument("--allow-skill-write", action="store_true", help="Allow Ollama to save a reviewed draft skill into --skills-dir.")
    parser.add_argument("--allow-skill-replace", action="store_true", help="Also allow an explicitly saved draft to overwrite an existing local skill.")
    parser.add_argument("--knowledge-index", type=Path, help="Optional JSON index built by build_knowledge_index.py from approved engineering references.")
    parser.add_argument("--vision-crops", choices=("none", "auto"), default="none", help="Attach original drawing, detector overlay, and focused crops for a vision-capable Ollama model.")
    parser.add_argument("--vision-image", type=Path, help="Original drawing image for --vision-crops. Defaults to drawing.image_path in the facts file.")
    parser.add_argument("--vision-max-side", type=int, default=1200, help="Maximum side length for each attached vision image.")
    parser.add_argument("--vision-max-images", type=int, default=3, help="Maximum attached vision images; lower this first if VRAM is tight.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature; 0 is recommended for factual review.")
    parser.add_argument("--num-ctx", type=int, default=4096, help="Ollama context window requested for this chat (4096 is the conservative 8 GB default).")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request Ollama timeout in seconds.")
    add_progress_argument(parser)
    args = parser.parse_args()
    if args.check_ollama:
        with RichProgress(enabled=not args.no_progress) as progress:
            task = progress.add_task("Checking local Ollama service", total=1)
            result = check_ollama(args.ollama_url, args.timeout, args.model)
            progress.complete(task, "Ollama check complete")
            return result
    if args.list_skills:
        with RichProgress(enabled=not args.no_progress) as progress:
            task = progress.add_task("Loading engineering skills", total=1)
            library = SkillLibrary(args.skills_dir, allow_write=args.allow_skill_write, allow_replace=args.allow_skill_replace)
            print(json.dumps(library.list(), indent=2))
            progress.complete(task, "Engineering skills loaded")
            return 0
    if not args.question and not args.interactive and not args.run_skill:
        parser.error("Specify --question, --interactive, or --run-skill (or use --check-ollama/--list-skills).")
    if not args.facts.is_file():
        print(f"[ERROR] Facts file not found: {args.facts}. Run build_drawing_facts.py first.", file=sys.stderr)
        return 2
    try:
        facts = json.loads(args.facts.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid facts JSON: {exc}", file=sys.stderr)
        return 2

    try:
        knowledge_index = KnowledgeIndex.load(args.knowledge_index) if args.knowledge_index else None
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Could not load --knowledge-index: {exc}", file=sys.stderr)
        return 2
    vision_evidence: list[dict[str, str]] = []
    if args.vision_crops != "none":
        image_path = args.vision_image or Path(str(facts.get("drawing", {}).get("image_path", "")))
        try:
            with RichProgress(enabled=not args.no_progress) as progress:
                task = progress.add_task("Preparing vision evidence", total=1)
                vision_evidence = build_vision_evidence(
                    image_path,
                    facts,
                    mode=args.vision_crops,
                    max_side=args.vision_max_side,
                    max_images=args.vision_max_images,
                )
                progress.complete(task, "Vision evidence prepared")
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"[ERROR] Could not prepare vision evidence: {exc}", file=sys.stderr)
            return 2
        print(f"[INFO] Attached {len(vision_evidence)} vision evidence image(s) for this session.")

    tools = FactTools(
        facts,
        skills_dir=args.skills_dir,
        allow_skill_write=args.allow_skill_write,
        allow_skill_replace=args.allow_skill_replace,
        knowledge_index=knowledge_index,
        vision_evidence=vision_evidence,
    )
    if args.run_skill:
        try:
            skill_input = json.loads(args.skill_input)
        except json.JSONDecodeError as exc:
            print(f"[ERROR] --skill-input must be a JSON object: {exc}", file=sys.stderr)
            return 2
        if not isinstance(skill_input, dict):
            print("[ERROR] --skill-input must be a JSON object.", file=sys.stderr)
            return 2
        with RichProgress(enabled=not args.no_progress) as progress:
            task = progress.add_task(f"Running skill: {args.run_skill}", total=1)
            print(json.dumps(tools.run_skill(args.run_skill, skill_input), indent=2, ensure_ascii=False))
            progress.complete(task, "Engineering skill complete")
        return 0

    conversation: list[dict[str, Any]] = [{"role": "system", "content": _system_prompt(facts, vision_evidence)}]

    def answer(question: str) -> None:
        try:
            with RichProgress(enabled=not args.no_progress) as progress:
                task = progress.add_task(f"Qwen reasoning with {args.model}", total=1)
                answer_text = ask_ollama(
                    facts,
                    args.model,
                    args.ollama_url,
                    question,
                    args.temperature,
                    args.num_ctx,
                    args.timeout,
                    tools=tools,
                    messages=conversation,
                    vision_evidence=vision_evidence,
                )
                progress.complete(task, "Grounded response complete")
            print(answer_text)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            print(
                f"[ERROR] Could not query Ollama at {args.ollama_url}: {exc}\n"
                "Confirm that Ollama is running, use --check-ollama to verify the model name, or pass --ollama-url for its service.",
                file=sys.stderr,
            )

    if args.question:
        answer(args.question)
    if args.interactive:
        print("Grounded drawing copilot. Enter a question, or type exit.")
        while True:
            try:
                question = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in {"exit", "quit"}:
                break
            if question:
                answer(question)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
