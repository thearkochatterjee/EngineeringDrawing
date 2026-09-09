"""Versioned, tool-bounded reusable workflows for the drawing copilot.

Unlike executable plugins, an engineering skill contains instructions, declared
inputs, and a fixed sequence of calls to the already allowlisted facts tools.
That makes skills inspectable, portable, and safe to create from an LLM while
still avoiding arbitrary Python, shell commands, network access, or arbitrary
file reads.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


SKILL_SCHEMA_VERSION = "engineering-skill/v1"
SKILL_FILE_SUFFIX = ".skill.json"
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]{2,63}$")
INPUT_PATTERN = re.compile(r"^\$input\.([a-z][a-z0-9_]{0,31})$")
MAX_SKILL_STEPS = 8
MAX_TEXT_LENGTH = 6_000

# This allowlist is deliberately smaller than the full agent toolset. Skills
# cannot acquire write, command, network, or model-management capabilities.
ALLOWED_FACT_TOOLS = {
    "get_drawing_summary",
    "list_features",
    "get_feature",
    "list_holes",
    "measure_center_distance",
    "list_dimensions",
    "get_constraints",
    "find_symmetry",
    "find_repeated_hole_patterns",
    "get_topology",
    "check_rules",
    "get_annotations",
    "get_provenance",
    "search_engineering_knowledge",
}


BUILTIN_SKILLS: list[dict[str, Any]] = [
    {
        "schema_version": SKILL_SCHEMA_VERSION,
        "name": "drawing-intake-review",
        "version": "1.0.0",
        "description": "First-pass inventory, annotation availability, and deterministic review risks.",
        "instructions": "Summarize the drawing, distinguish predictions from verified labels/OCR, then prioritize review findings. Cite feature or dimension IDs.",
        "input_schema": {},
        "tool_plan": [
            {"tool": "get_drawing_summary", "arguments": {}},
            {"tool": "get_provenance", "arguments": {}},
            {"tool": "check_rules", "arguments": {"limit": 25}},
            {"tool": "get_annotations", "arguments": {"limit": 25}},
        ],
        "author": "built-in",
        "trust": "built-in",
        "requires_engineer_review": True,
    },
    {
        "schema_version": SKILL_SCHEMA_VERSION,
        "name": "hole-pattern-review",
        "version": "1.0.0",
        "description": "Review circular-feature candidates, repeated patterns, and overlap findings.",
        "instructions": "Treat circles as hole candidates only. Report repeated groups, image-space evidence, and anything requiring drawing-callout verification.",
        "input_schema": {},
        "tool_plan": [
            {"tool": "list_holes", "arguments": {"source": "predicted", "limit": 50}},
            {"tool": "find_repeated_hole_patterns", "arguments": {}},
            {"tool": "check_rules", "arguments": {"severity": "warning", "limit": 25}},
        ],
        "author": "built-in",
        "trust": "built-in",
        "requires_engineer_review": True,
    },
    {
        "schema_version": SKILL_SCHEMA_VERSION,
        "name": "geometry-confidence-review",
        "version": "1.0.0",
        "description": "Find uncertain, tiny, and duplicated predicted geometry before downstream use.",
        "instructions": "Use the review findings to identify geometry that should be visually checked. Do not call a prediction confirmed solely because its score is high.",
        "input_schema": {},
        "tool_plan": [
            {"tool": "get_drawing_summary", "arguments": {}},
            {"tool": "check_rules", "arguments": {"severity": "review", "limit": 50}},
            {"tool": "check_rules", "arguments": {"severity": "warning", "limit": 50}},
        ],
        "author": "built-in",
        "trust": "built-in",
        "requires_engineer_review": True,
    },
    {
        "schema_version": SKILL_SCHEMA_VERSION,
        "name": "dataset-evaluation-review",
        "version": "1.0.0",
        "description": "Compare prediction evidence with ParaCAD dimensions and constraints when ground truth is supplied.",
        "instructions": "State clearly that ground-truth labels are evaluation-only. Compare their coverage with prediction evidence; do not use them as facts from an unlabeled drawing.",
        "input_schema": {},
        "tool_plan": [
            {"tool": "get_drawing_summary", "arguments": {}},
            {"tool": "list_dimensions", "arguments": {"source": "label", "limit": 50}},
            {"tool": "get_constraints", "arguments": {"limit": 50}},
            {"tool": "check_rules", "arguments": {"limit": 25}},
        ],
        "author": "built-in",
        "trust": "built-in",
        "requires_engineer_review": True,
    },
    {
        "schema_version": SKILL_SCHEMA_VERSION,
        "name": "standards-reference-review",
        "version": "1.0.0",
        "description": "Retrieve approved local standards or process notes relevant to an engineer question.",
        "instructions": "Use only retrieved citations. State that applicability, revision, and governing requirements must be confirmed by the engineer.",
        "input_schema": {"query": {"description": "Specific standard, process, tolerance, or drawing-review question", "required": True}},
        "tool_plan": [{"tool": "search_engineering_knowledge", "arguments": {"query": "$input.query", "limit": 5}}],
        "author": "built-in",
        "trust": "built-in",
        "requires_engineer_review": True,
    },
]


class SkillValidationError(ValueError):
    """A skill failed the strict declarative workflow schema."""


def _plain_text(value: Any, field: str, required: bool = True, max_length: int = MAX_TEXT_LENGTH) -> str:
    if not isinstance(value, str) or (required and not value.strip()):
        raise SkillValidationError(f"{field} must be a non-empty string.")
    if len(value) > max_length:
        raise SkillValidationError(f"{field} exceeds the {max_length}-character limit.")
    return value.strip()


def _validate_json_value(value: Any, input_names: set[str], path: str = "arguments") -> Any:
    """Accept only small JSON data and declared $input placeholders."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > MAX_TEXT_LENGTH:
            raise SkillValidationError(f"{path} contains an overlong string.")
        match = INPUT_PATTERN.fullmatch(value)
        if match and match.group(1) not in input_names:
            raise SkillValidationError(f"{path} references undeclared input {match.group(1)!r}.")
        return value
    if isinstance(value, list):
        if len(value) > 50:
            raise SkillValidationError(f"{path} has too many list entries.")
        return [_validate_json_value(item, input_names, f"{path}[]") for item in value]
    if isinstance(value, dict):
        if len(value) > 50:
            raise SkillValidationError(f"{path} has too many keys.")
        return {str(key): _validate_json_value(item, input_names, f"{path}.{key}") for key, item in value.items()}
    raise SkillValidationError(f"{path} must contain JSON-compatible values only.")


def validate_skill(skill: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate an inspectable, non-executable skill manifest."""
    if not isinstance(skill, dict):
        raise SkillValidationError("Skill must be a JSON object.")
    name = _plain_text(skill.get("name"), "name", max_length=64).lower()
    if not NAME_PATTERN.fullmatch(name):
        raise SkillValidationError("name must be a 3-64 character lowercase hyphen slug.")
    input_schema_raw = skill.get("input_schema", {})
    if not isinstance(input_schema_raw, dict) or len(input_schema_raw) > 16:
        raise SkillValidationError("input_schema must be an object with at most 16 entries.")
    input_schema: dict[str, dict[str, Any]] = {}
    for input_name, details in input_schema_raw.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", str(input_name)):
            raise SkillValidationError(f"Invalid input name {input_name!r}.")
        if not isinstance(details, dict):
            raise SkillValidationError(f"Input {input_name!r} must be an object.")
        input_schema[str(input_name)] = {
            "description": _plain_text(details.get("description"), f"input_schema.{input_name}.description"),
            "required": bool(details.get("required", False)),
        }
    input_names = set(input_schema)
    plan_raw = skill.get("tool_plan")
    if not isinstance(plan_raw, list) or not plan_raw or len(plan_raw) > MAX_SKILL_STEPS:
        raise SkillValidationError(f"tool_plan must contain between 1 and {MAX_SKILL_STEPS} steps.")
    plan: list[dict[str, Any]] = []
    for index, step in enumerate(plan_raw, start=1):
        if not isinstance(step, dict):
            raise SkillValidationError(f"tool_plan step {index} must be an object.")
        tool = _plain_text(step.get("tool"), f"tool_plan step {index}.tool", max_length=80)
        if tool not in ALLOWED_FACT_TOOLS:
            raise SkillValidationError(f"tool_plan step {index} uses disallowed tool {tool!r}.")
        arguments = step.get("arguments", {})
        if not isinstance(arguments, dict):
            raise SkillValidationError(f"tool_plan step {index}.arguments must be an object.")
        plan.append({"tool": tool, "arguments": _validate_json_value(arguments, input_names, f"tool_plan[{index}].arguments")})
    normalized = {
        "schema_version": SKILL_SCHEMA_VERSION,
        "name": name,
        "version": _plain_text(skill.get("version", "1.0.0"), "version", max_length=32),
        "description": _plain_text(skill.get("description"), "description", max_length=500),
        "instructions": _plain_text(skill.get("instructions"), "instructions"),
        "input_schema": input_schema,
        "tool_plan": plan,
        "author": _plain_text(skill.get("author", "user"), "author", max_length=120),
        "trust": "built-in" if skill.get("trust") == "built-in" else "local-user-approved",
        "requires_engineer_review": True,
    }
    return normalized


def _interpolate(value: Any, inputs: dict[str, Any]) -> Any:
    if isinstance(value, str):
        match = INPUT_PATTERN.fullmatch(value)
        return inputs[match.group(1)] if match else value
    if isinstance(value, list):
        return [_interpolate(item, inputs) for item in value]
    if isinstance(value, dict):
        return {key: _interpolate(item, inputs) for key, item in value.items()}
    return value


class SkillLibrary:
    """Read local skills, hold session drafts, and execute only safe tool plans."""

    def __init__(self, root: Path, allow_write: bool = False, allow_replace: bool = False) -> None:
        self.root = root
        self.allow_write = allow_write
        self.allow_replace = allow_replace
        self._drafts: dict[str, dict[str, Any]] = {}

    def _local_skills(self) -> dict[str, dict[str, Any]]:
        skills: dict[str, dict[str, Any]] = {}
        if not self.root.is_dir():
            return skills
        for path in sorted(self.root.glob(f"*{SKILL_FILE_SUFFIX}")):
            try:
                loaded = validate_skill(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, SkillValidationError):
                continue
            loaded["file"] = str(path)
            skills[loaded["name"]] = loaded
        return skills

    def _all(self) -> dict[str, dict[str, Any]]:
        # Built-ins intentionally win to prevent a local file from changing a
        # baseline workflow silently. User skills must use their own names.
        skills = {skill["name"]: validate_skill(skill) for skill in BUILTIN_SKILLS}
        for name, skill in self._local_skills().items():
            if name not in skills:
                skills[name] = skill
        return skills

    def list(self) -> dict[str, Any]:
        records = []
        for skill in self._all().values():
            records.append(
                {
                    "name": skill["name"],
                    "version": skill["version"],
                    "description": skill["description"],
                    "inputs": skill["input_schema"],
                    "trust": skill["trust"],
                    "step_count": len(skill["tool_plan"]),
                }
            )
        return {"skill_directory": str(self.root), "write_enabled": self.allow_write, "skills": records}

    def get(self, name: str) -> dict[str, Any]:
        skill = self._all().get(name)
        return copy.deepcopy(skill) if skill else {"error": f"Unknown skill {name!r}. Use list_skills first."}

    def create_draft(self, **candidate: Any) -> dict[str, Any]:
        try:
            skill = validate_skill({**candidate, "author": "ollama-draft", "trust": "local-user-approved"})
        except SkillValidationError as exc:
            return {"error": str(exc)}
        all_skills = self._all()
        existing = all_skills.get(skill["name"])
        if existing and existing["trust"] == "built-in":
            return {"error": f"{skill['name']!r} is a built-in skill and cannot be replaced."}
        if existing and not self.allow_replace:
            return {"error": f"A local skill named {skill['name']!r} already exists. Restart with --allow-skill-replace to draft a replacement."}
        draft_id = f"draft-{len(self._drafts) + 1}"
        self._drafts[draft_id] = skill
        return {
            "draft_id": draft_id,
            "skill": skill,
            "next_step": "Ask the user to review the fixed tool plan. save_skill requires --allow-skill-write.",
        }

    def save_draft(self, draft_id: str) -> dict[str, Any]:
        if not self.allow_write:
            return {"error": "Skill writes are disabled. Restart with --allow-skill-write after reviewing the draft."}
        skill = self._drafts.get(draft_id)
        if skill is None:
            return {"error": f"Unknown draft {draft_id!r}. Create a draft in this session first."}
        path = self.root / f"{skill['name']}{SKILL_FILE_SUFFIX}"
        if path.exists() and not self.allow_replace:
            return {"error": f"Skill file already exists: {path}. Restart with --allow-skill-replace to overwrite it."}
        persisted = copy.deepcopy(skill)
        persisted["author"] = "ollama-user-approved"
        persisted["created_at_utc"] = datetime.now(UTC).isoformat()
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(persisted, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            return {"error": f"Could not save skill to {path}: {exc}"}
        return {"saved": True, "name": skill["name"], "path": str(path), "skill": persisted}

    def run(self, name: str, inputs: dict[str, Any], call_fact_tool: Callable[[str, dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        skill = self._all().get(name)
        if skill is None:
            return {"error": f"Unknown skill {name!r}. Use list_skills first."}
        if not isinstance(inputs, dict):
            return {"error": "inputs must be a JSON object."}
        missing = [key for key, schema in skill["input_schema"].items() if schema["required"] and key not in inputs]
        if missing:
            return {"error": f"Missing required skill input(s): {', '.join(missing)}."}
        results = []
        for step in skill["tool_plan"]:
            arguments = _interpolate(step["arguments"], inputs)
            results.append({"tool": step["tool"], "arguments": arguments, "result": call_fact_tool(step["tool"], arguments)})
        return {
            "skill": {key: skill[key] for key in ("name", "version", "description", "instructions", "input_schema", "trust", "requires_engineer_review")},
            "inputs": inputs,
            "results": results,
        }
