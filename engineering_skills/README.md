# Engineering copilot skills

This directory stores user-approved `*.skill.json` workflows created by the
local drawing copilot. Built-in skills are shipped in `engineering_skills.py`
and are always available even when this directory is empty.

A local skill is declarative rather than executable. It contains:

- a name, version, description, and engineer-review instructions;
- optional named inputs; and
- a fixed plan using only existing drawing-facts tools.

The copilot validates every skill before it is drafted or loaded. Skills cannot
run a shell command, write to a drawing, access the network, or read arbitrary
files. Review the returned draft before running the copilot with
`--allow-skill-write`; overwriting a saved local skill also requires
`--allow-skill-replace`.
