"""Small, dependency-tolerant Rich progress helpers for local CLI tools.

Progress always renders on stderr so commands that write JSON or JSONL to
stdout remain safe to pipe into another program.  Rich is optional for scripts
that otherwise have no external dependencies; ``--no-progress`` selects the
same quiet, plain-text fallback explicitly.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except ImportError:  # pragma: no cover - supports lightweight script installs
    Console = None  # type: ignore[assignment,misc]
    Progress = None  # type: ignore[assignment,misc]


class RichProgress:
    """A no-op-safe Rich progress display with an API suited to batch tools."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled and Progress is not None and Console is not None)
        self._progress: Any = None
        if self.enabled:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=Console(stderr=True),
                refresh_per_second=8,
                transient=False,
            )

    def __enter__(self) -> "RichProgress":
        return self.start()

    def start(self) -> "RichProgress":
        if self._progress is not None:
            self._progress.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()

    def stop(self) -> None:
        if self._progress is not None:
            self._progress.stop()

    def add_task(self, description: str, total: Optional[float] = None) -> Optional[int]:
        if self._progress is None:
            return None
        return int(self._progress.add_task(description, total=total))

    def update(
        self,
        task_id: Optional[int],
        *,
        advance: Optional[float] = None,
        completed: Optional[float] = None,
        total: Optional[float] = None,
        description: Optional[str] = None,
    ) -> None:
        if self._progress is None or task_id is None:
            return
        changes: dict[str, Any] = {}
        if advance is not None:
            changes["advance"] = advance
        if completed is not None:
            changes["completed"] = completed
        if total is not None:
            changes["total"] = total
        if description is not None:
            changes["description"] = description
        if changes:
            self._progress.update(task_id, **changes)

    def advance(self, task_id: Optional[int], amount: float = 1) -> None:
        self.update(task_id, advance=amount)

    def complete(self, task_id: Optional[int], description: Optional[str] = None) -> None:
        if self._progress is None or task_id is None:
            return
        task = self._progress.tasks[task_id]
        self.update(task_id, completed=task.total if task.total is not None else 1, total=task.total or 1, description=description)

    def log(self, message: str) -> None:
        """Write a stable diagnostic without corrupting the live display."""
        if self._progress is not None:
            self._progress.console.print(message)
        else:
            print(message, file=sys.stderr)


def add_progress_argument(parser: Any) -> None:
    """Give a CLI an explicit opt-out, useful for logs and CI."""
    parser.add_argument("--no-progress", action="store_true", help="Disable Rich progress output.")
