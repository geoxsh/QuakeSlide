"""Compact timestamped console output for QuakeSlide runs.

Date: August 2026
"""

from contextlib import contextmanager, redirect_stdout
from datetime import datetime
import re
import sys
import time


_STEP_LINE = re.compile(r"^\[\d+/\d+\]")
_DETAIL_INDENT = "      "


def run_log(message, stream=None):
    """Write one consistently formatted QuakeSlide status line."""
    target = stream if stream is not None else sys.stdout
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} | {message}\n"
    try:
        target.write(line)
    except UnicodeEncodeError:
        # Logging must never abort a scientific run because the host console
        # uses a legacy encoding such as Windows GBK.
        encoding = getattr(target, "encoding", None) or "utf-8"
        safe_line = line.encode(encoding, errors="replace").decode(encoding)
        target.write(safe_line)
    target.flush()


class TimestampedOutput:
    """Prefix legacy module output while preserving useful indentation."""

    def __init__(self, stream):
        self.stream = stream
        self.buffer = ""

    def write(self, text):
        if not text:
            return 0
        for character in str(text):
            if character == "\r":
                # Progress bars overwrite the current line; retain only the latest state.
                self.buffer = ""
            elif character == "\n":
                self._emit_buffer()
            else:
                self.buffer += character
        return len(text)

    def flush(self):
        self.stream.flush()

    def finish(self):
        self._emit_buffer()
        self.flush()

    def _emit_buffer(self):
        line = self.buffer.strip()
        self.buffer = ""
        if not line or _is_decorative_line(line):
            return
        if not _STEP_LINE.match(line):
            line = f"{_DETAIL_INDENT}{line}"
        run_log(line, stream=self.stream)


def _is_decorative_line(line):
    stripped = line.strip()
    return bool(stripped) and all(character in "=-#_" for character in stripped)


@contextmanager
def timestamped_output():
    """Timestamp stdout emitted by existing workflow modules."""
    formatter = TimestampedOutput(sys.stdout)
    with redirect_stdout(formatter):
        try:
            yield
        finally:
            formatter.finish()


@contextmanager
def workflow_step(index, total, label):
    """Log the start and any failure of one workflow step."""
    prefix = f"[{index}/{total}] {label}"
    short_label = label.split(":", 1)[0]
    run_log(prefix)
    started = time.perf_counter()
    try:
        with timestamped_output():
            yield
    except Exception as error:
        elapsed = time.perf_counter() - started
        run_log(
            f"[{index}/{total}] {short_label} failed ({elapsed:.1f} s): "
            f"{type(error).__name__}: {error}"
        )
        raise
