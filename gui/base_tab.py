from __future__ import annotations
import functools
import logging
import threading
from contextlib import contextmanager
from typing import Callable, Any

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QWidget, QApplication, QMessageBox

__all__ = ["BaseTabWidget"]

log = logging.getLogger(__name__)


class _ThreadBridge(QObject):
    """Helper object that proxies a workload executed on a background thread
    back to the GUI thread via a Qt signal. This avoids directly touching
    QWidget methods from non‑GUI threads.
    """
    done = pyqtSignal(object)
    progress = pyqtSignal(int)


class BaseTabWidget(QWidget):
    """Common functionality shared by every GUI tab.

    *   **threaded(task_fn, on_done)** – run `task_fn()` in a Python thread and
        invoke `on_done(result)` on the GUI thread.
    *   **busy_cursor** – context‑manager that shows the Qt *WaitCursor*.
    *   **alert(msg)** – convenience wrapper around `QMessageBox.information`.
    """

    # ------------------------------------------------------------------ threading helper
    def threaded(self, fn: Callable[[], Any], on_done: Callable[[Any], None] | None = None):
        """Execute *fn* in a Python thread; call *on_done(result)* on the GUI thread."""

        bridge = _ThreadBridge()

        def _worker():
            try:
                result = fn()
            except Exception as exc:
                log.exception("threaded task failed")
                result = exc
            bridge.done.emit(result)

        bridge.done.connect(on_done or (lambda *_: None))
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    # ------------------------------------------------------------------
    @contextmanager
    def busy_cursor(self):
        """Temporarily switch to the *WaitCursor* while inside the context."""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # type: ignore[name-defined]
        try:
            yield
        finally:
            QApplication.restoreOverrideCursor()

    # ------------------------------------------------------------------
    def alert(self, message: str, title: str = "Info"):
        QMessageBox.information(self, title, message)