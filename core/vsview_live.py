"""Live vsview preview: the preview script starts this inside vsview. It watches the job file DVD Restore rewrites on
every change and then triggers vsview's own Reload (Ctrl+R), which keeps the open tab, frame and zoom.
vsview calls the destroy hook before every reload and when the tab closes, so exactly one watcher runs."""
import os
import threading
import time


def watch(json_path, vpy_path):
    try:
        from vsview import register_on_workspace_destroy
        from PySide6.QtCore import QObject, QTimer, Signal, Slot
        from PySide6.QtWidgets import QApplication
    except Exception:                                    # noqa: BLE001 - not running inside vsview
        return
    app = QApplication.instance()
    if app is None:
        return
    target = os.path.normcase(os.path.abspath(vpy_path))

    class Bridge(QObject):
        fire = Signal()

        @Slot()
        def reload(self):                                # runs in vsview's main (GUI) thread
            for w in app.allWidgets():
                c = getattr(w, "content", None)
                if c is None or not hasattr(w, "reload_content"):
                    continue
                try:
                    same = os.path.normcase(os.path.abspath(str(c))) == target
                except Exception:                        # noqa: BLE001
                    same = False
                if not same:
                    continue
                pb = getattr(w, "playback", None)
                if pb is not None and not getattr(pb, "can_reload", True):
                    QTimer.singleShot(500, self.reload)  # busy (playing / rendering): try again shortly
                    return
                w.reload_content()
                return

    bridge = Bridge()
    bridge.moveToThread(app.thread())
    bridge.fire.connect(bridge.reload)
    stop = threading.Event()

    def mtime():
        try:
            return os.path.getmtime(json_path)
        except OSError:
            return None

    def loop():
        last = mtime()
        while not stop.wait(0.3):
            m = mtime()
            if m is None or m == last:
                continue
            time.sleep(0.25)                             # let a burst of changes settle
            if stop.is_set():
                return
            last = mtime()
            bridge.fire.emit()
            return                                       # the reload re-runs the script, which starts a new watcher

    threading.Thread(target=loop, name="dvdrestore-live", daemon=True).start()
    register_on_workspace_destroy(stop.set)
    _KEEP.append(bridge)                                 # keep the Qt object alive


_KEEP = []
