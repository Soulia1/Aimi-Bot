"""Deliver Windows hotkeys independently of capture and GPU inference."""

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from threading import Event, Thread


WM_QUIT = 0x0012
WM_HOTKEY = 0x0312
MOD_NOREPEAT = 0x4000
HOTKEY_ID = 1


@contextmanager
def windows_hotkey(key, callback):
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32")
    user32.PostThreadMessageW.argtypes = (
        wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
    )
    ready = Event()
    registered, error = False, None
    thread_id = None

    def listen():
        nonlocal registered, error, thread_id
        message = wintypes.MSG()
        thread_id = kernel32.GetCurrentThreadId()
        user32.PeekMessageW(ctypes.byref(message), None, 0, 0, 0)
        registered = bool(user32.RegisterHotKey(None, HOTKEY_ID, MOD_NOREPEAT, key))
        if not registered:
            error = ctypes.get_last_error()
        ready.set()
        if not registered:
            return
        try:
            while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
                if message.message == WM_HOTKEY and message.wParam == HOTKEY_ID:
                    callback()
        finally:
            user32.UnregisterHotKey(None, HOTKEY_ID)

    thread = Thread(target=listen, name="AIMi hotkey", daemon=True)
    thread.start()
    ready.wait()
    try:
        yield registered, error
    finally:
        if registered:
            user32.PostThreadMessageW(thread_id, WM_QUIT, 0, 0)
        thread.join()
