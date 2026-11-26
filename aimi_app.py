"""
AIMi Studio — Desktop Management & GPU Training Frontend
========================================================
Features:
  1. Bot Control: Start/Stop AIMi (Aimbot or Detector-Only), live console,
     confidence / NMS / capture-range sliders, model selector & image tester.
  2. GPU Training Studio:
       - Add ZIP archives, massive image/video folders, or individual videos
       - Safe extraction (Zip Slip protection, corrupt archive/file handling,
         disk space check)
       - Video frame extraction at configurable FPS
       - GPU-accelerated auto-annotation (person + head) & YOLOv8 training
       - Live progress bar, stage indicator, ETA, and cancellation
       - Automatic ONNX export & integration into AIMi models/
  3. System & Logs: GPU/CUDA/VRAM diagnostics, disk space, and full error logs.
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

# ── Paths ──────────────────────────────────────────────────────────────
APP_ROOT = Path(__file__).resolve().parent
MODELS_DIR = APP_ROOT / "models"
TRAINING_DIR = APP_ROOT / "training"
DATASET_DIR = TRAINING_DIR / "dataset"
RUNS_DIR = TRAINING_DIR / "runs"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def find_bot_python() -> str:
    """Return the best Python executable for running the AIMi bot (prefer CUDA .venv)."""
    venv_py = APP_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    aimi_py = APP_ROOT / ".aimi-env" / "Scripts" / "python.exe"
    if aimi_py.exists():
        return str(aimi_py)
    return sys.executable


def find_train_python() -> str:
    """Return the Python executable that has torch/ultralytics installed."""
    venv_py = APP_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


class AIMiStudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AIMi Studio — Bot Manager & GPU Trainer")
        self.geometry("1100x760")
        self.minsize(940, 660)

        self.ui_queue: queue.Queue = queue.Queue()
        self.bot_proc: subprocess.Popen | None = None
        self.train_cancel = threading.Event()
        self.train_thread: threading.Thread | None = None
        self.train_subproc: subprocess.Popen | None = None
        self.sources: list[dict] = []  # [{"type": "zip"|"folder"|"video", "path": Path}]

        self._build_ui()
        self._refresh_hardware_info()
        self._refresh_model_status()
        self.after(100, self._poll_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══════════════════════════════════════════════════════════════════
    # UI Construction
    # ══════════════════════════════════════════════════════════════════
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header bar
        header = ctk.CTkFrame(self, height=58, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header, text="AIMi Studio", font=ctk.CTkFont(size=22, weight="bold")
        ).grid(row=0, column=0, padx=18, pady=12, sticky="w")

        self.hw_badge = ctk.CTkLabel(
            header, text="Checking GPU...", font=ctk.CTkFont(size=12), text_color="#9aa4b2"
        )
        self.hw_badge.grid(row=0, column=1, padx=12, sticky="w")

        self.model_badge = ctk.CTkLabel(
            header, text="Model: checking...", font=ctk.CTkFont(size=12, weight="bold")
        )
        self.model_badge.grid(row=0, column=2, padx=18, sticky="e")

        # Tabs
        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=1, column=0, padx=14, pady=(8, 14), sticky="nsew")
        self.tab_bot = self.tabs.add("Bot Control")
        self.tab_train = self.tabs.add("Train Model (GPU)")
        self.tab_logs = self.tabs.add("System & Logs")

        self._build_bot_tab()
        self._build_train_tab()
        self._build_logs_tab()

    # ── Tab 1: Bot Control ────────────────────────────────────────────
    def _build_bot_tab(self):
        self.tab_bot.grid_columnconfigure(0, weight=1)
        self.tab_bot.grid_columnconfigure(1, weight=2)
        self.tab_bot.grid_rowconfigure(0, weight=1)

        # Left panel: controls & sliders
        left = ctk.CTkFrame(self.tab_bot)
        left.grid(row=0, column=0, padx=(4, 8), pady=6, sticky="nsew")
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            left, text="Bot Launcher", font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, padx=14, pady=(14, 6), sticky="w")

        self.bot_status_lbl = ctk.CTkLabel(
            left, text="● Stopped", font=ctk.CTkFont(size=14, weight="bold"), text_color="#ef4444"
        )
        self.bot_status_lbl.grid(row=1, column=0, padx=14, pady=(0, 10), sticky="w")

        self.aimbot_mode_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(
            left, text="Enable Aimbot (off = Detector GUI only)",
            variable=self.aimbot_mode_var
        ).grid(row=2, column=0, padx=14, pady=6, sticky="w")

        btn_row = ctk.CTkFrame(left, fg_color="transparent")
        btn_row.grid(row=3, column=0, padx=14, pady=10, sticky="ew")
        btn_row.grid_columnconfigure((0, 1), weight=1)

        self.btn_start_bot = ctk.CTkButton(
            btn_row, text="Start AIMi", fg_color="#16a34a", hover_color="#15803d",
            command=self._start_bot
        )
        self.btn_start_bot.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self.btn_stop_bot = ctk.CTkButton(
            btn_row, text="Stop AIMi", fg_color="#dc2626", hover_color="#b91c1c",
            state="disabled", command=self._stop_bot
        )
        self.btn_stop_bot.grid(row=0, column=1, padx=(6, 0), sticky="ew")

        # Sliders
        ctk.CTkLabel(
            left, text="Detection Parameters", font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=4, column=0, padx=14, pady=(16, 4), sticky="w")

        self.conf_var = ctk.DoubleVar(value=0.36)
        self.conf_lbl = ctk.CTkLabel(left, text="Confidence Threshold: 0.36")
        self.conf_lbl.grid(row=5, column=0, padx=14, pady=(6, 0), sticky="w")
        ctk.CTkSlider(
            left, from_=0.10, to=0.90, number_of_steps=80, variable=self.conf_var,
            command=lambda v: self.conf_lbl.configure(text=f"Confidence Threshold: {v:.2f}")
        ).grid(row=6, column=0, padx=14, pady=(2, 8), sticky="ew")

        self.nms_var = ctk.DoubleVar(value=0.22)
        self.nms_lbl = ctk.CTkLabel(left, text="NMS IoU Threshold: 0.22")
        self.nms_lbl.grid(row=7, column=0, padx=14, pady=(6, 0), sticky="w")
        ctk.CTkSlider(
            left, from_=0.10, to=0.80, number_of_steps=70, variable=self.nms_var,
            command=lambda v: self.nms_lbl.configure(text=f"NMS IoU Threshold: {v:.2f}")
        ).grid(row=8, column=0, padx=14, pady=(2, 8), sticky="ew")

        self.range_var = ctk.IntVar(value=800)
        self.range_lbl = ctk.CTkLabel(left, text=f"Capture Width: {self.range_var.get()} px")
        self.range_lbl.grid(row=9, column=0, padx=14, pady=(6, 0), sticky="w")
        ctk.CTkSlider(
            left, from_=120, to=1200, number_of_steps=108, variable=self.range_var,
            command=lambda v: self.range_lbl.configure(text=f"Capture Width: {int(v)} px")
        ).grid(row=10, column=0, padx=14, pady=(2, 12), sticky="ew")

        # Model actions
        ctk.CTkLabel(
            left, text="Model Management", font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=11, column=0, padx=14, pady=(10, 4), sticky="w")

        self.model_info_lbl = ctk.CTkLabel(
            left, text="", justify="left", font=ctk.CTkFont(size=12), text_color="#9aa4b2"
        )
        self.model_info_lbl.grid(row=12, column=0, padx=14, pady=(0, 8), sticky="w")

        ctk.CTkButton(
            left, text="Test Model on Sample Image", command=self._test_model_on_image
        ).grid(row=13, column=0, padx=14, pady=(4, 14), sticky="ew")

        # Right panel: live bot console
        right = ctk.CTkFrame(self.tab_bot)
        right.grid(row=0, column=1, padx=(8, 4), pady=6, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            right, text="Bot Console Output (F1 = Enable/Pause, Hold Right Mouse to Aim)",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, padx=12, pady=(12, 6), sticky="w")

        self.bot_console = ctk.CTkTextbox(right, font=ctk.CTkFont(family="Consolas", size=12))
        self.bot_console.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        self.bot_console.configure(state="disabled")

    # ── Tab 2: GPU Training Studio ────────────────────────────────────
    def _build_train_tab(self):
        self.tab_train.grid_columnconfigure(0, weight=3)
        self.tab_train.grid_columnconfigure(1, weight=2)
        self.tab_train.grid_rowconfigure(1, weight=1)

        # Top source toolbar
        src_bar = ctk.CTkFrame(self.tab_train)
        src_bar.grid(row=0, column=0, columnspan=2, padx=4, pady=(6, 8), sticky="ew")

        ctk.CTkButton(src_bar, text="+ Add ZIP Archive", command=self._add_zip).pack(side="left", padx=6, pady=8)
        ctk.CTkButton(src_bar, text="+ Add Folder (Images / Videos)", command=self._add_folder).pack(side="left", padx=6, pady=8)
        ctk.CTkButton(src_bar, text="+ Add Video File(s)", command=self._add_videos).pack(side="left", padx=6, pady=8)
        ctk.CTkButton(
            src_bar, text="Clear List", fg_color="#4b5563", hover_color="#374151",
            command=self._clear_sources
        ).pack(side="right", padx=8, pady=8)

        # Left: Source queue + progress
        left = ctk.CTkFrame(self.tab_train)
        left.grid(row=1, column=0, padx=(4, 8), pady=(0, 6), sticky="nsew")
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            left, text="Queued Training Sources (ZIPs, Folders, Videos)",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, padx=12, pady=(10, 4), sticky="w")

        self.source_box = ctk.CTkTextbox(left, height=160, font=ctk.CTkFont(family="Consolas", size=12))
        self.source_box.grid(row=1, column=0, padx=12, pady=4, sticky="nsew")
        self.source_box.configure(state="disabled")

        # Progress section
        prog_frame = ctk.CTkFrame(left)
        prog_frame.grid(row=2, column=0, padx=12, pady=(8, 12), sticky="ew")
        prog_frame.grid_columnconfigure(0, weight=1)

        self.stage_lbl = ctk.CTkLabel(
            prog_frame, text="Ready — add ZIPs, folders, or videos to train.",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.stage_lbl.grid(row=0, column=0, padx=12, pady=(10, 4), sticky="w")

        self.progress_bar = ctk.CTkProgressBar(prog_frame, height=18)
        self.progress_bar.grid(row=1, column=0, padx=12, pady=6, sticky="ew")
        self.progress_bar.set(0)

        self.progress_detail_lbl = ctk.CTkLabel(
            prog_frame, text="0%  |  Waiting to start", font=ctk.CTkFont(size=12), text_color="#9aa4b2"
        )
        self.progress_detail_lbl.grid(row=2, column=0, padx=12, pady=(0, 10), sticky="w")

        # Right: Hyperparameters & Start/Cancel
        right = ctk.CTkFrame(self.tab_train)
        right.grid(row=1, column=1, padx=(8, 4), pady=(0, 6), sticky="nsew")
        right.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            right, text="GPU Training Settings", font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 8), sticky="w")

        # Epochs
        ctk.CTkLabel(right, text="Epochs:").grid(row=1, column=0, padx=12, pady=6, sticky="w")
        self.epochs_entry = ctk.CTkEntry(right)
        self.epochs_entry.insert(0, "100")
        self.epochs_entry.grid(row=1, column=1, padx=12, pady=6, sticky="ew")

        # Batch
        ctk.CTkLabel(right, text="Batch Size:").grid(row=2, column=0, padx=12, pady=6, sticky="w")
        self.batch_opt = ctk.CTkOptionMenu(right, values=["48", "64", "32", "16", "8", "-1 (Auto)"])
        self.batch_opt.set("48")
        self.batch_opt.grid(row=2, column=1, padx=12, pady=6, sticky="ew")

        # Image size
        ctk.CTkLabel(right, text="Image Size:").grid(row=3, column=0, padx=12, pady=6, sticky="w")
        self.imgsz_opt = ctk.CTkOptionMenu(right, values=["416", "640", "320"])
        self.imgsz_opt.set("416")
        self.imgsz_opt.grid(row=3, column=1, padx=12, pady=6, sticky="ew")

        # Auto-label conf
        ctk.CTkLabel(right, text="Auto-Label Conf:").grid(row=4, column=0, padx=12, pady=6, sticky="w")
        self.ann_conf_entry = ctk.CTkEntry(right)
        self.ann_conf_entry.insert(0, "0.25")
        self.ann_conf_entry.grid(row=4, column=1, padx=12, pady=6, sticky="ew")

        # Video sample FPS
        ctk.CTkLabel(right, text="Video Sample FPS:").grid(row=5, column=0, padx=12, pady=6, sticky="w")
        self.video_fps_entry = ctk.CTkEntry(right)
        self.video_fps_entry.insert(0, "2.0")
        self.video_fps_entry.grid(row=5, column=1, padx=12, pady=6, sticky="ew")

        # Device
        ctk.CTkLabel(right, text="Compute Device:").grid(row=6, column=0, padx=12, pady=6, sticky="w")
        ctk.CTkLabel(right, text="GPU (CUDA 0)").grid(row=6, column=1, padx=12, pady=6, sticky="w")

        # Append vs replace
        self.append_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            right, text="Append to existing dataset (keep old frames)", variable=self.append_var
        ).grid(row=7, column=0, columnspan=2, padx=12, pady=10, sticky="w")

        # Action buttons
        self.btn_start_train = ctk.CTkButton(
            right, text="Start GPU Training Pipeline", height=40,
            fg_color="#2563eb", hover_color="#1d4ed8",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._start_training_pipeline
        )
        self.btn_start_train.grid(row=8, column=0, columnspan=2, padx=12, pady=(12, 6), sticky="ew")

        self.btn_cancel_train = ctk.CTkButton(
            right, text="Cancel Training", fg_color="#dc2626", hover_color="#b91c1c",
            state="disabled", command=self._cancel_training
        )
        self.btn_cancel_train.grid(row=9, column=0, columnspan=2, padx=12, pady=(0, 12), sticky="ew")

    # ── Tab 3: System & Logs ──────────────────────────────────────────
    def _build_logs_tab(self):
        self.tab_logs.grid_columnconfigure(0, weight=1)
        self.tab_logs.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(self.tab_logs)
        top.grid(row=0, column=0, padx=4, pady=(6, 8), sticky="ew")
        top.grid_columnconfigure(0, weight=1)

        self.sys_detail_lbl = ctk.CTkLabel(
            top, text="Loading hardware diagnostics...", justify="left", font=ctk.CTkFont(size=12)
        )
        self.sys_detail_lbl.grid(row=0, column=0, padx=12, pady=10, sticky="w")

        ctk.CTkButton(top, text="Refresh", width=90, command=self._refresh_hardware_info).grid(
            row=0, column=1, padx=6, pady=10
        )
        ctk.CTkButton(top, text="Clear Logs", width=100, fg_color="#4b5563", command=self._clear_logs).grid(
            row=0, column=2, padx=6, pady=10
        )
        ctk.CTkButton(top, text="Save Logs", width=100, command=self._save_logs).grid(
            row=0, column=3, padx=(6, 12), pady=10
        )

        self.log_box = ctk.CTkTextbox(self.tab_logs, font=ctk.CTkFont(family="Consolas", size=12))
        self.log_box.grid(row=1, column=0, padx=4, pady=(0, 6), sticky="nsew")
        self.log_box.configure(state="disabled")

    # ══════════════════════════════════════════════════════════════════
    # Logging & Queue Polling
    # ══════════════════════════════════════════════════════════════════
    def log(self, msg: str, level: str = "INFO"):
        ts = time.strftime("%H:%M:%S")
        self.ui_queue.put(("log", f"[{ts}] [{level}] {msg}\n"))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "log":
                    self.log_box.configure(state="normal")
                    self.log_box.insert("end", payload)
                    self.log_box.see("end")
                    self.log_box.configure(state="disabled")
                elif kind == "bot_log":
                    self.bot_console.configure(state="normal")
                    self.bot_console.insert("end", payload)
                    self.bot_console.see("end")
                    self.bot_console.configure(state="disabled")
                elif kind == "bot_stopped":
                    self._on_bot_stopped(payload)
                elif kind == "progress":
                    frac, stage, detail = payload
                    self.progress_bar.set(max(0.0, min(1.0, frac)))
                    self.stage_lbl.configure(text=stage)
                    self.progress_detail_lbl.configure(text=detail)
                elif kind == "train_done":
                    success, message = payload
                    self._on_training_finished(success, message)
                elif kind == "hw":
                    badge, detail = payload
                    self.hw_badge.configure(text=badge)
                    self.sys_detail_lbl.configure(text=detail)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _clear_logs(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def _save_logs(self):
        out = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text files", "*.txt")],
            initialfile="aimi_studio_log.txt"
        )
        if out:
            Path(out).write_text(self.log_box.get("1.0", "end"), encoding="utf-8")
            messagebox.showinfo("Saved", f"Logs saved to:\n{out}")

    # ══════════════════════════════════════════════════════════════════
    # Hardware & Model Diagnostics
    # ══════════════════════════════════════════════════════════════════
    def _refresh_hardware_info(self):
        def worker():
            py = find_train_python()
            disk = shutil.disk_usage(APP_ROOT)
            free_gb = disk.free / (1024 ** 3)
            badge = f"Disk Free: {free_gb:.1f} GB"
            detail = f"Project Root: {APP_ROOT}\nTrain Python: {py}\nDisk Free: {free_gb:.1f} GB"
            try:
                res = subprocess.run(
                    [py, "-c",
                     "import torch; print(torch.__version__); print(torch.cuda.is_available()); "
                     "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); "
                     "print(round(torch.cuda.get_device_properties(0).total_memory/(1024**3), 2) if torch.cuda.is_available() else 0)"],
                    capture_output=True, text=True, timeout=15
                )
                if res.returncode == 0:
                    lines = [l.strip() for l in res.stdout.strip().splitlines()]
                    if len(lines) >= 4:
                        tver, cuda_ok, gpu_name, vram = lines[:4]
                        if cuda_ok == "True":
                            badge = f"GPU: {gpu_name} ({vram} GB VRAM)  |  Disk Free: {free_gb:.1f} GB"
                        else:
                            badge = f"CUDA unavailable  |  Disk Free: {free_gb:.1f} GB"
                        detail = (
                            f"GPU: {gpu_name} ({vram} GB VRAM)  |  CUDA: {cuda_ok}  |  PyTorch: {tver}\n"
                            f"Train Python: {py}  |  Bot Python: {find_bot_python()}  |  Disk Free: {free_gb:.1f} GB"
                        )
            except Exception as e:
                detail += f"\nHardware probe error: {e}"
            self.ui_queue.put(("hw", (badge, detail)))

        threading.Thread(target=worker, daemon=True).start()

    def _refresh_model_status(self):
        pt_path = MODELS_DIR / "aimi-custom.pt"
        onnx_path = MODELS_DIR / "aimi-custom.onnx"
        best_pt = RUNS_DIR / "aimi_robrun" / "weights" / "best.pt"
        if pt_path.exists() or best_pt.exists():
            src_pt = pt_path if pt_path.exists() else best_pt
            mb = src_pt.stat().st_size / (1024 ** 2)
            self.model_badge.configure(text=f"CUDA model: {src_pt.name} ({mb:.1f} MB)", text_color="#22c55e")
        else:
            self.model_badge.configure(text="CUDA weights missing", text_color="#ef4444")

        tr_n = len(list((DATASET_DIR / "images" / "train").glob("*.*"))) if (DATASET_DIR / "images" / "train").exists() else 0
        va_n = len(list((DATASET_DIR / "images" / "val").glob("*.*"))) if (DATASET_DIR / "images" / "val").exists() else 0
        info = (
            f"ONNX Custom: {'Ready' if onnx_path.exists() else 'Not exported yet'}\n"
            f"Best PyTorch (.pt): {'Ready' if best_pt.exists() else 'None'}\n"
            f"Dataset Cache: {tr_n} train / {va_n} val images"
        )
        self.model_info_lbl.configure(text=info)

    # ══════════════════════════════════════════════════════════════════
    # Bot Control
    # ══════════════════════════════════════════════════════════════════
    def _start_bot(self):
        if self.bot_proc is not None and self.bot_proc.poll() is None:
            messagebox.showwarning("Already Running", "AIMi is already running.")
            return

        py = find_bot_python()
        enable_aim = self.aimbot_mode_var.get()
        env = os.environ.copy()
        env["AIMI_CONFIDENCE"] = f"{self.conf_var.get():.2f}"
        env["AIMI_THRESHOLD"] = f"{self.nms_var.get():.2f}"
        env["AIMI_RANGE"] = str(int(self.range_var.get()))
        env["PYTHONUNBUFFERED"] = "1"

        # Only add .aimi-packages when running outside .venv
        use_legacy_pkgs = (".venv" not in py) and (APP_ROOT / ".aimi-packages").exists()
        aimi_pkgs = APP_ROOT / ".aimi-packages"
        code = (
            f"import site, sys; "
            f"p = {repr(str(aimi_pkgs))}; "
            f"(site.addsitedir(p), sys.path.remove(p), sys.path.insert(0, p)) if {use_legacy_pkgs} else None; "
            f"from utils.inputs import aimbot; "
            f"aimbot(ENABLE_AIMBOT={enable_aim})"
        )

        try:
            self.bot_proc = subprocess.Popen(
                [py, "-c", code],
                cwd=str(APP_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            messagebox.showerror("Launch Failed", f"Could not start AIMi:\n{e}")
            self.log(f"Bot launch failed: {e}", "ERROR")
            return

        self.bot_status_lbl.configure(text="● Running", text_color="#22c55e")
        self.btn_start_bot.configure(state="disabled")
        self.btn_stop_bot.configure(state="normal")
        self.log(f"Started AIMi (ENABLE_AIMBOT={enable_aim}, conf={env['AIMI_CONFIDENCE']}, range={env['AIMI_RANGE']})")

        def reader():
            assert self.bot_proc is not None and self.bot_proc.stdout is not None
            for line in self.bot_proc.stdout:
                self.ui_queue.put(("bot_log", line))
            rc = self.bot_proc.wait()
            self.ui_queue.put(("bot_stopped", rc))

        threading.Thread(target=reader, daemon=True).start()

    def _stop_bot(self):
        if self.bot_proc is not None and self.bot_proc.poll() is None:
            try:
                self.bot_proc.terminate()
                self.bot_proc.wait(timeout=3)
            except Exception:
                self.bot_proc.kill()
            self.log("Stopped AIMi process.")

    def _on_bot_stopped(self, rc: int):
        self.bot_status_lbl.configure(text=f"● Stopped (exit {rc})", text_color="#ef4444")
        self.btn_start_bot.configure(state="normal")
        self.btn_stop_bot.configure(state="disabled")
        self.bot_proc = None

    def _test_model_on_image(self):
        img_path = filedialog.askopenfilename(
            title="Select image to test detection",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not img_path:
            return
        py = find_bot_python()
        script = (
            "import cv2\n"
            "from lib.gpu_detector import DEVICE, IMAGE_SIZE, load_detector\n"
            f"frame = cv2.imread({img_path!r})\n"
            "if frame is None: raise ValueError('Failed to read image')\n"
            "model = load_detector()\n"
            f"result = model.predict(frame, imgsz=IMAGE_SIZE, conf={self.conf_var.get():.2f}, "
            f"iou={self.nms_var.get():.2f}, device=DEVICE, verbose=False)[0]\n"
            "print(f'Detections: {len(result.boxes)}')\n"
            "cv2.imshow('AIMi CUDA Model Test (press any key)', result.plot())\n"
            "cv2.waitKey(0)\n"
            "cv2.destroyAllWindows()\n"
        )

        def worker():
            try:
                result = subprocess.run(
                    [py, "-c", script], cwd=str(APP_ROOT), capture_output=True, text=True
                )
                details = result.stderr.strip() if result.returncode else result.stdout.strip()
                if details:
                    self.ui_queue.put(("bot_log", details))
            except OSError as exc:
                self.ui_queue.put(("bot_log", f"CUDA image test failed: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════
    # Training Sources Management
    # ══════════════════════════════════════════════════════════════════
    def _render_sources(self):
        self.source_box.configure(state="normal")
        self.source_box.delete("1.0", "end")
        if not self.sources:
            self.source_box.insert("end", "No sources added yet. Use the buttons above to add ZIPs, folders, or videos.\n")
        else:
            for idx, item in enumerate(self.sources, 1):
                self.source_box.insert("end", f"{idx}. [{item['type'].upper()}] {item['path']}\n")
        self.source_box.configure(state="disabled")

    def _add_zip(self):
        paths = filedialog.askopenfilenames(
            title="Select ZIP archive(s) containing images or videos",
            filetypes=[("ZIP archives", "*.zip")]
        )
        for p in paths:
            self.sources.append({"type": "zip", "path": Path(p)})
        self._render_sources()

    def _add_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing images and/or videos")
        if folder:
            self.sources.append({"type": "folder", "path": Path(folder)})
            self._render_sources()

    def _add_videos(self):
        paths = filedialog.askopenfilenames(
            title="Select video file(s)",
            filetypes=[("Videos", "*.mp4 *.avi *.mkv *.mov *.webm *.m4v")]
        )
        for p in paths:
            self.sources.append({"type": "video", "path": Path(p)})
        self._render_sources()

    def _clear_sources(self):
        self.sources.clear()
        self._render_sources()

    # ══════════════════════════════════════════════════════════════════
    # GPU Training Pipeline (with full error handling)
    # ══════════════════════════════════════════════════════════════════
    def _validate_training_inputs(self):
        if not self.sources:
            raise ValueError("Please add at least one ZIP file, folder, or video before starting.")
        try:
            epochs = int(self.epochs_entry.get().strip())
            if not (1 <= epochs <= 1000):
                raise ValueError
        except ValueError:
            raise ValueError("Epochs must be an integer between 1 and 1000.")

        try:
            ann_conf = float(self.ann_conf_entry.get().strip())
            if not (0.05 <= ann_conf <= 0.95):
                raise ValueError
        except ValueError:
            raise ValueError("Auto-Label Confidence must be between 0.05 and 0.95.")

        try:
            video_fps = float(self.video_fps_entry.get().strip())
            if not (0.1 <= video_fps <= 60.0):
                raise ValueError
        except ValueError:
            raise ValueError("Video Sample FPS must be between 0.1 and 60.0.")

        batch_raw = self.batch_opt.get()
        batch = -1 if batch_raw.startswith("-1") else int(batch_raw)
        imgsz = int(self.imgsz_opt.get())

        free_gb = shutil.disk_usage(APP_ROOT).free / (1024 ** 3)
        if free_gb < 2.0:
            raise RuntimeError(f"Low disk space ({free_gb:.2f} GB free). At least 2.0 GB is required.")

        return {
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "ann_conf": ann_conf,
            "video_fps": video_fps,
            "device": "0",
            "append": self.append_var.get(),
        }

    def _start_training_pipeline(self):
        if self.train_thread and self.train_thread.is_alive():
            messagebox.showwarning("Busy", "A training pipeline is already running.")
            return
        try:
            cfg = self._validate_training_inputs()
        except Exception as e:
            messagebox.showerror("Invalid Settings", str(e))
            return

        self.train_cancel.clear()
        self.btn_start_train.configure(state="disabled")
        self.btn_cancel_train.configure(state="normal")
        self.ui_queue.put(("progress", (0.0, "Starting pipeline...", "0% | Initializing")))

        self.train_thread = threading.Thread(
            target=self._run_pipeline_worker, args=(list(self.sources), cfg), daemon=True
        )
        self.train_thread.start()

    def _cancel_training(self):
        self.train_cancel.set()
        if self.train_subproc and self.train_subproc.poll() is None:
            try:
                self.train_subproc.terminate()
            except Exception:
                pass
        self.log("Cancellation requested by user...", "WARN")
        self.ui_queue.put(("progress", (0.0, "Cancelling...", "Stopping active process")))

    def _run_pipeline_worker(self, sources: list[dict], cfg: dict):
        temp_root = Path(tempfile.mkdtemp(prefix="aimi_ingest_"))
        try:
            # ── Stage 1: Ingest ZIPs, folders, videos ─────────────────
            self.ui_queue.put(("progress", (0.02, "Stage 1/4: Scanning & extracting sources...", "2%")))
            collected_images: list[Path] = []
            collected_videos: list[Path] = []
            skipped_errors = 0

            for src in sources:
                if self.train_cancel.is_set():
                    raise InterruptedError("Cancelled by user.")
                p: Path = src["path"]
                stype = src["type"]
                if not p.exists():
                    self.log(f"Source missing, skipped: {p}", "WARN")
                    skipped_errors += 1
                    continue

                if stype == "zip":
                    self.log(f"Extracting ZIP: {p.name}")
                    try:
                        with zipfile.ZipFile(p, "r") as zf:
                            members = zf.infolist()
                            extract_dir = temp_root / f"zip_{p.stem}"
                            extract_dir.mkdir(parents=True, exist_ok=True)
                            for idx, m in enumerate(members, 1):
                                if self.train_cancel.is_set():
                                    raise InterruptedError("Cancelled by user.")
                                if m.is_dir():
                                    continue
                                # Zip Slip protection
                                target = (extract_dir / m.filename).resolve()
                                if not str(target).startswith(str(extract_dir.resolve())):
                                    self.log(f"Skipped unsafe ZIP path: {m.filename}", "WARN")
                                    continue
                                ext = target.suffix.lower()
                                if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
                                    continue
                                target.parent.mkdir(parents=True, exist_ok=True)
                                with zf.open(m) as src_f, open(target, "wb") as dst_f:
                                    shutil.copyfileobj(src_f, dst_f)
                                if ext in IMAGE_EXTS:
                                    collected_images.append(target)
                                else:
                                    collected_videos.append(target)
                                if idx % 100 == 0:
                                    pct = 0.02 + 0.10 * (idx / max(1, len(members)))
                                    self.ui_queue.put(("progress", (pct, f"Extracting {p.name}...", f"{idx}/{len(members)} entries")))
                    except zipfile.BadZipFile:
                        self.log(f"Corrupt or invalid ZIP archive skipped: {p}", "ERROR")
                        skipped_errors += 1
                    except RuntimeError as e:
                        self.log(f"ZIP error ({p.name} — possibly password-protected): {e}", "ERROR")
                        skipped_errors += 1

                elif stype == "folder":
                    self.log(f"Scanning folder recursively: {p}")
                    for f in p.rglob("*"):
                        if self.train_cancel.is_set():
                            raise InterruptedError("Cancelled by user.")
                        if not f.is_file():
                            continue
                        ext = f.suffix.lower()
                        if ext in IMAGE_EXTS:
                            collected_images.append(f)
                        elif ext in VIDEO_EXTS:
                            collected_videos.append(f)

                elif stype == "video":
                    collected_videos.append(p)

            # Extract frames from videos using OpenCV
            if collected_videos:
                import cv2  # available in .venv or system
                vid_out = temp_root / "video_frames"
                vid_out.mkdir(parents=True, exist_ok=True)
                for v_idx, vp in enumerate(collected_videos, 1):
                    if self.train_cancel.is_set():
                        raise InterruptedError("Cancelled by user.")
                    cap = cv2.VideoCapture(str(vp))
                    if not cap.isOpened():
                        self.log(f"Could not open video (corrupt/unsupported codec): {vp}", "ERROR")
                        skipped_errors += 1
                        continue
                    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    step = max(1, int(round(native_fps / cfg["video_fps"])))
                    frame_i = 0
                    saved_i = 0
                    while True:
                        if self.train_cancel.is_set():
                            cap.release()
                            raise InterruptedError("Cancelled by user.")
                        ok, frame = cap.read()
                        if not ok:
                            break
                        if frame_i % step == 0:
                            out_img = vid_out / f"{vp.stem}_f{saved_i:05d}.jpg"
                            if cv2.imwrite(str(out_img), frame):
                                collected_images.append(out_img)
                                saved_i += 1
                        frame_i += 1
                    cap.release()
                    self.log(f"Extracted {saved_i} frames from {vp.name}")
                    pct = 0.12 + 0.08 * (v_idx / len(collected_videos))
                    self.ui_queue.put(("progress", (pct, f"Extracting video {v_idx}/{len(collected_videos)}", f"{saved_i} frames from {vp.name}")))

            if not collected_images:
                raise RuntimeError("No valid images or video frames were found in the selected sources.")

            self.log(f"Total frames ready for annotation: {len(collected_images)} (skipped errors: {skipped_errors})")

            # ── Stage 2: GPU Auto-Annotation + Train/Val Split ────────
            self.ui_queue.put(("progress", (0.20, "Stage 2/4: Auto-annotating frames...", "20% | Preparing dataset")))
            if not cfg["append"] and DATASET_DIR.exists():
                shutil.rmtree(DATASET_DIR, ignore_errors=True)
            for split in ("train", "val"):
                (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
                (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

            # Write a clean list of image paths for the worker script
            manifest_file = temp_root / "images_list.txt"
            manifest_file.write_text("\n".join(str(p) for p in collected_images), encoding="utf-8")

            py = find_train_python()
            annotate_code = f"""
import os, shutil, sys
from pathlib import Path
from ultralytics import YOLO

imgs = [Path(l) for l in Path({repr(str(manifest_file))}).read_text(encoding='utf-8').splitlines() if l.strip()]
model = YOLO('yolov8n.pt')
dataset = Path({repr(str(DATASET_DIR))})
conf_th = {cfg['ann_conf']}
dev = {repr(cfg['device'])}
total = len(imgs)
persons = heads = bad = 0

for idx, p in enumerate(imgs, 1):
    split = 'val' if (idx % 5 == 0) else 'train'
    try:
        res = model.predict(source=str(p), conf=conf_th, classes=[0], verbose=False, device=dev)[0]
    except Exception as e:
        bad += 1
        print(f'WARN_IMG: {{p.name}}: {{e}}', flush=True)
        continue
    H, W = res.orig_shape
    labels = []
    if res.boxes is not None:
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            bw, bh = x2 - x1, y2 - y1
            labels.append(f'0 {{(x1+bw/2)/W:.6f}} {{(y1+bh/2)/H:.6f}} {{bw/W:.6f}} {{bh/H:.6f}}')
            persons += 1
            hh = bh * 0.20
            if hh >= 8:
                labels.append(f'1 {{(x1+bw/2)/W:.6f}} {{(y1+hh/2)/H:.6f}} {{(bw*0.6)/W:.6f}} {{hh/H:.6f}}')
                heads += 1
    stem = f'u_{{int(os.path.getmtime(p))}}_{{idx:06d}}_{{p.stem[:32]}}'
    shutil.copy2(str(p), str(dataset / 'images' / split / f'{{stem}}.jpg'))
    (dataset / 'labels' / split / f'{{stem}}.txt').write_text('\\n'.join(labels))
    if idx % 20 == 0 or idx == total:
        print(f'ANN_PROG:{{idx}}/{{total}}|{{persons}}|{{heads}}', flush=True)
print(f'ANN_DONE:{{total-bad}}|{{persons}}|{{heads}}|{{bad}}', flush=True)
"""
            self.train_subproc = subprocess.Popen(
                [py, "-c", annotate_code],
                cwd=str(APP_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert self.train_subproc.stdout is not None
            for line in self.train_subproc.stdout:
                if self.train_cancel.is_set():
                    self.train_subproc.terminate()
                    raise InterruptedError("Cancelled by user.")
                line = line.strip()
                if line.startswith("ANN_PROG:"):
                    parts = line.split(":", 1)[1].split("|")
                    done_i, tot_i = map(int, parts[0].split("/"))
                    pct = 0.20 + 0.20 * (done_i / max(1, tot_i))
                    self.ui_queue.put((
                        "progress",
                        (pct, f"Stage 2/4: Auto-annotating ({done_i}/{tot_i})",
                         f"{int(pct*100)}%  |  Persons: {parts[1]}  Heads: {parts[2]}")
                    ))
                elif line:
                    self.log(line)
            rc = self.train_subproc.wait()
            if rc != 0:
                raise RuntimeError(f"Auto-annotation failed with exit code {rc}.")

            # ── Stage 3: GPU Training ─────────────────────────────────
            self.ui_queue.put(("progress", (0.40, "Stage 3/4: Training YOLOv8n on GPU...", "40% | Starting epochs")))
            dataset_yaml = TRAINING_DIR / "dataset.yaml"
            dataset_yaml.write_text(
                f"path: {DATASET_DIR.as_posix()}\ntrain: images/train\nval: images/val\n\nnc: 2\nnames:\n  0: person\n  1: head\n",
                encoding="utf-8"
            )

            train_code = f"""
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(
    data={repr(str(dataset_yaml))},
    epochs={cfg['epochs']},
    batch={cfg['batch']},
    imgsz={cfg['imgsz']},
    patience=20,
    project={repr(str(RUNS_DIR))},
    name='aimi_robrun',
    device={repr(cfg['device'])},
    workers=8,
    cache='disk',
    exist_ok=True,
    save=True,
    plots=True,
    val=True,
)
"""
            self.train_subproc = subprocess.Popen(
                [py, "-c", train_code],
                cwd=str(APP_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            epoch_re = re.compile(r"^\s*(\d+)/(\d+)\s+")
            oom_detected = False
            assert self.train_subproc.stdout is not None
            for line in self.train_subproc.stdout:
                if self.train_cancel.is_set():
                    self.train_subproc.terminate()
                    raise InterruptedError("Cancelled by user.")
                clean = line.strip()
                if "out of memory" in clean.lower():
                    oom_detected = True
                m = epoch_re.match(clean)
                if m:
                    cur_ep, tot_ep = int(m.group(1)), int(m.group(2))
                    pct = 0.40 + 0.50 * (cur_ep / max(1, tot_ep))
                    self.ui_queue.put((
                        "progress",
                        (pct, f"Stage 3/4: Training Epoch {cur_ep}/{tot_ep}", f"{int(pct*100)}%  |  {clean[:70]}")
                    ))
                elif clean:
                    self.log(clean)
            rc = self.train_subproc.wait()
            if rc != 0:
                if oom_detected:
                    raise RuntimeError("CUDA Out of Memory! Lower Batch Size (e.g. 8 or 4) or Image Size and retry.")
                raise RuntimeError(f"Training exited with code {rc}. See System & Logs for details.")

            # ── Stage 4: Export ONNX & Integrate ──────────────────────
            self.ui_queue.put(("progress", (0.92, "Stage 4/4: Exporting ONNX model...", "92% | Integrating into AIMi")))
            export_script = TRAINING_DIR / "export_model.py"
            res = subprocess.run([py, str(export_script)], cwd=str(APP_ROOT), capture_output=True, text=True)
            if res.stdout:
                self.log(res.stdout.strip())
            if res.returncode != 0:
                raise RuntimeError(f"ONNX export failed:\n{res.stderr}")

            self.ui_queue.put(("progress", (1.0, "Complete — Custom model ready!", "100% | Saved to models/aimi-custom.onnx")))
            self.ui_queue.put(("train_done", (True, "Training & ONNX export finished successfully!")))

        except InterruptedError as e:
            self.ui_queue.put(("progress", (0.0, "Cancelled", str(e))))
            self.ui_queue.put(("train_done", (False, "Training cancelled by user.")))
        except Exception as e:
            self.log(f"Pipeline error: {e}\n{traceback.format_exc()}", "ERROR")
            self.ui_queue.put(("progress", (0.0, "Failed — check System & Logs", str(e)[:80])))
            self.ui_queue.put(("train_done", (False, str(e))))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def _on_training_finished(self, success: bool, message: str):
        self.btn_start_train.configure(state="normal")
        self.btn_cancel_train.configure(state="disabled")
        self.train_subproc = None
        self._refresh_model_status()
        if success:
            messagebox.showinfo("Training Complete", message)
        else:
            messagebox.showwarning("Training Stopped", message)

    def _on_close(self):
        self._stop_bot()
        self.train_cancel.set()
        if self.train_subproc and self.train_subproc.poll() is None:
            try:
                self.train_subproc.terminate()
            except Exception:
                pass
        self.destroy()


if __name__ == "__main__":
    app = AIMiStudioApp()
    app.mainloop()
