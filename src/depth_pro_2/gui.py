#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import atexit
import ctypes
import datetime
import gc
import itertools
import logging
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np

MODEL_OPTIONS: tuple[tuple[str, str], ...] = (
    ("depthpro2-s", "depthpro2-s \u2014 Small (24M, 97 MB)"),
    ("depthpro2-b", "depthpro2-b \u2014 Base (97M, 386 MB)"),
    ("depthpro2-l", "depthpro2-l \u2014 Large (334M, 1.3 GB)"),
    ("depthpro2-h", "depthpro2-h \u2014 Huge+ (872M, 3.5 GB)"),
)
MODEL_LABEL_TO_KEY: dict[str, str] = {label: key for key, label in MODEL_OPTIONS}
MODEL_KEY_TO_LABEL: dict[str, str] = {key: label for key, label in MODEL_OPTIONS}
DEFAULT_MODEL_KEY = "depthpro2-h"

DEVICE_CHOICES: tuple[str, ...] = ("Auto", "cuda", "mps", "cpu")
PRECISION_CHOICES: tuple[str, ...] = ("Auto", "bf16", "fp16", "fp32")
VIDEO_MODE_CHOICES: tuple[str, ...] = ("buffered", "unbuffered")
VIDEO_RES_OPTIONS: tuple[tuple[int | None, str], ...] = (
    (None, "Native (full res)"),
    (1080, "1080p (max short 1080)"),
    (720, "720p (max short 720)"),
    (540, "540p (max short 540)"),
    (480, "480p (max short 480)"),
)
VIDEO_RES_LABEL_TO_PX: dict[str, int | None] = {
    label: px for px, label in VIDEO_RES_OPTIONS
}
COLORMAP_CHOICES: tuple[str, ...] = ("turbo", "inferno", "magma", "plasma", "viridis", "gray")
VIEW_MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("side", "Side-by-Side"),
    ("split", "Split Slider"),
    ("depth", "Depth Only"),
    ("input", "Input Only"),
)

VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mov",
    ".mp4",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
    ".mpg",
    ".mpeg",
    ".mts",
)
IMAGE_FILE_PATTERNS = "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff *.heic *.heif *.avif"
VIDEO_FILE_PATTERNS = "*.mp4 *.mov *.m4v *.avi *.mkv *.webm *.mpg *.mpeg *.mts"

PREVIEW_MAX_SIDE = 1280
POLL_INTERVAL_MS = 50
RESIZE_DEBOUNCE_MS = 30
VIDEO_UI_THROTTLE_S = 0.15
CONSOLE_PROGRESS_INTERVAL_S = 2.0
CONSOLE_PROGRESS_STEP_PCT = 5.0
MAX_VIDEO_CACHE_BYTES = 2 * 1024**3

PALETTE: dict[str, str] = {
    "bg": "#181a1f",
    "panel": "#21252b",
    "canvas": "#121418",
    "input_bg": "#15181e",
    "border": "#323842",
    "text": "#e6edf3",
    "muted": "#9da5b4",
    "accent": "#2f81f7",
    "accent_hover": "#3b8bfd",
    "danger": "#d73a49",
    "danger_hover": "#e54857",
    "secondary": "#2d333b",
    "secondary_hover": "#373e47",
}

_COLORMAP_LUT_CACHE: dict[str, np.ndarray] = {}

TOOLTIP_TEXTS: dict[str, str] = {
    "input": (
        "Photo, video, or folder to estimate depth for. Folders batch-process every supported "
        "image and video inside."
    ),
    "output": (
        "Folder where color depth previews (-depth.png / .mp4) and optional .npz or 16-bit "
        "PNG files are saved."
    ),
    "model": (
        "Model size: S is fastest and lightest (97 MB); B/L balance speed and detail; "
        "H+ (3.5 GB) is sharpest but uses the most VRAM."
    ),
    "device": (
        "Hardware to run on. Auto picks CUDA (NVIDIA GPU), then MPS (Apple Silicon), then CPU."
    ),
    "precision": (
        "Numeric precision for inference. Auto uses bf16 on GPU (fastest) and fp32 on CPU. "
        "Use fp16 on older GPUs without bf16."
    ),
    "focal": (
        "Horizontal camera focal length in pixels for metric scale. Leave blank to read EXIF "
        "on photos or auto-estimate FOV."
    ),
    "mode": (
        "Video temporal mode. 'buffered' looks ahead and behind (smoother); 'unbuffered' "
        "predicts frame-by-frame from past frames (still keeps a Window-sized KV cache)."
    ),
    "window": (
        "Frames attended together. Higher is steadier in time but multiplies VRAM usage. "
        "For H+/L on 16 GB GPUs at HD+, use Window 4-8."
    ),
    "overlap": (
        "Frames shared between consecutive clips in buffered mode for smooth transitions. "
        "Must be less than Window."
    ),
    "max_short": (
        "Downscale video frames if their shorter side exceeds this limit. Lowering to 720p "
        "or 540p cuts VRAM use and speeds up videos."
    ),
    "colormap": (
        "Color palette for preview and saved color depth maps. Updates preview immediately "
        "without re-running inference."
    ),
    "append_colormap": (
        "Include the selected colormap name in color output filenames (e.g. "
        "example-depth-inferno.png) so multiple colormaps don't overwrite each other."
    ),
    "save_png16": (
        "Also save 16-bit grayscale inverse depth PNGs (white = near, black = far) for Nuke, "
        "Blender, or compositing."
    ),
    "save_npz": (
        "Also save raw metric depth (meters) and focal length (px) as .npz arrays. Warning: "
        "for videos, writes one large file per frame."
    ),
    "reexport": (
        "Re-save the last estimated image or video using the current Colormap, Save options, "
        "and Output folder without re-running the neural network."
    ),
}

PARAMETER_GUIDE_SECTIONS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Input & Output",
        (
            (
                "Input (image, video, or folder)",
                "Select a single photo, a video file, or a folder for batch processing. When "
                "selecting an image or video, a preview loads automatically.",
            ),
            (
                "Output folder",
                "Destination directory for output files. Photos write <name>-depth.png and "
                "videos write <name>-depth.mp4.",
            ),
        ),
    ),
    (
        "Model Settings",
        (
            (
                "Variant",
                "depthpro2-s (24M, 97 MB): fastest, lowest VRAM.\n"
                "depthpro2-b (97M, 386 MB): great balance for video.\n"
                "depthpro2-l (334M, 1.3 GB): high accuracy.\n"
                "depthpro2-h (872M, 3.5 GB): sharpest details, highest VRAM demand.",
            ),
            (
                "Device",
                "Auto selects CUDA on NVIDIA GPUs, MPS on Apple Silicon, or CPU otherwise. "
                "Override only if you want to force CPU or a specific backend.",
            ),
            (
                "Precision",
                "Auto uses bfloat16 on GPU (recommended) and float32 on CPU. Choose fp16 if "
                "your GPU does not support bfloat16, or fp32 for maximum precision.",
            ),
            (
                "Focal (px)",
                "Horizontal camera focal length in pixels used to convert depth into meters. "
                "Leave blank to read EXIF on photos or run the built-in FOV estimator. "
                "A 10% larger focal length scales all predicted depths by +10%.",
            ),
        ),
    ),
    (
        "Video Options",
        (
            (
                "Mode",
                "buffered (default) processes overlapping clips of Window frames so each frame "
                "sees frames before and after it. unbuffered predicts frames sequentially using "
                "a Window-frame KV cache (lower latency, still uses Window-scaled VRAM).",
            ),
            (
                "Window",
                "Number of frames attended together. Larger windows give steadier temporal "
                "depth but multiply GPU memory usage.",
            ),
            (
                "Overlap",
                "Frames shared between consecutive buffered clips (cross-faded at seams). Must "
                "be strictly smaller than Window. Disabled in unbuffered mode.",
            ),
            (
                "Max short",
                "Caps the shorter side of decoded video frames before inference (e.g. 720p or "
                "540p). Native keeps full resolution for sharpest edges; downscaling greatly "
                "reduces VRAM and speeds up processing (manual Focal px is auto-scaled).",
            ),
        ),
    ),
    (
        "Output & Display",
        (
            (
                "Colormap",
                "Visual palette for colorized depth previews and saved PNG/MP4 files "
                "(turbo is the CLI default). Changing this updates the preview live.",
            ),
            (
                "Append colormap to filename",
                "Adds -<colormap> to colorized output filenames (e.g. photo-depth-inferno.png "
                "or video-depth-magma.mp4) so exporting different palettes keeps each file.",
            ),
            (
                "Save 16-bit PNG (--save-png16)",
                "Saves 16-bit grayscale inverse depth images (white near, black infinitely far) "
                "for VFX/compositing tools. The log prints the meter conversion factor.",
            ),
            (
                "Save metric depth .npz (--save-npz)",
                "Saves raw float32 metric depth (meters) and focallength_px. Note: videos "
                "produce one .npz file per frame, which can use significant disk space.",
            ),
            (
                "Re-export Outputs & Overwrite Prompt",
                "After depth estimation finishes, you can change Colormap, Append colormap, "
                "Save 16-bit PNG, or Output folder and click Re-export Outputs to regenerate the "
                "saved PNG or MP4 without re-running the model (cached up to 2 GiB for videos; "
                "video .npz re-export requires enabling Save metric depth .npz before running). "
                "When an output filename already exists, a prompt asks whether to Overwrite, "
                "Rename (custom or auto-numbered), Add Date & Time, or Cancel.",
            ),
        ),
    ),
    (
        "Preview & Tips",
        (
            (
                "View Modes",
                "Switch between Side-by-Side, interactive Split Slider (drag the white divider "
                "on the canvas), Depth Only, and Input Only.",
            ),
            (
                "Hover Inspection",
                "Move your mouse over any displayed input or depth image to read exact pixel "
                "coordinates and metric depth in meters at the bottom bar.",
            ),
            (
                "VRAM & Speed Tips (16 GB GPUs)",
                "On Windows, exceeding physical VRAM spills silently into shared system RAM, "
                "making clip 1 appear stuck at 0%. Both buffered and unbuffered modes scale "
                "memory with Window. On a 16 GB GPU at HD+, use depthpro2-h with Window=4 "
                "(Overlap=1), lower Max short (e.g. 720p), or switch to depthpro2-b / s.",
            ),
        ),
    ),
)


@dataclass(frozen=True)
class RunConfig:
    """Validated settings for one inference run."""

    input_path: Path
    output_path: Path
    model: str
    device: str
    precision: str
    mode: str
    window: int
    overlap: int
    focal_px: float | None
    colormap: str
    save_png16: bool
    save_npz: bool
    max_short_side: int | None = None
    append_colormap: bool = False
    conflict_policy: str = "overwrite"
    custom_stem: str | None = None


@dataclass(frozen=True)
class _OutputTargets:
    """Concrete file/folder paths written for a single image or video export."""

    color_path: Path
    png16_path: Path | None
    npz_path: Path | None

    def all_paths(self) -> list[Path]:
        """Return all non-None output paths in display order."""
        paths = [self.color_path]
        if self.png16_path is not None:
            paths.append(self.png16_path)
        if self.npz_path is not None:
            paths.append(self.npz_path)
        return paths


@dataclass
class _ExportState:
    """Cached post-inference depth data enabling fast re-export without re-running the model."""

    kind: str
    stem: str
    scale: float
    focal_px: float
    source: str
    p05: float | None
    p95: float | None
    inf_pct: float
    inverse_depth: np.ndarray | None = None
    frame_shape: tuple[int, int] | None = None
    frame_count: int = 0
    fps: float = 30.0
    cache_dir: Path | None = None
    cache_file: Path | None = None
    npz_dir: Path | None = None


_TEMP_CACHE_DIRS: set[Path] = set()


def _register_temp_dir(path: Path) -> None:
    _TEMP_CACHE_DIRS.add(path)


def _remove_temp_dir(path: Path | None) -> None:
    if path is None:
        return
    _TEMP_CACHE_DIRS.discard(path)
    shutil.rmtree(path, ignore_errors=True)


def _cleanup_all_temp_dirs() -> None:
    for path in list(_TEMP_CACHE_DIRS):
        _remove_temp_dir(path)


atexit.register(_cleanup_all_temp_dirs)


class _QueueLogHandler(logging.Handler):
    """Forward logger records to a thread-safe GUI message queue."""

    def __init__(self, msg_queue: queue.Queue[tuple[Any, ...]]):
        super().__init__()
        self.msg_queue = msg_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.msg_queue.put(("log", self.format(record)))
        except Exception:
            self.handleError(record)


class _Tooltip:
    """Lightweight hover tooltip for Tkinter and ttk widgets."""

    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 350):
        """Attach enter/leave/click/destroy bindings to ``widget``."""
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id: str | None = None
        self._popup: tk.Toplevel | None = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")
        widget.bind("<Destroy>", self._on_leave, add="+")

    def _on_enter(self, _event: tk.Event[Any] | None = None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self.show)

    def _on_leave(self, _event: tk.Event[Any] | None = None) -> None:
        self.hide()

    def _cancel(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def show(self) -> None:
        """Display the tooltip near the pointer, clamped to screen bounds."""
        self._after_id = None
        if self._popup is not None or not self.widget.winfo_exists():
            return
        popup = tk.Toplevel(self.widget)
        popup.wm_overrideredirect(True)
        try:
            popup.wm_attributes("-topmost", True)
        except Exception:
            pass
        frame = tk.Frame(
            popup,
            bg="#282d35",
            highlightbackground="#464f5e",
            highlightthickness=1,
        )
        frame.pack(fill="both", expand=True)
        label = tk.Label(
            frame,
            text=self.text,
            bg="#282d35",
            fg=PALETTE["text"],
            justify="left",
            wraplength=310,
            padx=8,
            pady=6,
        )
        label.pack()
        popup.update_idletasks()
        px, py = self.widget.winfo_pointerxy()
        pw = popup.winfo_reqwidth()
        ph = popup.winfo_reqheight()
        sw = popup.winfo_screenwidth()
        sh = popup.winfo_screenheight()
        x = min(max(8, px + 12), max(8, sw - pw - 8))
        y = min(max(8, py + 16), max(8, sh - ph - 8))
        popup.wm_geometry(f"+{x}+{y}")
        self._popup = popup

    def hide(self) -> None:
        """Cancel any pending popup and destroy the active tooltip window."""
        self._cancel()
        if self._popup is not None:
            try:
                self._popup.destroy()
            except Exception:
                pass
            self._popup = None


def get_colormap_lut(cmap_name: str) -> np.ndarray:
    """Return a cached ``(256, 3)`` uint8 RGB lookup table for a Matplotlib colormap."""
    lut = _COLORMAP_LUT_CACHE.get(cmap_name)
    if lut is None:
        import numpy as np
        from matplotlib import colormaps

        lut = (colormaps[cmap_name](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        _COLORMAP_LUT_CACHE[cmap_name] = lut
    return lut


def colorize_with_cmap(
    inverse_depth: np.ndarray, scale: float, cmap_name: str = "turbo"
) -> np.ndarray:
    """Color-map inverse depth from 0 (far) to ``scale`` (near) as RGB uint8."""
    import numpy as np

    lut = get_colormap_lut(cmap_name)
    safe_scale = scale if scale > 0 else 1.0
    indices = np.clip(inverse_depth * (256.0 / safe_scale), 0, 255).astype(np.int32)
    return lut[indices]


def make_preview_rgb(rgb: np.ndarray, max_side: int = PREVIEW_MAX_SIDE) -> np.ndarray:
    """Downscale an RGB uint8 array if its longest side exceeds ``max_side``."""
    height, width = rgb.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return rgb
    import numpy as np
    from PIL import Image

    ratio = max_side / longest
    new_size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    resized = Image.fromarray(rgb).resize(new_size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def resize_short_side(rgb: np.ndarray, max_short_side: int | None) -> tuple[np.ndarray, float]:
    """Downscale an RGB uint8 array if its shorter side exceeds ``max_short_side``."""
    height, width = rgb.shape[:2]
    short = min(height, width)
    if max_short_side is None or short <= max_short_side:
        return rgb, 1.0
    import numpy as np
    from PIL import Image

    ratio = max_short_side / short
    new_w = max(1, round(width * ratio))
    new_h = max(1, round(height * ratio))
    resized = Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8), new_w / width


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as ``MM:SS`` or ``H:MM:SS``."""
    total = max(0, round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_video_progress(
    name: str,
    count: int,
    expected: int,
    elapsed_s: float,
    vram_text: str = "",
) -> tuple[float | None, str]:
    """Format video frame progress percentage, FPS, ETA, and optional VRAM text."""
    fps = count / max(elapsed_s, 1e-6)
    if expected > 0:
        pct = min(100.0, 100.0 * count / expected)
        eta_s = max(0.0, (expected - count) / fps) if count > 0 else 0.0
        text = (
            f"[{name}] {count}/{expected} ({pct:.1f}%) | "
            f"{fps:.2f} fps | ETA {format_duration(eta_s)}{vram_text}"
        )
        return pct, text
    return None, f"[{name}] {count} frames | {fps:.2f} fps{vram_text}"


def _cuda_vram_stats(device: Any) -> tuple[str, float | None]:
    """Return formatted CUDA VRAM usage and fraction used, or empty when unavailable."""
    if getattr(device, "type", None) != "cuda":
        return "", None
    try:
        import torch

        free_b, total_b = torch.cuda.mem_get_info(device)
        if total_b <= 0:
            return "", None
        used_b = total_b - free_b
        gib = 1024**3
        return f" | VRAM {used_b / gib:.1f}/{total_b / gib:.1f} GB", used_b / total_b
    except Exception:
        return "", None


def _cuda_vram_text(device: Any) -> str:
    """Return formatted CUDA VRAM usage suffix, or empty string when unavailable."""
    return _cuda_vram_stats(device)[0]


def format_depth_value(inverse_depth_val: float) -> str:
    """Format an inverse depth value (1/m) as a human-readable metric depth string."""
    if not math.isfinite(inverse_depth_val) or inverse_depth_val <= 0.0:
        return "\u221e (far / sky)"
    depth_m = 1.0 / inverse_depth_val
    if not math.isfinite(depth_m):
        return "\u221e (far / sky)"
    if depth_m >= 100.0:
        return f"{depth_m:.1f} m"
    return f"{depth_m:.2f} m"


def compute_depth_stats(
    counts: np.ndarray, depth_bins: np.ndarray
) -> tuple[float | None, float | None, float]:
    """Compute 5th/95th percentile finite depths in meters and infinite-pixel percentage."""
    import numpy as np

    total = max(1, int(counts.sum()))
    inf_pct = 100.0 * float(counts[-1]) / total
    finite = counts[:-1]
    finite_sum = finite.sum()
    if not finite_sum:
        return None, None, inf_pct
    centers = np.sqrt(depth_bins[:-1] * depth_bins[1:])
    cdf = np.cumsum(finite) / finite_sum
    low, high = centers[np.searchsorted(cdf, (0.05, 0.95))]
    return float(low), float(high), inf_pct


def format_summary_lines(
    written: list[str],
    focal_px: float,
    source: str,
    p05: float | None,
    p95: float | None,
    inf_pct: float,
    png16_scale: float | None,
) -> list[str]:
    """Format CLI-compatible summary lines for an image or video prediction."""
    lines = [
        f"Wrote {', '.join(written)}.",
        f"Focal length: {focal_px:.1f} px, {source}.",
    ]
    depth_range = (
        f"{p05:.2f} m to {p95:.2f} m (5th to 95th percentile), "
        if p05 is not None and p95 is not None
        else ""
    )
    lines.append(f"Depth: {depth_range}{inf_pct:.1f}% infinitely far.")
    if png16_scale is not None:
        k = 65535.0 / png16_scale
        lines.append(f"16-bit PNG: depth in meters = {k:.2f} / value, 0 is infinitely far.")
    return lines


def quantize_u16(inverse_depth: np.ndarray, scale: float) -> np.ndarray:
    """Quantize inverse depth from ``0..scale`` into a 16-bit unsigned integer array."""
    import numpy as np

    safe_scale = scale if scale > 0 else 1.0
    return (np.clip(inverse_depth / safe_scale, 0.0, 1.0) * 65535.0).round().astype(np.uint16)


def inverse_depth_to_meters(inverse_depth: np.ndarray) -> np.ndarray:
    """Convert inverse depth (1/m) to metric depth in meters (inf where <= 0)."""
    import numpy as np

    out = np.full(inverse_depth.shape, np.inf, dtype=np.float32)
    return np.divide(1.0, inverse_depth, out=out, where=inverse_depth > 0)


_INVALID_STEM_CHARS = set('/\\:*?"<>|')


def sanitize_stem(name: str) -> str | None:
    """Validate and clean a user-entered output filename stem, or return ``None`` if invalid."""
    cleaned = name.strip()
    if not cleaned or cleaned in (".", ".."):
        return None
    if any(ch in _INVALID_STEM_CHARS or ord(ch) < 32 for ch in cleaned):
        return None
    return cleaned


def format_timestamp_suffix(now: datetime.datetime | None = None) -> str:
    """Format a filename-safe date/time suffix ``_YYYYMMDD-HHMMSS``."""
    return (now or datetime.datetime.now()).strftime("_%Y%m%d-%H%M%S")


def build_output_paths(
    output_dir: Path,
    stem: str,
    *,
    is_video: bool,
    colormap: str,
    append_colormap: bool,
    save_png16: bool,
    save_npz: bool,
) -> _OutputTargets:
    """Compute output file/directory paths for an image or video export."""
    cmap_tag = f"-{colormap}" if append_colormap else ""
    if is_video:
        color_path = output_dir / f"{stem}-depth{cmap_tag}.mp4"
        png16_path = output_dir / f"{stem}-depth16" if save_png16 else None
        npz_path = output_dir / f"{stem}-depth" if save_npz else None
    else:
        color_path = output_dir / f"{stem}-depth{cmap_tag}.png"
        png16_path = output_dir / f"{stem}-depth16.png" if save_png16 else None
        npz_path = output_dir / f"{stem}-depth.npz" if save_npz else None
    return _OutputTargets(color_path=color_path, png16_path=png16_path, npz_path=npz_path)


def find_existing_outputs(
    targets: _OutputTargets,
    *,
    ignore_paths: set[Path] | None = None,
) -> list[Path]:
    """Return paths in ``targets`` that already exist on disk and are not ignored."""
    resolved_ignored = {p.resolve() for p in ignore_paths} if ignore_paths else set()
    existing: list[Path] = []
    for path in targets.all_paths():
        if path.exists():
            if resolved_ignored and path.resolve() in resolved_ignored:
                continue
            existing.append(path)
    return existing


def next_available_stem(
    output_dir: Path,
    stem: str,
    *,
    is_video: bool,
    colormap: str,
    append_colormap: bool,
    save_png16: bool,
    save_npz: bool,
    base_suffix: str = "",
    ignore_paths: set[Path] | None = None,
) -> str:
    """Return the first stem variant whose output targets do not collide on disk."""
    if base_suffix:
        candidate = f"{stem}{base_suffix}"
        targets = build_output_paths(
            output_dir,
            candidate,
            is_video=is_video,
            colormap=colormap,
            append_colormap=append_colormap,
            save_png16=save_png16,
            save_npz=save_npz,
        )
        if not find_existing_outputs(targets, ignore_paths=ignore_paths):
            return candidate
    prefix = f"{stem}{base_suffix}"
    for idx in itertools.count(2):
        candidate = f"{prefix}_{idx}"
        targets = build_output_paths(
            output_dir,
            candidate,
            is_video=is_video,
            colormap=colormap,
            append_colormap=append_colormap,
            save_png16=save_png16,
            save_npz=save_npz,
        )
        if not find_existing_outputs(targets, ignore_paths=ignore_paths):
            return candidate
    return stem  # pragma: no cover


def reexport_image(
    state: _ExportState,
    output_dir: Path,
    colormap: str,
    save_png16: bool,
    save_npz: bool,
    *,
    append_colormap: bool = False,
    export_stem: str | None = None,
) -> tuple[Path, list[str], np.ndarray]:
    """Re-export cached image depth with updated colormap and output options."""
    import numpy as np
    from PIL import Image

    if state.inverse_depth is None:
        raise RuntimeError("No cached image depth data available for re-export.")
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_stem = export_stem or state.stem
    targets = build_output_paths(
        output_dir,
        effective_stem,
        is_video=False,
        colormap=colormap,
        append_colormap=append_colormap,
        save_png16=save_png16,
        save_npz=save_npz,
    )
    colored = colorize_with_cmap(state.inverse_depth, state.scale, colormap)
    png = targets.color_path
    Image.fromarray(colored).save(png, compress_level=1)
    written = [str(png)]
    if targets.npz_path is not None:
        np.savez(
            targets.npz_path,
            depth=inverse_depth_to_meters(state.inverse_depth),
            focallength_px=np.float32(state.focal_px),
        )
        written.append(str(targets.npz_path))
    if targets.png16_path is not None:
        Image.fromarray(quantize_u16(state.inverse_depth, state.scale)).save(
            targets.png16_path, compress_level=1
        )
        written.append(str(targets.png16_path))
    lines = format_summary_lines(
        written,
        state.focal_px,
        state.source,
        state.p05,
        state.p95,
        state.inf_pct,
        state.scale if save_png16 else None,
    )
    return png, lines, colored


def reexport_video(
    state: _ExportState,
    output_dir: Path,
    colormap: str,
    save_png16: bool,
    save_npz: bool,
    *,
    append_colormap: bool = False,
    export_stem: str | None = None,
    cancel_event: threading.Event,
    on_progress: Callable[[int, int], None],
    on_log: Callable[[str], None],
) -> Path | None:
    """Re-export cached video depth with updated colormap and output options."""
    import numpy as np
    from PIL import Image

    if __package__:
        from .cli import VideoWriter
    else:
        from depth_pro_2.cli import VideoWriter

    use_npz = state.npz_dir is not None and state.npz_dir.is_dir()
    use_cache = (
        not use_npz
        and state.cache_file is not None
        and state.cache_file.exists()
        and state.frame_shape is not None
    )
    if not use_npz and not use_cache:
        raise RuntimeError("No cached video depth data available for re-export.")

    output_dir.mkdir(parents=True, exist_ok=True)
    effective_stem = export_stem or state.stem
    targets = build_output_paths(
        output_dir,
        effective_stem,
        is_video=True,
        colormap=colormap,
        append_colormap=append_colormap,
        save_png16=save_png16,
        save_npz=save_npz,
    )
    video_path = targets.color_path
    tmp_video_path = video_path.with_name(f"{video_path.stem}.tmp.mp4")
    png16_dir = targets.png16_path
    if png16_dir is not None:
        png16_dir.mkdir(exist_ok=True)

    if save_npz and not use_npz:
        on_log(
            "Note: exact unclipped video .npz files require enabling 'Save metric depth .npz' "
            "before running depth estimation; skipped .npz during re-export."
        )

    copied_npz = False
    target_npz_dir = targets.npz_path
    if target_npz_dir is not None and use_npz and state.npz_dir is not None:
        if target_npz_dir.resolve() != state.npz_dir.resolve():
            target_npz_dir.mkdir(exist_ok=True)
            copied_npz = True

    safe_scale = state.scale if state.scale > 0 else 1.0
    inv_factor = np.float32(safe_scale / 65535.0)
    written_frames = 0
    try:
        with VideoWriter(tmp_video_path, state.fps) as writer:
            if use_npz and state.npz_dir is not None:
                for idx in range(state.frame_count):
                    if cancel_event.is_set():
                        break
                    src_npz = state.npz_dir / f"{idx:06d}.npz"
                    with np.load(src_npz) as npz_data:
                        depth = npz_data["depth"].astype(np.float32)
                    inv = np.divide(
                        1.0,
                        depth,
                        out=np.zeros_like(depth, dtype=np.float32),
                        where=np.isfinite(depth) & (depth > 0),
                    )
                    colored = colorize_with_cmap(inv, state.scale, colormap)
                    writer.write(colored)
                    if png16_dir is not None:
                        Image.fromarray(quantize_u16(inv, state.scale)).save(
                            png16_dir / f"{png16_dir.name}.{idx:06d}.png", compress_level=1
                        )
                    if copied_npz and target_npz_dir is not None:
                        shutil.copy2(src_npz, target_npz_dir / f"{idx:06d}.npz")
                    written_frames += 1
                    on_progress(written_frames, state.frame_count)
            else:
                assert state.cache_file is not None and state.frame_shape is not None
                height, width = state.frame_shape
                frame_bytes = height * width * 2
                with open(state.cache_file, "rb") as fh:
                    for idx in range(state.frame_count):
                        if cancel_event.is_set():
                            break
                        raw = fh.read(frame_bytes)
                        if len(raw) != frame_bytes:
                            raise RuntimeError("Incomplete cached video depth frame.")
                        q16 = np.frombuffer(raw, dtype=np.uint16).reshape(height, width)
                        inv = q16.astype(np.float32) * inv_factor
                        colored = colorize_with_cmap(inv, state.scale, colormap)
                        writer.write(colored)
                        if png16_dir is not None:
                            Image.fromarray(q16).save(
                                png16_dir / f"{png16_dir.name}.{idx:06d}.png", compress_level=1
                            )
                        written_frames += 1
                        on_progress(written_frames, state.frame_count)

        if cancel_event.is_set() or written_frames == 0:
            return None
        tmp_video_path.replace(video_path)
    finally:
        if tmp_video_path.exists():
            tmp_video_path.unlink(missing_ok=True)

    written = [f"{written_frames} frames at {state.fps:.5g} fps to {video_path}"]
    if target_npz_dir is not None and use_npz:
        written.append(f"{target_npz_dir}/")
    if png16_dir is not None:
        written.append(f"{png16_dir}/")
    for line in format_summary_lines(
        written,
        state.focal_px,
        state.source,
        state.p05,
        state.p95,
        state.inf_pct,
        state.scale if save_png16 else None,
    ):
        on_log(line)
    return video_path


def map_canvas_to_pixel(
    cx: int,
    cy: int,
    rects: list[tuple[int, int, int, int]],
    width: int,
    height: int,
) -> tuple[int, int] | None:
    """Map canvas coordinates inside any rect ``(x0, y0, w, h)`` to source pixel ``(px, py)``."""
    if width <= 0 or height <= 0:
        return None
    for x0, y0, rw, rh in rects:
        if rw <= 0 or rh <= 0:
            continue
        if x0 <= cx < x0 + rw and y0 <= cy < y0 + rh:
            px = min(width - 1, max(0, int((cx - x0) * width / rw)))
            py = min(height - 1, max(0, int((cy - y0) * height / rh)))
            return px, py
    return None


def open_path_in_os(path: Path) -> None:
    """Open a file or directory using the operating system's default handler."""
    if sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))


def _fit_size(img_w: int, img_h: int, box_w: int, box_h: int) -> tuple[int, int]:
    if img_w <= 0 or img_h <= 0 or box_w <= 0 or box_h <= 0:
        return (1, 1)
    scale = min(box_w / img_w, box_h / img_h)
    return max(1, int(img_w * scale)), max(1, int(img_h * scale))


class DepthPro2App:
    """Native desktop GUI for running Depth Pro 2 on images, videos, and folders."""

    def __init__(self, root: tk.Tk):
        """Initialize window layout, state variables, logging bridge, and background pollers."""
        self.root = root
        self.root.title("Depth Pro 2")
        self.root.geometry("1220x800")
        self.root.minsize(960, 720)
        self.root.configure(bg=PALETTE["bg"])

        self._queue: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self._cancel_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._is_running = False
        self._model_cache: tuple[tuple[str, str, str], Any, Any] | None = None
        self._model_lock = threading.Lock()
        self._last_validation_error: str | None = None

        self._current_input_rgb: np.ndarray | None = None
        self._current_inverse_depth: np.ndarray | None = None
        self._current_depth_scale: float = 1.0
        self._current_color_rgb: np.ndarray | None = None
        self._current_meta: dict[str, Any] = {}
        self._last_output_path: Path | None = None
        self._export_state: _ExportState | None = None

        self._photo_image: Any = None
        self._colorbar_photo: Any = None
        self._render_rects: list[tuple[int, int, int, int]] = []
        self._split_ratio: float = 0.5
        self._resize_after_id: str | None = None
        self._colorbar_after_id: str | None = None
        self._poll_after_id: str | None = None
        self._stage_base_text: str = "Idle"
        self._stage_start_time: float | None = None
        self._last_elapsed_second: int = -1
        self._tooltips: list[_Tooltip] = []
        self._help_window: tk.Toplevel | None = None
        self._empty_message: str = (
            "Select an image, video, or folder to preview and estimate metric depth.\n\n"
            "Tip: hover any setting for a quick explanation, or click '? Help' for the "
            "Parameter Guide."
        )

        self.input_var = tk.StringVar(value="")
        self.output_var = tk.StringVar(value=str(Path.cwd() / "out"))
        self.model_var = tk.StringVar(value=MODEL_KEY_TO_LABEL[DEFAULT_MODEL_KEY])
        self.device_var = tk.StringVar(value="Auto")
        self.precision_var = tk.StringVar(value="Auto")
        self.focal_var = tk.StringVar(value="")
        self.mode_var = tk.StringVar(value="buffered")
        self.window_var = tk.StringVar(value="16")
        self.overlap_var = tk.StringVar(value="4")
        self.max_res_var = tk.StringVar(value=VIDEO_RES_OPTIONS[0][1])
        self.colormap_var = tk.StringVar(value="turbo")
        self.append_colormap_var = tk.BooleanVar(value=False)
        self.save_png16_var = tk.BooleanVar(value=False)
        self.save_npz_var = tk.BooleanVar(value=False)
        self.view_mode_var = tk.StringVar(value="side")

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="Idle")
        self.hover_text_var = tk.StringVar(value="Hover over preview to inspect metric depth")
        self.summary_text_var = tk.StringVar(value="")
        self.near_label_var = tk.StringVar(value="near")
        self.status_left_var = tk.StringVar(value="Device: detecting...")
        self.status_right_var = tk.StringVar(value="Ready")

        self._configure_style()
        self._build_ui()
        self._bind_events()

        self._log_handler = _QueueLogHandler(self._queue)
        self._log_handler.setFormatter(logging.Formatter("%(message)s"))
        logger = logging.getLogger("depth_pro_2")
        logger.setLevel(logging.INFO)
        logger.addHandler(self._log_handler)

        self._sync_mode_state()
        self._update_button_states()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_after_id = self.root.after(POLL_INTERVAL_MS, self._poll_queue)
        threading.Thread(target=self._probe_environment, daemon=True).start()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        self.root.option_add("*TCombobox*Listbox.background", PALETTE["panel"])
        self.root.option_add("*TCombobox*Listbox.foreground", PALETTE["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", PALETTE["accent"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

        base_font = ("Segoe UI", 9) if sys.platform == "win32" else ("TkDefaultFont", 10)
        bold_font = (
            ("Segoe UI", 9, "bold") if sys.platform == "win32" else ("TkDefaultFont", 10, "bold")
        )

        style.configure(
            ".", background=PALETTE["panel"], foreground=PALETTE["text"], font=base_font
        )
        style.configure("App.TFrame", background=PALETTE["bg"])
        style.configure("Panel.TFrame", background=PALETTE["panel"])
        style.configure("Muted.TLabel", background=PALETTE["panel"], foreground=PALETTE["muted"])
        style.configure("Status.TLabel", background=PALETTE["bg"], foreground=PALETTE["muted"])
        style.configure(
            "Section.TLabelframe",
            background=PALETTE["panel"],
            bordercolor=PALETTE["border"],
            lightcolor=PALETTE["border"],
            darkcolor=PALETTE["border"],
            relief="solid",
        )
        style.configure(
            "Section.TLabelframe.Label",
            background=PALETTE["panel"],
            foreground=PALETTE["text"],
            font=bold_font,
        )

        style.configure(
            "TEntry",
            fieldbackground=PALETTE["input_bg"],
            foreground=PALETTE["text"],
            insertcolor=PALETTE["text"],
            bordercolor=PALETTE["border"],
            lightcolor=PALETTE["border"],
            darkcolor=PALETTE["border"],
            padding=(4, 2),
        )
        style.configure(
            "TCombobox",
            fieldbackground=PALETTE["input_bg"],
            background=PALETTE["secondary"],
            foreground=PALETTE["text"],
            arrowcolor=PALETTE["text"],
            bordercolor=PALETTE["border"],
            lightcolor=PALETTE["border"],
            darkcolor=PALETTE["border"],
            padding=(4, 2),
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", PALETTE["input_bg"]), ("disabled", PALETTE["bg"])],
            foreground=[("readonly", PALETTE["text"]), ("disabled", PALETTE["muted"])],
            selectbackground=[("readonly", PALETTE["input_bg"])],
            selectforeground=[("readonly", PALETTE["text"])],
        )
        style.configure(
            "TSpinbox",
            fieldbackground=PALETTE["input_bg"],
            background=PALETTE["secondary"],
            foreground=PALETTE["text"],
            arrowcolor=PALETTE["text"],
            insertcolor=PALETTE["text"],
            bordercolor=PALETTE["border"],
            lightcolor=PALETTE["border"],
            darkcolor=PALETTE["border"],
            padding=(4, 2),
        )
        style.map(
            "TSpinbox",
            fieldbackground=[("disabled", PALETTE["bg"])],
            foreground=[("disabled", PALETTE["muted"])],
        )

        style.configure(
            "Accent.TButton",
            background=PALETTE["accent"],
            foreground="#ffffff",
            bordercolor=PALETTE["accent"],
            lightcolor=PALETTE["accent"],
            darkcolor=PALETTE["accent"],
            font=bold_font,
            padding=(8, 4),
        )
        style.map(
            "Accent.TButton",
            background=[("active", PALETTE["accent_hover"]), ("disabled", "#263244")],
            foreground=[("disabled", PALETTE["muted"])],
        )

        style.configure(
            "Secondary.TButton",
            background=PALETTE["secondary"],
            foreground=PALETTE["text"],
            bordercolor=PALETTE["border"],
            lightcolor=PALETTE["border"],
            darkcolor=PALETTE["border"],
            padding=(6, 2),
        )
        style.map(
            "Secondary.TButton",
            background=[("active", PALETTE["secondary_hover"]), ("disabled", PALETTE["bg"])],
            foreground=[("disabled", PALETTE["muted"])],
        )

        style.configure(
            "Danger.TButton",
            background=PALETTE["danger"],
            foreground="#ffffff",
            bordercolor=PALETTE["danger"],
            lightcolor=PALETTE["danger"],
            darkcolor=PALETTE["danger"],
            padding=(6, 2),
        )
        style.map(
            "Danger.TButton",
            background=[("active", PALETTE["danger_hover"]), ("disabled", PALETTE["secondary"])],
            foreground=[("disabled", PALETTE["muted"])],
        )

        style.configure(
            "TCheckbutton",
            background=PALETTE["panel"],
            foreground=PALETTE["text"],
            indicatorcolor=PALETTE["input_bg"],
            indicatorrelief="flat",
        )
        style.map(
            "TCheckbutton",
            background=[("active", PALETTE["panel"])],
            foreground=[("active", PALETTE["text"]), ("disabled", PALETTE["muted"])],
            indicatorcolor=[("selected", PALETTE["accent"])],
        )

        style.configure(
            "TRadiobutton",
            background=PALETTE["panel"],
            foreground=PALETTE["text"],
            indicatorcolor=PALETTE["input_bg"],
        )
        style.map(
            "TRadiobutton",
            background=[("active", PALETTE["panel"])],
            foreground=[("active", PALETTE["text"]), ("disabled", PALETTE["muted"])],
            indicatorcolor=[("selected", PALETTE["accent"])],
        )

        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=PALETTE["canvas"],
            background=PALETTE["accent"],
            bordercolor=PALETTE["border"],
            lightcolor=PALETTE["accent"],
            darkcolor=PALETTE["accent"],
            thickness=8,
        )
        style.configure("TPanedwindow", background=PALETTE["border"])
        style.configure(
            "Vertical.TScrollbar",
            background=PALETTE["secondary"],
            troughcolor=PALETTE["canvas"],
            bordercolor=PALETTE["border"],
            arrowcolor=PALETTE["text"],
        )

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, style="App.TFrame", padding=8)
        container.pack(fill="both", expand=True)

        body = ttk.Frame(container, style="App.TFrame")
        body.pack(fill="both", expand=True)

        # Left sidebar
        sidebar = ttk.Frame(body, style="Panel.TFrame", width=356, padding=8)
        sidebar.pack(side="left", fill="y", padx=(0, 8))
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        # Right workspace (preview top, logs bottom)
        right = ttk.Frame(body, style="App.TFrame")
        right.pack(side="left", fill="both", expand=True)

        paned = ttk.PanedWindow(right, orient="vertical")
        paned.pack(fill="both", expand=True)

        preview_pane = ttk.Frame(paned, style="Panel.TFrame", padding=8)
        log_pane = ttk.Frame(paned, style="Panel.TFrame", padding=8)
        paned.add(preview_pane, weight=4)
        paned.add(log_pane, weight=1)

        self._build_preview_pane(preview_pane)
        self._build_log_pane(log_pane)

        # Bottom status bar
        status_bar = ttk.Frame(container, style="App.TFrame", padding=(2, 6, 2, 0))
        status_bar.pack(fill="x", side="bottom")
        ttk.Label(status_bar, textvariable=self.status_left_var, style="Status.TLabel").pack(
            side="left"
        )
        ttk.Label(status_bar, textvariable=self.status_right_var, style="Status.TLabel").pack(
            side="right"
        )

    def _add_tooltip(self, widget: tk.Widget, key: str) -> None:
        self._tooltips.append(_Tooltip(widget, TOOLTIP_TEXTS[key]))

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        # 1. Input & Output
        io_frame = ttk.LabelFrame(
            parent, text="Input & Output", style="Section.TLabelframe", padding=(6, 4)
        )
        io_frame.pack(fill="x", pady=(0, 4))

        in_lbl = ttk.Label(io_frame, text="Input (image, video, or folder):")
        in_lbl.pack(anchor="w")
        self.input_entry = ttk.Entry(io_frame, textvariable=self.input_var)
        self.input_entry.pack(fill="x", pady=(1, 3))
        self._add_tooltip(in_lbl, "input")
        self._add_tooltip(self.input_entry, "input")

        browse_row = ttk.Frame(io_frame, style="Panel.TFrame")
        browse_row.pack(fill="x", pady=(0, 3))
        for col in range(3):
            browse_row.columnconfigure(col, weight=1)
        ttk.Button(
            browse_row, text="Image...", style="Secondary.TButton", command=self._browse_image
        ).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(
            browse_row, text="Video...", style="Secondary.TButton", command=self._browse_video
        ).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(
            browse_row, text="Folder...", style="Secondary.TButton", command=self._browse_folder
        ).grid(row=0, column=2, sticky="ew", padx=(3, 0))

        out_lbl = ttk.Label(io_frame, text="Output folder:")
        out_lbl.pack(anchor="w")
        out_row = ttk.Frame(io_frame, style="Panel.TFrame")
        out_row.pack(fill="x", pady=(1, 0))
        self.output_entry = ttk.Entry(out_row, textvariable=self.output_var)
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._add_tooltip(out_lbl, "output")
        self._add_tooltip(self.output_entry, "output")
        ttk.Button(
            out_row, text="Browse...", style="Secondary.TButton", command=self._browse_output
        ).pack(side="right")

        # 2. Model
        model_frame = ttk.LabelFrame(
            parent, text="Model", style="Section.TLabelframe", padding=(6, 4)
        )
        model_frame.pack(fill="x", pady=(0, 4))
        model_frame.columnconfigure(1, weight=1)

        var_lbl = ttk.Label(model_frame, text="Variant:")
        var_lbl.grid(row=0, column=0, sticky="w", pady=1, padx=(0, 8))
        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=[label for _, label in MODEL_OPTIONS],
            state="readonly",
        )
        self.model_combo.grid(row=0, column=1, sticky="ew", pady=1)
        self._add_tooltip(var_lbl, "model")
        self._add_tooltip(self.model_combo, "model")

        dev_lbl = ttk.Label(model_frame, text="Device:")
        dev_lbl.grid(row=1, column=0, sticky="w", pady=1, padx=(0, 8))
        self.device_combo = ttk.Combobox(
            model_frame, textvariable=self.device_var, values=DEVICE_CHOICES, state="readonly"
        )
        self.device_combo.grid(row=1, column=1, sticky="ew", pady=1)
        self._add_tooltip(dev_lbl, "device")
        self._add_tooltip(self.device_combo, "device")

        prec_lbl = ttk.Label(model_frame, text="Precision:")
        prec_lbl.grid(row=2, column=0, sticky="w", pady=1, padx=(0, 8))
        self.precision_combo = ttk.Combobox(
            model_frame,
            textvariable=self.precision_var,
            values=PRECISION_CHOICES,
            state="readonly",
        )
        self.precision_combo.grid(row=2, column=1, sticky="ew", pady=1)
        self._add_tooltip(prec_lbl, "precision")
        self._add_tooltip(self.precision_combo, "precision")

        foc_lbl = ttk.Label(model_frame, text="Focal (px):")
        foc_lbl.grid(row=3, column=0, sticky="w", pady=1, padx=(0, 8))
        self.focal_entry = ttk.Entry(model_frame, textvariable=self.focal_var)
        self.focal_entry.grid(row=3, column=1, sticky="ew", pady=1)
        self._add_tooltip(foc_lbl, "focal")
        self._add_tooltip(self.focal_entry, "focal")
        ttk.Label(
            model_frame, text="Blank = Auto (EXIF / FOV estimator)", style="Muted.TLabel"
        ).grid(row=4, column=1, sticky="w")

        # 3. Video Options
        video_frame = ttk.LabelFrame(
            parent, text="Video Options", style="Section.TLabelframe", padding=(6, 4)
        )
        video_frame.pack(fill="x", pady=(0, 4))
        video_frame.columnconfigure(1, weight=1)

        mode_lbl = ttk.Label(video_frame, text="Mode:")
        mode_lbl.grid(row=0, column=0, sticky="w", pady=1, padx=(0, 8))
        self.mode_combo = ttk.Combobox(
            video_frame, textvariable=self.mode_var, values=VIDEO_MODE_CHOICES, state="readonly"
        )
        self.mode_combo.grid(row=0, column=1, sticky="ew", pady=1)
        self._add_tooltip(mode_lbl, "mode")
        self._add_tooltip(self.mode_combo, "mode")

        win_lbl = ttk.Label(video_frame, text="Window:")
        win_lbl.grid(row=1, column=0, sticky="w", pady=1, padx=(0, 8))
        self.window_spin = ttk.Spinbox(video_frame, from_=1, to=64, textvariable=self.window_var)
        self.window_spin.grid(row=1, column=1, sticky="ew", pady=1)
        self._add_tooltip(win_lbl, "window")
        self._add_tooltip(self.window_spin, "window")

        ov_lbl = ttk.Label(video_frame, text="Overlap:")
        ov_lbl.grid(row=2, column=0, sticky="w", pady=1, padx=(0, 8))
        self.overlap_spin = ttk.Spinbox(video_frame, from_=0, to=63, textvariable=self.overlap_var)
        self.overlap_spin.grid(row=2, column=1, sticky="ew", pady=1)
        self._add_tooltip(ov_lbl, "overlap")
        self._add_tooltip(self.overlap_spin, "overlap")

        res_lbl = ttk.Label(video_frame, text="Max short:")
        res_lbl.grid(row=3, column=0, sticky="w", pady=1, padx=(0, 8))
        self.max_res_combo = ttk.Combobox(
            video_frame,
            textvariable=self.max_res_var,
            values=[label for _, label in VIDEO_RES_OPTIONS],
            state="readonly",
        )
        self.max_res_combo.grid(row=3, column=1, sticky="ew", pady=1)
        self._add_tooltip(res_lbl, "max_short")
        self._add_tooltip(self.max_res_combo, "max_short")

        # 4. Output & Display
        disp_frame = ttk.LabelFrame(
            parent, text="Output & Display", style="Section.TLabelframe", padding=(6, 4)
        )
        disp_frame.pack(fill="x", pady=(0, 4))
        disp_frame.columnconfigure(1, weight=1)

        cmap_lbl = ttk.Label(disp_frame, text="Colormap:")
        cmap_lbl.grid(row=0, column=0, sticky="w", pady=1, padx=(0, 8))
        self.colormap_combo = ttk.Combobox(
            disp_frame, textvariable=self.colormap_var, values=COLORMAP_CHOICES, state="readonly"
        )
        self.colormap_combo.grid(row=0, column=1, sticky="ew", pady=1)
        self._add_tooltip(cmap_lbl, "colormap")
        self._add_tooltip(self.colormap_combo, "colormap")

        append_cmap_chk = ttk.Checkbutton(
            disp_frame,
            text="Append colormap to filename",
            variable=self.append_colormap_var,
        )
        append_cmap_chk.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self._add_tooltip(append_cmap_chk, "append_colormap")

        save_row = ttk.Frame(disp_frame, style="Panel.TFrame")
        save_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(1, 0))
        png16_chk = ttk.Checkbutton(
            save_row,
            text="Save 16-bit PNG",
            variable=self.save_png16_var,
        )
        png16_chk.pack(side="left", padx=(0, 10))
        self._add_tooltip(png16_chk, "save_png16")

        npz_chk = ttk.Checkbutton(
            save_row,
            text="Save metric .npz",
            variable=self.save_npz_var,
        )
        npz_chk.pack(side="left")
        self._add_tooltip(npz_chk, "save_npz")

        self.reexport_btn = ttk.Button(
            disp_frame,
            text="Re-export Outputs",
            style="Secondary.TButton",
            command=self._start_reexport,
        )
        self.reexport_btn.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(3, 0))
        self._add_tooltip(self.reexport_btn, "reexport")

        # 5. Actions & Progress
        action_frame = ttk.Frame(parent, style="Panel.TFrame", padding=(0, 2, 0, 0))
        action_frame.pack(fill="x", side="bottom")

        self.run_btn = ttk.Button(
            action_frame,
            text="Run Depth Estimation",
            style="Accent.TButton",
            command=self._start_run,
        )
        self.run_btn.pack(fill="x", pady=(0, 3))

        sub_actions = ttk.Frame(action_frame, style="Panel.TFrame")
        sub_actions.pack(fill="x", pady=(0, 4))
        sub_actions.columnconfigure(0, weight=1)
        sub_actions.columnconfigure(1, weight=1)

        self.cancel_btn = ttk.Button(
            sub_actions, text="Cancel", style="Danger.TButton", command=self._cancel_run
        )
        self.cancel_btn.grid(row=0, column=0, sticky="ew", padx=(0, 3))

        self.unload_btn = ttk.Button(
            sub_actions, text="Unload Model", style="Secondary.TButton", command=self._unload_model
        )
        self.unload_btn.grid(row=0, column=1, sticky="ew", padx=(3, 0))

        self.progress_bar = ttk.Progressbar(
            action_frame,
            mode="determinate",
            maximum=100.0,
            variable=self.progress_var,
            style="Horizontal.TProgressbar",
        )
        self.progress_bar.pack(fill="x", pady=(0, 3))
        ttk.Label(
            action_frame, textvariable=self.progress_text_var, style="Muted.TLabel"
        ).pack(anchor="w")

    def _build_preview_pane(self, parent: ttk.Frame) -> None:
        toolbar = ttk.Frame(parent, style="Panel.TFrame")
        toolbar.pack(fill="x", pady=(0, 6))

        for mode_val, label in VIEW_MODE_OPTIONS:
            ttk.Radiobutton(
                toolbar,
                text=label,
                value=mode_val,
                variable=self.view_mode_var,
                command=self._redraw_canvas,
            ).pack(side="left", padx=(0, 10))

        self.help_btn = ttk.Button(
            toolbar, text="? Help", style="Secondary.TButton", command=self._show_help_dialog
        )
        self.help_btn.pack(side="right", padx=(6, 0))

        self.open_folder_btn = ttk.Button(
            toolbar, text="Open Folder", style="Secondary.TButton", command=self._open_output_folder
        )
        self.open_folder_btn.pack(side="right")

        self.open_output_btn = ttk.Button(
            toolbar, text="Open Output", style="Secondary.TButton", command=self._open_last_output
        )
        self.open_output_btn.pack(side="right", padx=(0, 6))

        self.canvas = tk.Canvas(
            parent,
            bg=PALETTE["canvas"],
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
        )
        self.canvas.pack(fill="both", expand=True)

        # Colorbar legend
        legend_row = ttk.Frame(parent, style="Panel.TFrame", padding=(0, 6, 0, 2))
        legend_row.pack(fill="x")
        ttk.Label(legend_row, text="\u221e (far)", style="Muted.TLabel").pack(side="left")
        self.colorbar_canvas = tk.Canvas(
            legend_row,
            height=14,
            bg=PALETTE["panel"],
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
        )
        self.colorbar_canvas.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Label(legend_row, textvariable=self.near_label_var, style="Muted.TLabel").pack(
            side="right"
        )

        # Metric inspector row
        inspector_row = ttk.Frame(parent, style="Panel.TFrame", padding=(0, 2, 0, 0))
        inspector_row.pack(fill="x")
        ttk.Label(inspector_row, textvariable=self.hover_text_var).pack(side="left")
        ttk.Label(inspector_row, textvariable=self.summary_text_var, style="Muted.TLabel").pack(
            side="right"
        )

    def _build_log_pane(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent, style="Panel.TFrame")
        header.pack(fill="x", pady=(0, 4))
        ttk.Label(header, text="Log", style="Muted.TLabel").pack(side="left")
        ttk.Button(
            header, text="Clear", style="Secondary.TButton", command=self._clear_log
        ).pack(side="right")

        text_frame = ttk.Frame(parent, style="Panel.TFrame")
        text_frame.pack(fill="both", expand=True)

        mono_font = ("Consolas", 9) if sys.platform == "win32" else ("TkFixedFont", 10)
        self.log_text = tk.Text(
            text_frame,
            height=6,
            wrap="word",
            state="disabled",
            bg=PALETTE["canvas"],
            fg=PALETTE["text"],
            insertbackground=PALETTE["text"],
            selectbackground=PALETTE["accent"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
            font=mono_font,
            padx=6,
            pady=4,
        )
        scroll = ttk.Scrollbar(
            text_frame, orient="vertical", command=self.log_text.yview, style="Vertical.TScrollbar"
        )
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _bind_events(self) -> None:
        self.mode_var.trace_add("write", lambda *_: self._sync_mode_state())
        self.colormap_var.trace_add("write", lambda *_: self._on_colormap_changed())
        self.input_entry.bind("<Return>", lambda _e: self._on_input_entered())

        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", lambda _e: self._reset_hover_text())
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_drag)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.colorbar_canvas.bind("<Configure>", self._on_colorbar_configure)

    def _probe_environment(self) -> None:
        try:
            import torch

            from .utils import default_device

            dev = default_device()
            if dev.type == "cuda":
                name = torch.cuda.get_device_name(0)
                text = f"Detected device: cuda ({name})"
            elif dev.type == "mps":
                text = "Detected device: mps (Apple Silicon)"
            else:
                text = "Detected device: cpu"
        except Exception as exc:
            text = (
                f"Environment incomplete ({type(exc).__name__}); "
                "launch via uv run depth-pro-2 gui"
            )
        self._queue.put(("env_status", text))

    def _initial_browse_dir(self) -> str:
        text = self.input_var.get().strip()
        if text:
            p = Path(text)
            if p.is_dir():
                return str(p)
            if p.parent.is_dir():
                return str(p.parent)
        assets_dir = Path(__file__).resolve().parents[2] / "assets"
        if assets_dir.is_dir():
            return str(assets_dir)
        return str(Path.cwd())

    def _browse_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select Input Image",
            initialdir=self._initial_browse_dir(),
            filetypes=[("Image files", IMAGE_FILE_PATTERNS), ("All files", "*.*")],
        )
        if selected:
            self.input_var.set(selected)
            self._load_input_preview(Path(selected))

    def _browse_video(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select Input Video",
            initialdir=self._initial_browse_dir(),
            filetypes=[("Video files", VIDEO_FILE_PATTERNS), ("All files", "*.*")],
        )
        if selected:
            self.input_var.set(selected)
            self._load_input_preview(Path(selected))

    def _browse_folder(self) -> None:
        selected = filedialog.askdirectory(
            title="Select Input Folder",
            initialdir=self._initial_browse_dir(),
        )
        if selected:
            self.input_var.set(selected)
            self._load_input_preview(Path(selected))

    def _browse_output(self) -> None:
        current = self.output_var.get().strip()
        initial = current if current and Path(current).is_dir() else str(Path.cwd())
        selected = filedialog.askdirectory(title="Select Output Folder", initialdir=initial)
        if selected:
            self.output_var.set(selected)

    def _on_input_entered(self) -> None:
        text = self.input_var.get().strip()
        if text:
            self._load_input_preview(Path(text))

    def _load_input_preview(self, path: Path) -> None:
        self._set_export_state(None)
        self._current_inverse_depth = None
        self._current_color_rgb = None
        self._current_meta = {}
        self.near_label_var.set("near")
        self.summary_text_var.set("")
        self._reset_hover_text()

        if not path.exists():
            self._current_input_rgb = None
            self._empty_message = f"Input path not found:\n{path}"
            self._redraw_canvas()
            return

        if path.is_dir():
            self._current_input_rgb = None
            try:
                from .utils import get_supported_image_extensions

                img_exts = set(get_supported_image_extensions())
            except ImportError:
                img_exts = {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                    ".bmp",
                    ".tif",
                    ".tiff",
                    ".heic",
                    ".heif",
                    ".avif",
                }
            vid_exts = set(VIDEO_EXTENSIONS)
            files = [p for p in path.iterdir() if p.is_file() and not p.name.startswith(".")]
            n_img = sum(1 for p in files if p.suffix.lower() in img_exts)
            n_vid = sum(1 for p in files if p.suffix.lower() in vid_exts)
            self._empty_message = (
                f"Folder: {path.name}\n"
                f"{n_img} image(s), {n_vid} video(s) found.\n\n"
                "Click 'Run Depth Estimation' to process this folder."
            )
            self._append_log(f"Selected folder {path}: {n_img} image(s), {n_vid} video(s).")
            self._redraw_canvas()
            return

        try:
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                import av

                from .utils import upright

                with av.open(str(path)) as container:
                    if not container.streams.video:
                        raise ValueError(f"No video stream found in {path.name}.")
                    frame = next(container.decode(video=0), None)
                    if frame is None:
                        raise ValueError(f"No frames decoded from {path.name}.")
                    rgb = upright(frame)
                self._current_input_rgb = make_preview_rgb(rgb)
                self._append_log(
                    f"Loaded video preview: {path.name} ({rgb.shape[1]}x{rgb.shape[0]})."
                )
            else:
                from .utils import load_rgb

                rgb, exif_f_px = load_rgb(path)
                self._current_input_rgb = make_preview_rgb(rgb)
                exif_msg = (
                    f"EXIF focal={exif_f_px:.1f} px" if exif_f_px is not None else "no EXIF focal"
                )
                self._append_log(
                    f"Loaded image: {path.name} ({rgb.shape[1]}x{rgb.shape[0]}, {exif_msg})."
                )
        except ImportError as exc:
            self._current_input_rgb = None
            self._empty_message = (
                f"Preview unavailable ({exc.name} not installed).\n"
                "Run with: uv run depth-pro-2 gui"
            )
            self._append_log(f"Missing dependency for preview: {exc}")
        except Exception as exc:
            self._current_input_rgb = None
            self._empty_message = f"Could not load preview:\n{exc}"
            self._append_log(f"Preview error for {path}: {exc}")

        self._redraw_canvas()

    def _sync_mode_state(self) -> None:
        if self._is_running:
            self.overlap_spin.configure(state="disabled")
            return
        state = "disabled" if self.mode_var.get() == "unbuffered" else "normal"
        self.overlap_spin.configure(state=state)

    def _on_colormap_changed(self) -> None:
        self._redraw_colorbar()
        if self._current_inverse_depth is not None:
            try:
                colored = colorize_with_cmap(
                    self._current_inverse_depth,
                    self._current_depth_scale,
                    self.colormap_var.get(),
                )
                self._current_color_rgb = make_preview_rgb(colored)
                self._redraw_canvas()
            except Exception as exc:
                self._append_log(f"Colormap update failed: {exc}")

    def _build_run_config(self, show_errors: bool = True) -> RunConfig | None:
        self._last_validation_error = None

        def fail(msg: str) -> None:
            self._last_validation_error = msg
            if show_errors:
                messagebox.showerror("Invalid Settings", msg)

        input_text = self.input_var.get().strip()
        if not input_text:
            fail("Please select an input image, video, or folder.")
            return None
        input_path = Path(input_text)
        if not input_path.exists():
            fail(f"Input path does not exist:\n{input_path}")
            return None

        output_text = self.output_var.get().strip()
        if not output_text:
            fail("Please select an output folder.")
            return None
        output_path = Path(output_text)

        mode = self.mode_var.get()
        try:
            window = int(self.window_var.get().strip())
        except ValueError:
            fail("Window must be an integer >= 1.")
            return None
        if window < 1:
            fail("Window must be at least 1.")
            return None

        try:
            overlap = int(self.overlap_var.get().strip())
        except ValueError:
            fail("Overlap must be an integer >= 0.")
            return None
        if overlap < 0:
            fail("Overlap must be at least 0.")
            return None

        if mode == "buffered" and overlap >= window:
            fail(f"Overlap ({overlap}) must be less than Window ({window}) in buffered mode.")
            return None

        focal_text = self.focal_var.get().strip()
        focal_px: float | None = None
        if focal_text:
            try:
                focal_px = float(focal_text)
            except ValueError:
                fail("Focal length must be a positive number in pixels (or blank for Auto).")
                return None
            if not math.isfinite(focal_px) or focal_px <= 0.0:
                fail("Focal length must be a finite positive number in pixels.")
                return None

        model_key = MODEL_LABEL_TO_KEY.get(self.model_var.get(), self.model_var.get())
        max_short_side = VIDEO_RES_LABEL_TO_PX.get(self.max_res_var.get())
        return RunConfig(
            input_path=input_path,
            output_path=output_path,
            model=model_key,
            device=self.device_var.get(),
            precision=self.precision_var.get(),
            mode=mode,
            window=window,
            overlap=overlap,
            focal_px=focal_px,
            colormap=self.colormap_var.get(),
            save_png16=self.save_png16_var.get(),
            save_npz=self.save_npz_var.get(),
            max_short_side=max_short_side,
            append_colormap=self.append_colormap_var.get(),
        )

    def _collect_conflict_check_inputs(self, input_path: Path, output_path: Path) -> list[Path]:
        if input_path.is_file():
            return [input_path]
        if not input_path.is_dir():
            return []
        try:
            from .utils import get_supported_image_extensions

            img_exts = set(get_supported_image_extensions())
        except ImportError:
            img_exts = {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
                ".bmp",
                ".tif",
                ".tiff",
                ".heic",
                ".heif",
                ".avif",
            }
        supported = set(VIDEO_EXTENSIONS) | img_exts
        found = sorted(
            p
            for p in input_path.iterdir()
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in supported
        )
        if input_path.resolve() == output_path.resolve():
            outputs = {f"{p.stem}-depth{tag}" for p in found for tag in ("", "16")}
            found = [p for p in found if p.stem not in outputs]
        return found

    def _prompt_export_conflict(
        self,
        *,
        output_dir: Path,
        existing_paths: list[Path],
        is_batch: bool,
        default_rename_stem: str | None = None,
        preview_builder: Callable[[str], str] | None = None,
        conflict_checker: Callable[[str], list[Path]] | None = None,
        timestamp_preview: str | None = None,
    ) -> tuple[str, str | None]:
        result: tuple[str, str | None] = ("cancel", None)

        dlg = tk.Toplevel(self.root)
        dlg.title("Output Already Exists")
        dlg.configure(bg=PALETTE["bg"])
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()

        body = ttk.Frame(dlg, style="Panel.TFrame", padding=14)
        body.pack(fill="both", expand=True)

        ttk.Label(
            body,
            text=f"Existing output(s) found in {output_dir}:",
        ).pack(anchor="w")

        shown = [f"\u2022 {p.name}" for p in existing_paths[:6]]
        if len(existing_paths) > 6:
            shown.append(f"... and {len(existing_paths) - 6} more.")
        ttk.Label(
            body,
            text="\n".join(shown),
            style="Muted.TLabel",
            justify="left",
        ).pack(anchor="w", pady=(4, 8))

        rename_var = tk.StringVar(value=default_rename_stem or "")
        preview_var = tk.StringVar(value="")
        err_var = tk.StringVar(value="")

        if not is_batch:
            ttk.Label(body, text="New base name (for Rename):").pack(anchor="w")
            rename_entry = ttk.Entry(body, textvariable=rename_var, width=44)
            rename_entry.pack(fill="x", pady=(2, 2))
            rename_entry.select_range(0, "end")
            rename_entry.focus_set()

            def update_preview(*_args: Any) -> None:
                err_var.set("")
                cleaned = sanitize_stem(rename_var.get())
                if cleaned is None:
                    preview_var.set("Enter a valid filename stem.")
                elif preview_builder is not None:
                    preview_var.set(f"Will save as: {preview_builder(cleaned)}")

            rename_var.trace_add("write", update_preview)
            update_preview()
            ttk.Label(body, textvariable=preview_var, style="Muted.TLabel").pack(anchor="w")
            err_lbl = tk.Label(
                body,
                textvariable=err_var,
                bg=PALETTE["panel"],
                fg="#ff7b72",
                anchor="w",
            )
            err_lbl.pack(fill="x", pady=(1, 4))

        if timestamp_preview:
            ttk.Label(
                body,
                text=f"Date & time example: {timestamp_preview}",
                style="Muted.TLabel",
            ).pack(anchor="w", pady=(0, 8))

        def on_overwrite() -> None:
            nonlocal result
            result = ("overwrite", None)
            dlg.destroy()

        def on_rename() -> None:
            nonlocal result
            if is_batch:
                result = ("auto_rename", None)
                dlg.destroy()
                return
            cleaned = sanitize_stem(rename_var.get())
            if cleaned is None:
                err_var.set('Invalid name (cannot be empty or contain / \\ : * ? " < > |).')
                return
            if conflict_checker is not None:
                collisions = conflict_checker(cleaned)
                if collisions:
                    err_var.set(f"Already exists: {collisions[0].name}")
                    return
            result = ("rename", cleaned)
            dlg.destroy()

        def on_timestamp() -> None:
            nonlocal result
            result = ("timestamp", None)
            dlg.destroy()

        def on_cancel() -> None:
            nonlocal result
            result = ("cancel", None)
            dlg.destroy()

        btn_row = ttk.Frame(body, style="Panel.TFrame", padding=(0, 4, 0, 0))
        btn_row.pack(fill="x")
        ttk.Button(
            btn_row, text="Overwrite", style="Danger.TButton", command=on_overwrite
        ).pack(side="left", padx=(0, 6))
        rename_label = "Auto-Rename" if is_batch else "Rename"
        ttk.Button(
            btn_row, text=rename_label, style="Accent.TButton", command=on_rename
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            btn_row, text="Add Date & Time", style="Secondary.TButton", command=on_timestamp
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            btn_row, text="Cancel", style="Secondary.TButton", command=on_cancel
        ).pack(side="right")

        dlg.protocol("WM_DELETE_WINDOW", on_cancel)
        dlg.bind("<Escape>", lambda _e: on_cancel())

        dlg.update_idletasks()
        rw = self.root.winfo_width()
        rh = self.root.winfo_height()
        rx = self.root.winfo_rootx()
        ry = self.root.winfo_rooty()
        dw = dlg.winfo_reqwidth()
        dh = dlg.winfo_reqheight()
        x = rx + max(0, (rw - dw) // 2)
        y = ry + max(0, (rh - dh) // 2)
        dlg.geometry(f"+{x}+{y}")
        self.root.wait_window(dlg)
        return result

    def _start_run(self) -> None:
        if self._is_running:
            return
        config = self._build_run_config(show_errors=True)
        if config is None:
            return

        check_paths = self._collect_conflict_check_inputs(config.input_path, config.output_path)
        existing_all: list[Path] = []
        for p in check_paths:
            is_vid = p.suffix.lower() in VIDEO_EXTENSIONS
            targets = build_output_paths(
                config.output_path,
                p.stem,
                is_video=is_vid,
                colormap=config.colormap,
                append_colormap=config.append_colormap,
                save_png16=config.save_png16,
                save_npz=config.save_npz,
            )
            existing_all.extend(find_existing_outputs(targets))

        if existing_all:
            is_batch = len(check_paths) > 1 or config.input_path.is_dir()
            if is_batch:
                action, _ = self._prompt_export_conflict(
                    output_dir=config.output_path,
                    existing_paths=existing_all,
                    is_batch=True,
                )
                if action == "cancel":
                    return
                if action in ("auto_rename", "timestamp", "overwrite"):
                    config = replace(config, conflict_policy=action)
            else:
                p0 = check_paths[0]
                is_vid = p0.suffix.lower() in VIDEO_EXTENSIONS
                default_rename = next_available_stem(
                    config.output_path,
                    p0.stem,
                    is_video=is_vid,
                    colormap=config.colormap,
                    append_colormap=config.append_colormap,
                    save_png16=config.save_png16,
                    save_npz=config.save_npz,
                )
                ts_stem = f"{p0.stem}{format_timestamp_suffix()}"
                ts_preview = build_output_paths(
                    config.output_path,
                    ts_stem,
                    is_video=is_vid,
                    colormap=config.colormap,
                    append_colormap=config.append_colormap,
                    save_png16=config.save_png16,
                    save_npz=config.save_npz,
                ).color_path.name

                def preview_fn(stem_val: str) -> str:
                    return build_output_paths(
                        config.output_path,
                        stem_val,
                        is_video=is_vid,
                        colormap=config.colormap,
                        append_colormap=config.append_colormap,
                        save_png16=config.save_png16,
                        save_npz=config.save_npz,
                    ).color_path.name

                def check_fn(stem_val: str) -> list[Path]:
                    t = build_output_paths(
                        config.output_path,
                        stem_val,
                        is_video=is_vid,
                        colormap=config.colormap,
                        append_colormap=config.append_colormap,
                        save_png16=config.save_png16,
                        save_npz=config.save_npz,
                    )
                    return find_existing_outputs(t)

                action, chosen_stem = self._prompt_export_conflict(
                    output_dir=config.output_path,
                    existing_paths=existing_all,
                    is_batch=False,
                    default_rename_stem=default_rename,
                    preview_builder=preview_fn,
                    conflict_checker=check_fn,
                    timestamp_preview=ts_preview,
                )
                if action == "cancel":
                    return
                if action == "rename" and chosen_stem is not None:
                    config = replace(config, conflict_policy="overwrite", custom_stem=chosen_stem)
                elif action == "timestamp":
                    config = replace(config, conflict_policy="timestamp", custom_stem=None)
                else:
                    config = replace(config, conflict_policy="overwrite", custom_stem=None)

        self._set_export_state(None)
        self._cancel_event.clear()
        self._is_running = True
        self._set_indeterminate(False)
        self._stage_base_text = "Starting..."
        self._stage_start_time = time.monotonic()
        self._last_elapsed_second = 0
        self.progress_var.set(0.0)
        self.progress_text_var.set("Starting... (00:00)")
        self.status_right_var.set("Starting... (00:00)")
        self._update_button_states()

        self._worker_thread = threading.Thread(
            target=self._run_worker, args=(config,), daemon=True
        )
        self._worker_thread.start()

    def _start_reexport(self) -> None:
        if self._is_running or self._export_state is None:
            return
        output_text = self.output_var.get().strip()
        if not output_text:
            messagebox.showerror("Invalid Settings", "Please select an output folder.")
            return
        output_dir = Path(output_text)
        state = self._export_state
        colormap = self.colormap_var.get()
        append_colormap = self.append_colormap_var.get()
        save_png16 = self.save_png16_var.get()
        save_npz = self.save_npz_var.get()
        is_video = state.kind == "video"
        ignore_paths = {state.npz_dir} if (is_video and state.npz_dir is not None) else set()

        targets = build_output_paths(
            output_dir,
            state.stem,
            is_video=is_video,
            colormap=colormap,
            append_colormap=append_colormap,
            save_png16=save_png16,
            save_npz=save_npz,
        )
        existing = find_existing_outputs(targets, ignore_paths=ignore_paths)
        export_stem = state.stem
        if existing:
            default_rename = next_available_stem(
                output_dir,
                state.stem,
                is_video=is_video,
                colormap=colormap,
                append_colormap=append_colormap,
                save_png16=save_png16,
                save_npz=save_npz,
                ignore_paths=ignore_paths,
            )
            ts_stem = f"{state.stem}{format_timestamp_suffix()}"
            ts_preview = build_output_paths(
                output_dir,
                ts_stem,
                is_video=is_video,
                colormap=colormap,
                append_colormap=append_colormap,
                save_png16=save_png16,
                save_npz=save_npz,
            ).color_path.name

            def preview_fn(stem_val: str) -> str:
                return build_output_paths(
                    output_dir,
                    stem_val,
                    is_video=is_video,
                    colormap=colormap,
                    append_colormap=append_colormap,
                    save_png16=save_png16,
                    save_npz=save_npz,
                ).color_path.name

            def check_fn(stem_val: str) -> list[Path]:
                t = build_output_paths(
                    output_dir,
                    stem_val,
                    is_video=is_video,
                    colormap=colormap,
                    append_colormap=append_colormap,
                    save_png16=save_png16,
                    save_npz=save_npz,
                )
                return find_existing_outputs(t, ignore_paths=ignore_paths)

            action, chosen_stem = self._prompt_export_conflict(
                output_dir=output_dir,
                existing_paths=existing,
                is_batch=False,
                default_rename_stem=default_rename,
                preview_builder=preview_fn,
                conflict_checker=check_fn,
                timestamp_preview=ts_preview,
            )
            if action == "cancel":
                return
            if action == "rename" and chosen_stem is not None:
                export_stem = chosen_stem
            elif action == "timestamp":
                export_stem = next_available_stem(
                    output_dir,
                    state.stem,
                    is_video=is_video,
                    colormap=colormap,
                    append_colormap=append_colormap,
                    save_png16=save_png16,
                    save_npz=save_npz,
                    base_suffix=format_timestamp_suffix(),
                    ignore_paths=ignore_paths,
                )

        self._cancel_event.clear()
        self._is_running = True
        self._set_indeterminate(False)
        self._stage_base_text = "Re-exporting outputs..."
        self._stage_start_time = time.monotonic()
        self._last_elapsed_second = 0
        self.progress_var.set(0.0)
        self.progress_text_var.set("Re-exporting outputs... (00:00)")
        self.status_right_var.set("Re-exporting outputs... (00:00)")
        self._update_button_states()

        self._worker_thread = threading.Thread(
            target=self._reexport_worker,
            args=(state, output_dir, colormap, append_colormap, save_png16, save_npz, export_stem),
            daemon=True,
        )
        self._worker_thread.start()

    def _reexport_worker(
        self,
        state: _ExportState,
        output_dir: Path,
        colormap: str,
        append_colormap: bool,
        save_png16: bool,
        save_npz: bool,
        export_stem: str,
    ) -> None:
        def emit_log(msg: str) -> None:
            print(msg, flush=True)
            self._queue.put(("log", msg))

        try:
            emit_log(f"Re-exporting {export_stem} with colormap={colormap}...")
            if state.kind == "image":
                out_path, lines, colored = reexport_image(
                    state,
                    output_dir,
                    colormap,
                    save_png16,
                    save_npz,
                    append_colormap=append_colormap,
                    export_stem=export_stem,
                )
                for line in lines:
                    emit_log(line)
                self._queue.put(("reexport_preview", make_preview_rgb(colored), out_path))
                self._queue.put(("done", False, out_path))
                return

            start_t = time.monotonic()
            last_log_t = 0.0
            last_logged_pct = -CONSOLE_PROGRESS_STEP_PCT

            def on_progress(done_frames: int, total_frames: int) -> None:
                nonlocal last_log_t, last_logged_pct
                now = time.monotonic()
                pct, line = format_video_progress(
                    f"{export_stem} (re-export)", done_frames, total_frames, now - start_t
                )
                if pct is not None:
                    self._queue.put(("progress_value", pct, line))
                should_log = (
                    done_frames == 1
                    or (total_frames > 0 and done_frames == total_frames)
                    or (now - last_log_t >= CONSOLE_PROGRESS_INTERVAL_S)
                    or (pct is not None and pct - last_logged_pct >= CONSOLE_PROGRESS_STEP_PCT)
                )
                if should_log:
                    last_log_t = now
                    if pct is not None:
                        last_logged_pct = pct
                    emit_log(line)

            out_path = reexport_video(
                state,
                output_dir,
                colormap,
                save_png16,
                save_npz,
                append_colormap=append_colormap,
                export_stem=export_stem,
                cancel_event=self._cancel_event,
                on_progress=on_progress,
                on_log=emit_log,
            )
            self._queue.put(("done", self._cancel_event.is_set(), out_path))
        except Exception as exc:
            self._queue.put(("error", str(exc)))

    def _cancel_run(self) -> None:
        if not self._is_running:
            return
        self._cancel_event.set()
        self._stage_base_text = "Cancelling..."
        self._stage_start_time = time.monotonic()
        self._last_elapsed_second = 0
        self.progress_text_var.set("Cancelling...")
        self.status_right_var.set("Cancelling")
        self._append_log("Cancellation requested; finishing current step...")

    def _unload_model(self) -> None:
        if self._is_running:
            return
        with self._model_lock:
            had_model = self._model_cache is not None
            self._model_cache = None
        if had_model:
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self._append_log("Unloaded cached model and released memory.")
            self.status_right_var.set("Model unloaded")
        self._update_button_states()

    def _run_worker(self, config: RunConfig) -> None:
        try:
            import av
            import numpy as np
            import torch
            from PIL import Image

            from .cli import (
                DEPTH_BINS,
                PRECISIONS,
                VideoWriter,
                collect_inputs,
                color_scale,
                count_depth,
                grayscale16,
                inverse_metric_depth,
                write_npz,
            )
            from .model import create_model_and_transforms
            from .utils import default_device, load_rgb, upright
        except ImportError as exc:
            self._queue.put(
                (
                    "error",
                    f"Required package not installed: {exc}.\n"
                    "Please launch via: uv run depth-pro-2 gui",
                )
            )
            return

        def emit_log(msg: str) -> None:
            print(msg, flush=True)
            self._queue.put(("log", msg))

        try:
            inputs = collect_inputs([config.input_path], config.output_path)
            device = default_device() if config.device == "Auto" else torch.device(config.device)
            if device.type == "cuda" and not torch.cuda.is_available():
                if torch.version.cuda is None:
                    raise RuntimeError(
                        "PyTorch in this environment was built without CUDA (CPU-only). "
                        "Run 'uv sync' in the repository to install the CUDA PyTorch wheel."
                    )
                raise RuntimeError(
                    "CUDA is not available on this system (check NVIDIA drivers or select CPU)."
                )
            precision_dtype = (
                None if config.precision == "Auto" else PRECISIONS[config.precision]
            )
            cache_key = (config.model, str(device), config.precision)

            with self._model_lock:
                cached = self._model_cache
            if cached is None or cached[0] != cache_key:
                if cached is not None:
                    with self._model_lock:
                        self._model_cache = None
                    del cached
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                self._queue.put(
                    ("progress_indeterminate", f"Loading {config.model} on {device}...")
                )
                emit_log(f"Loading {config.model} onto {device} ({config.precision})...")
                net, transform = create_model_and_transforms(
                    config.model, device, precision_dtype
                )
                with self._model_lock:
                    self._model_cache = (cache_key, net, transform)
                self._queue.put(("model_loaded", True))
                emit_log(
                    f"Loaded {config.model} onto {device} in "
                    f"{str(net.precision).removeprefix('torch.')}."
                )
            else:
                _, net, transform = cached
                emit_log(f"Reusing cached {config.model} on {device}.")

            config.output_path.mkdir(parents=True, exist_ok=True)
            total_items = len(inputs)
            last_output: Path | None = None
            run_timestamp = format_timestamp_suffix()

            for idx, path in enumerate(inputs):
                if self._cancel_event.is_set():
                    break

                is_video = path.suffix.lower() in VIDEO_EXTENSIONS
                if config.custom_stem is not None and total_items == 1:
                    effective_stem = config.custom_stem
                else:
                    default_targets = build_output_paths(
                        config.output_path,
                        path.stem,
                        is_video=is_video,
                        colormap=config.colormap,
                        append_colormap=config.append_colormap,
                        save_png16=config.save_png16,
                        save_npz=config.save_npz,
                    )
                    if find_existing_outputs(default_targets):
                        if config.conflict_policy == "auto_rename":
                            effective_stem = next_available_stem(
                                config.output_path,
                                path.stem,
                                is_video=is_video,
                                colormap=config.colormap,
                                append_colormap=config.append_colormap,
                                save_png16=config.save_png16,
                                save_npz=config.save_npz,
                            )
                        elif config.conflict_policy == "timestamp":
                            effective_stem = next_available_stem(
                                config.output_path,
                                path.stem,
                                is_video=is_video,
                                colormap=config.colormap,
                                append_colormap=config.append_colormap,
                                save_png16=config.save_png16,
                                save_npz=config.save_npz,
                                base_suffix=run_timestamp,
                            )
                        else:
                            effective_stem = path.stem
                    else:
                        effective_stem = path.stem

                targets = build_output_paths(
                    config.output_path,
                    effective_stem,
                    is_video=is_video,
                    colormap=config.colormap,
                    append_colormap=config.append_colormap,
                    save_png16=config.save_png16,
                    save_npz=config.save_npz,
                )
                if not is_video:
                    self._queue.put(
                        (
                            "progress_value",
                            100.0 * idx / total_items,
                            f"[{idx + 1}/{total_items}] {path.name}",
                        )
                    )
                    image, exif_f_px = load_rgb(path)
                    f_px = config.focal_px if config.focal_px is not None else exif_f_px
                    prediction = net.infer(transform(image), f_px=f_px)
                    inverse_depth = inverse_metric_depth(prediction)
                    scale = color_scale(inverse_depth)
                    colored = colorize_with_cmap(inverse_depth, scale, config.colormap)

                    png = targets.color_path
                    Image.fromarray(colored).save(png, compress_level=1)
                    written = [str(png)]
                    if targets.npz_path is not None:
                        write_npz(targets.npz_path, prediction)
                        written.append(str(targets.npz_path))
                    if targets.png16_path is not None:
                        grayscale16(inverse_depth, scale).save(
                            targets.png16_path, compress_level=1
                        )
                        written.append(str(targets.png16_path))

                    counts = np.zeros(len(DEPTH_BINS), dtype=np.int64)
                    count_depth(counts, inverse_depth)
                    p05, p95, inf_pct = compute_depth_stats(counts, DEPTH_BINS)
                    focal_val = float(prediction["focallength_px"])
                    source = (
                        "manual"
                        if config.focal_px is not None
                        else ("from EXIF" if exif_f_px is not None else "estimated")
                    )
                    for line in format_summary_lines(
                        written,
                        focal_val,
                        source,
                        p05,
                        p95,
                        inf_pct,
                        scale if config.save_png16 else None,
                    ):
                        emit_log(line)

                    meta = {
                        "focal_px": focal_val,
                        "source": source,
                        "p05": p05,
                        "p95": p95,
                        "inf_pct": inf_pct,
                    }
                    last_output = png
                    self._queue.put(
                        (
                            "export_state",
                            _ExportState(
                                kind="image",
                                stem=path.stem,
                                scale=scale,
                                focal_px=focal_val,
                                source=source,
                                p05=p05,
                                p95=p95,
                                inf_pct=inf_pct,
                                inverse_depth=inverse_depth,
                            ),
                        )
                    )
                    self._queue.put(
                        (
                            "preview",
                            make_preview_rgb(image),
                            inverse_depth,
                            make_preview_rgb(colored),
                            scale,
                            meta,
                            png,
                        )
                    )
                    self._queue.put(
                        (
                            "progress_value",
                            100.0 * (idx + 1) / total_items,
                            f"[{idx + 1}/{total_items}] {path.name}",
                        )
                    )
                    continue

                # Video processing
                video_path = targets.color_path
                npz_dir = targets.npz_path
                png16_dir = targets.png16_path
                if npz_dir is not None:
                    npz_dir.mkdir(exist_ok=True)
                if png16_dir is not None:
                    png16_dir.mkdir(exist_ok=True)

                counts = np.zeros(len(DEPTH_BINS), dtype=np.int64)
                scale = prediction = None
                count = 0
                preview_queue: deque[np.ndarray] = deque()
                last_frame_state: tuple[Any, ...] | None = None

                cache_dir: Path | None = None
                cache_file: Path | None = None
                cache_handle: Any = None
                cache_enabled = not config.save_npz
                cache_bytes = 0
                cache_handed_off = False
                if cache_enabled:
                    try:
                        cache_dir = Path(tempfile.mkdtemp(prefix="depthpro2_cache_"))
                        _register_temp_dir(cache_dir)
                        cache_file = cache_dir / "inv16.bin"
                        cache_handle = open(cache_file, "wb")
                    except Exception as exc:
                        cache_enabled = False
                        emit_log(f"Re-export cache disabled ({exc}).")
                        if cache_dir is not None:
                            _remove_temp_dir(cache_dir)
                            cache_dir = None
                            cache_file = None

                try:
                    with av.open(str(path)) as container:
                        stream = container.streams.video[0]
                        stream.thread_type = "AUTO"
                        fps = float(stream.average_rate or 30.0)
                        expected = stream.frames or round(
                            (container.duration or 0) / av.time_base * fps
                        )
                        res_label = (
                            f"max short {config.max_short_side}"
                            if config.max_short_side is not None
                            else "native"
                        )
                        emit_log(
                            f"Starting video {path.name}: ~{expected or '?'} frames @ {fps:.2f} "
                            f"fps (mode={config.mode}, window={config.window}, "
                            f"overlap={config.overlap}, res={res_label})."
                        )
                        self._queue.put(
                            ("progress_indeterminate", f"{path.name}: decoding first clip...")
                        )

                        frame_iter = container.decode(stream)
                        first_frame = next(frame_iter, None)
                        if first_frame is None:
                            raise RuntimeError(f"No frames decoded from {path}.")

                        first_rgb, scale_factor = resize_short_side(
                            upright(first_frame), config.max_short_side
                        )
                        actual_h, actual_w = first_rgb.shape[:2]
                        video_f_px = (
                            None if config.focal_px is None else config.focal_px * scale_factor
                        )
                        if scale_factor != 1.0:
                            emit_log(
                                f"Resized video input to {actual_w}x{actual_h} "
                                f"(scale {scale_factor:.3f})."
                            )
                        else:
                            emit_log(f"Video frame size: {actual_w}x{actual_h}.")

                        if device.type == "cuda":
                            total_gb = torch.cuda.get_device_properties(device).total_memory / (
                                1024**3
                            )
                            if (
                                config.model in ("depthpro2-h", "depthpro2-l")
                                and total_gb <= 18.0
                                and config.window * actual_w * actual_h > 4 * 1920 * 1080
                            ):
                                emit_log(
                                    f"Warning: {config.model} ({config.mode}, "
                                    f"window={config.window}) at {actual_w}x{actual_h} may "
                                    f"exceed {total_gb:.0f} GB VRAM and spill to Windows shared "
                                    "RAM (very slow). If stalled, lower Window (e.g. 4 with "
                                    "Overlap 1), lower Max short (e.g. 720p), or use "
                                    "depthpro2-b / depthpro2-s."
                                )

                        first_clip_target = (
                            1
                            if config.mode == "unbuffered"
                            else min(config.window, expected or config.window)
                        )
                        decoded = 0
                        video_start_t = time.monotonic()

                        def iter_frames() -> Any:
                            nonlocal decoded
                            for raw_frame in itertools.chain((first_frame,), frame_iter):
                                if self._cancel_event.is_set():
                                    break
                                if decoded == 0:
                                    rgb = first_rgb
                                else:
                                    rgb, _ = resize_short_side(
                                        upright(raw_frame), config.max_short_side
                                    )
                                decoded += 1
                                preview_queue.append(make_preview_rgb(rgb))
                                tensor = transform(rgb)
                                if decoded == first_clip_target:
                                    stage_msg = (
                                        f"{path.name}: running clip 1 "
                                        f"({decoded} frames @ {actual_w}x{actual_h})..."
                                    )
                                    self._queue.put(("progress_indeterminate", stage_msg))
                                    emit_log(
                                        f"Decoded {decoded} frames ({actual_w}x{actual_h}); "
                                        "running clip 1..."
                                    )
                                yield tensor

                        predictions = net.infer_video(
                            iter_frames(),
                            config.window,
                            config.overlap,
                            f_px=video_f_px,
                            mode=config.mode,  # type: ignore[arg-type]
                        )
                        last_preview_time = 0.0
                        last_log_time = 0.0
                        last_logged_pct = -CONSOLE_PROGRESS_STEP_PCT
                        vram_warned = False
                        vram_text = ""

                        with VideoWriter(video_path, fps) as writer:
                            for prediction in predictions:
                                if self._cancel_event.is_set():
                                    break
                                preview_rgb = preview_queue.popleft()
                                inverse_depth = inverse_metric_depth(prediction)
                                if scale is None:
                                    scale = color_scale(inverse_depth)
                                if cache_enabled and cache_handle is not None:
                                    q16 = quantize_u16(inverse_depth, scale)
                                    if cache_bytes + q16.nbytes <= MAX_VIDEO_CACHE_BYTES:
                                        cache_handle.write(q16.tobytes())
                                        cache_bytes += q16.nbytes
                                    else:
                                        cache_enabled = False
                                        cache_handle.close()
                                        cache_handle = None
                                        _remove_temp_dir(cache_dir)
                                        cache_dir = None
                                        cache_file = None
                                        emit_log(
                                            "Note: video exceeded 2 GiB re-export cache limit; "
                                            "post-run re-export disabled for this video."
                                        )
                                colored = colorize_with_cmap(inverse_depth, scale, config.colormap)
                                writer.write(colored)
                                if npz_dir is not None:
                                    write_npz(npz_dir / f"{count:06d}.npz", prediction)
                                if png16_dir is not None:
                                    grayscale16(inverse_depth, scale).save(
                                        png16_dir / f"{png16_dir.name}.{count:06d}.png",
                                        compress_level=1,
                                    )
                                count_depth(counts, inverse_depth)
                                count += 1
                                focal_val = float(prediction["focallength_px"])
                                last_frame_state = (
                                    preview_rgb,
                                    inverse_depth,
                                    colored,
                                    scale,
                                    focal_val,
                                )

                                now = time.monotonic()
                                elapsed_s = now - video_start_t
                                approx_pct = (100.0 * count / expected) if expected > 0 else None
                                should_log = (
                                    count == 1
                                    or (expected > 0 and count == expected)
                                    or (now - last_log_time >= CONSOLE_PROGRESS_INTERVAL_S)
                                    or (
                                        approx_pct is not None
                                        and approx_pct - last_logged_pct
                                        >= CONSOLE_PROGRESS_STEP_PCT
                                    )
                                )
                                if should_log:
                                    vram_text, vram_ratio = _cuda_vram_stats(device)
                                    if (
                                        not vram_warned
                                        and vram_ratio is not None
                                        and vram_ratio >= 0.90
                                    ):
                                        vram_warned = True
                                        emit_log(
                                            f"Warning: CUDA VRAM usage at "
                                            f"{100.0 * vram_ratio:.0f}% \u2014 may spill to "
                                            "Windows shared RAM. Consider smaller Window, lower "
                                            "Max short, or depthpro2-b / depthpro2-s."
                                        )

                                pct, progress_line = format_video_progress(
                                    path.name, count, expected, elapsed_s, vram_text
                                )
                                if pct is not None:
                                    self._queue.put(("progress_value", pct, progress_line))
                                else:
                                    self._queue.put(("progress_indeterminate", progress_line))

                                if should_log:
                                    last_log_time = now
                                    if pct is not None:
                                        last_logged_pct = pct
                                    emit_log(progress_line)

                                if count == 1 or now - last_preview_time >= VIDEO_UI_THROTTLE_S:
                                    last_preview_time = now
                                    p05, p95, inf_pct = compute_depth_stats(counts, DEPTH_BINS)
                                    source = (
                                        "manual"
                                        if config.focal_px is not None
                                        else (
                                            f"estimated from first {min(config.window, count)} "
                                            "frames"
                                        )
                                    )
                                    meta = {
                                        "focal_px": focal_val,
                                        "source": source,
                                        "p05": p05,
                                        "p95": p95,
                                        "inf_pct": inf_pct,
                                    }
                                    self._queue.put(
                                        (
                                            "preview",
                                            preview_rgb,
                                            inverse_depth,
                                            make_preview_rgb(colored),
                                            scale,
                                            meta,
                                            None,
                                        )
                                    )

                    if count == 0:
                        if self._cancel_event.is_set():
                            emit_log(f"Cancelled {path.name} before first frame.")
                            break
                        raise RuntimeError(f"No frames decoded from {path}.")

                    if cache_handle is not None:
                        cache_handle.close()
                        cache_handle = None

                    written = [f"{count} frames at {fps:.5g} fps to {video_path}"]
                    if npz_dir is not None:
                        written.append(f"{npz_dir}/")
                    if png16_dir is not None:
                        written.append(f"{png16_dir}/")
                    p05, p95, inf_pct = compute_depth_stats(counts, DEPTH_BINS)
                    source = (
                        "manual"
                        if config.focal_px is not None
                        else f"estimated from the first {min(config.window, count)} frames"
                    )
                    assert last_frame_state is not None
                    preview_rgb, inverse_depth, colored, final_scale, focal_val = last_frame_state
                    for line in format_summary_lines(
                        written,
                        focal_val,
                        source,
                        p05,
                        p95,
                        inf_pct,
                        final_scale if config.save_png16 else None,
                    ):
                        emit_log(line)
                    last_output = video_path
                    if not self._cancel_event.is_set() and (cache_enabled or config.save_npz):
                        export_state = _ExportState(
                            kind="video",
                            stem=path.stem,
                            scale=final_scale,
                            focal_px=focal_val,
                            source=source,
                            p05=p05,
                            p95=p95,
                            inf_pct=inf_pct,
                            frame_shape=inverse_depth.shape[:2],
                            frame_count=count,
                            fps=fps,
                            cache_dir=cache_dir if cache_enabled else None,
                            cache_file=cache_file if cache_enabled else None,
                            npz_dir=npz_dir if config.save_npz else None,
                        )
                        cache_handed_off = True
                        self._queue.put(("export_state", export_state))
                    meta = {
                        "focal_px": focal_val,
                        "source": source,
                        "p05": p05,
                        "p95": p95,
                        "inf_pct": inf_pct,
                    }
                    self._queue.put(
                        (
                            "preview",
                            preview_rgb,
                            inverse_depth,
                            make_preview_rgb(colored),
                            final_scale,
                            meta,
                            video_path,
                        )
                    )
                finally:
                    if cache_handle is not None:
                        try:
                            cache_handle.close()
                        except Exception:
                            pass
                    if not cache_handed_off and cache_dir is not None:
                        _remove_temp_dir(cache_dir)

            cancelled = self._cancel_event.is_set()
            self._queue.put(("done", cancelled, last_output))
        except Exception as exc:
            self._queue.put(("error", str(exc)))

    def _poll_queue(self) -> None:
        latest_progress: tuple[Any, ...] | None = None
        latest_preview: tuple[Any, ...] | None = None
        latest_preview_out_path: Path | None = None
        try:
            while True:
                msg = self._queue.get_nowait()
                kind = msg[0]
                if kind == "env_status":
                    self.status_left_var.set(msg[1])
                elif kind == "log":
                    self._append_log(msg[1])
                elif kind == "model_loaded":
                    self._update_button_states()
                elif kind == "export_state":
                    self._set_export_state(msg[1])
                elif kind == "reexport_preview":
                    self._current_color_rgb = msg[1]
                    self._last_output_path = msg[2]
                    self._update_button_states()
                    self._redraw_canvas()
                elif kind in ("progress_indeterminate", "progress_value"):
                    latest_progress = msg
                elif kind == "preview":
                    latest_preview = msg
                    if msg[6] is not None:
                        latest_preview_out_path = msg[6]
                elif kind == "done":
                    if latest_progress is not None:
                        self._apply_progress_msg(latest_progress)
                        latest_progress = None
                    if latest_preview is not None:
                        self._apply_preview_msg(latest_preview, latest_preview_out_path)
                        latest_preview = None
                        latest_preview_out_path = None
                    _, cancelled, last_out = msg
                    self._finish_run(cancelled=cancelled, last_output=last_out)
                elif kind == "error":
                    latest_progress = None
                    if latest_preview is not None:
                        self._apply_preview_msg(latest_preview, latest_preview_out_path)
                        latest_preview = None
                        latest_preview_out_path = None
                    self._finish_run(error=msg[1])
        except queue.Empty:
            pass

        if latest_progress is not None and self._is_running:
            self._apply_progress_msg(latest_progress)
        if latest_preview is not None:
            self._apply_preview_msg(latest_preview, latest_preview_out_path)

        if self._is_running and self._stage_start_time is not None:
            elapsed = time.monotonic() - self._stage_start_time
            elapsed_sec = int(elapsed)
            if elapsed_sec != self._last_elapsed_second:
                self._last_elapsed_second = elapsed_sec
                if str(self.progress_bar.cget("mode")) == "indeterminate":
                    display = f"{self._stage_base_text} ({format_duration(elapsed)})"
                    self.progress_text_var.set(display)
                    self.status_right_var.set(display)

        if self.root.winfo_exists():
            self._poll_after_id = self.root.after(POLL_INTERVAL_MS, self._poll_queue)

    def _apply_progress_msg(self, msg: tuple[Any, ...]) -> None:
        if msg[0] == "progress_indeterminate":
            text = msg[1]
            self._set_indeterminate(True)
            if text != self._stage_base_text:
                self._stage_base_text = text
                self._stage_start_time = time.monotonic()
                self._last_elapsed_second = 0
                display = f"{text} (00:00)"
                self.progress_text_var.set(display)
                self.status_right_var.set(display)
        else:
            _, value, text = msg
            self._set_indeterminate(False)
            self._stage_base_text = text
            self._stage_start_time = None
            self._last_elapsed_second = -1
            self.progress_var.set(value)
            self.progress_text_var.set(text)
            self.status_right_var.set(text)

    def _apply_preview_msg(
        self, msg: tuple[Any, ...], out_path: Path | None
    ) -> None:
        _, inp_rgb, inv_depth, col_rgb, scale, meta, msg_out_path = msg
        self._current_input_rgb = inp_rgb
        self._current_inverse_depth = inv_depth
        self._current_color_rgb = col_rgb
        self._current_depth_scale = scale
        self._current_meta = meta
        final_out = out_path if out_path is not None else msg_out_path
        if final_out is not None:
            self._last_output_path = final_out
            self._update_button_states()
        self.near_label_var.set(format_depth_value(scale))
        self._update_summary_label(meta)
        self._redraw_colorbar()
        self._redraw_canvas()

    def _finish_run(
        self,
        *,
        cancelled: bool = False,
        last_output: Path | None = None,
        error: str | None = None,
    ) -> None:
        self._is_running = False
        self._stage_start_time = None
        self._last_elapsed_second = -1
        self._set_indeterminate(False)
        if last_output is not None:
            self._last_output_path = last_output
        if error is not None:
            self._stage_base_text = "Error"
            self.progress_var.set(0.0)
            self.progress_text_var.set("Error")
            self.status_right_var.set("Error")
            self._append_log(f"Error: {error}")
            messagebox.showerror("Depth Pro 2 Error", error)
        elif cancelled:
            self._stage_base_text = "Cancelled"
            self.progress_text_var.set("Cancelled")
            self.status_right_var.set("Cancelled")
            self._append_log("Run cancelled.")
        else:
            self._stage_base_text = "Done"
            self.progress_var.set(100.0)
            self.progress_text_var.set("Done")
            self.status_right_var.set("Done")
        self._sync_mode_state()
        self._update_button_states()

    def _set_indeterminate(self, active: bool) -> None:
        current_mode = str(self.progress_bar.cget("mode"))
        target = "indeterminate" if active else "determinate"
        if current_mode != target:
            self.progress_bar.stop()
            self.progress_bar.configure(mode=target)
            if active:
                self.progress_bar.start(12)

    def _set_export_state(self, state: _ExportState | None) -> None:
        old = self._export_state
        old_dir = old.cache_dir if old is not None else None
        new_dir = state.cache_dir if state is not None else None
        if old_dir is not None and old_dir != new_dir:
            _remove_temp_dir(old_dir)
        self._export_state = state
        if new_dir is not None:
            _register_temp_dir(new_dir)
        self._update_button_states()

    def _update_button_states(self) -> None:
        with self._model_lock:
            has_model = self._model_cache is not None
        self.run_btn.configure(state="disabled" if self._is_running else "normal")
        self.cancel_btn.configure(state="normal" if self._is_running else "disabled")
        self.unload_btn.configure(
            state="normal" if (has_model and not self._is_running) else "disabled"
        )
        can_reexport = self._export_state is not None and not self._is_running
        self.reexport_btn.configure(state="normal" if can_reexport else "disabled")
        has_output = self._last_output_path is not None and self._last_output_path.exists()
        self.open_output_btn.configure(state="normal" if has_output else "disabled")

    def _update_summary_label(self, meta: dict[str, Any]) -> None:
        focal = meta.get("focal_px")
        p05 = meta.get("p05")
        p95 = meta.get("p95")
        inf_pct = meta.get("inf_pct")
        parts: list[str] = []
        if focal is not None:
            parts.append(f"f={focal:.1f}px")
        if p05 is not None and p95 is not None:
            parts.append(f"5\u201395%: {p05:.2f}\u2013{p95:.2f} m")
        if inf_pct is not None:
            parts.append(f"sky: {inf_pct:.1f}%")
        self.summary_text_var.set(" | ".join(parts))

    def _on_canvas_configure(self, _event: tk.Event[Any]) -> None:
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(RESIZE_DEBOUNCE_MS, self._redraw_canvas)

    def _on_colorbar_configure(self, _event: tk.Event[Any]) -> None:
        if self._colorbar_after_id is not None:
            self.root.after_cancel(self._colorbar_after_id)
        self._colorbar_after_id = self.root.after(RESIZE_DEBOUNCE_MS, self._redraw_colorbar)

    def _redraw_colorbar(self) -> None:
        self._colorbar_after_id = None
        w = self.colorbar_canvas.winfo_width()
        h = self.colorbar_canvas.winfo_height()
        if w < 10 or h < 4:
            return
        try:
            from PIL import Image, ImageTk

            lut = get_colormap_lut(self.colormap_var.get())
            bar_img = Image.fromarray(lut[None, :, :]).resize((w, h), Image.Resampling.BILINEAR)
            self._colorbar_photo = ImageTk.PhotoImage(bar_img)
            self.colorbar_canvas.delete("all")
            self.colorbar_canvas.create_image(0, 0, anchor="nw", image=self._colorbar_photo)
        except Exception:
            pass

    def _redraw_canvas(self) -> None:
        self._resize_after_id = None
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 20 or ch < 20:
            return

        self.canvas.delete("all")
        self._render_rects = []

        inp = self._current_input_rgb
        dep = self._current_color_rgb
        view = self.view_mode_var.get()

        if inp is None and dep is None:
            self.canvas.create_text(
                cw // 2,
                ch // 2,
                text=self._empty_message,
                fill=PALETTE["muted"],
                justify="center",
                width=max(220, cw - 60),
            )
            return

        try:
            from PIL import Image, ImageTk
        except ImportError:
            self.canvas.create_text(
                cw // 2,
                ch // 2,
                text="Pillow is required for image previews.\nLaunch via: uv run depth-pro-2 gui",
                fill=PALETTE["muted"],
                justify="center",
            )
            return

        bg_rgb = _hex_to_rgb(PALETTE["canvas"])
        composite = Image.new("RGB", (cw, ch), bg_rgb)
        labels: list[tuple[int, int, str]] = []
        split_line: tuple[int, int, int] | None = None

        if view == "side":
            slot_w = max(40, (cw - 36) // 2)
            slot_h = max(40, ch - 44)
            left_x0 = 12
            right_x0 = left_x0 + slot_w + 12
            top_y0 = 30
            ref = inp if inp is not None else dep
            assert ref is not None
            draw_w, draw_h = _fit_size(ref.shape[1], ref.shape[0], slot_w, slot_h)
            lx = left_x0 + (slot_w - draw_w) // 2
            ly = top_y0 + (slot_h - draw_h) // 2
            rx = right_x0 + (slot_w - draw_w) // 2
            ry = ly

            if inp is not None:
                in_img = Image.fromarray(inp).resize((draw_w, draw_h), Image.Resampling.BILINEAR)
                composite.paste(in_img, (lx, ly))
                self._render_rects.append((lx, ly, draw_w, draw_h))
                labels.append((lx, ly - 8, "Input"))

            if dep is not None:
                dep_img = Image.fromarray(dep).resize((draw_w, draw_h), Image.Resampling.BILINEAR)
                composite.paste(dep_img, (rx, ry))
                self._render_rects.append((rx, ry, draw_w, draw_h))
                labels.append((rx, ry - 8, "Depth"))
        elif view == "split":
            ref = inp if inp is not None else dep
            assert ref is not None
            draw_w, draw_h = _fit_size(ref.shape[1], ref.shape[0], cw - 24, ch - 44)
            ix = (cw - draw_w) // 2
            iy = 30 + max(0, (ch - 44 - draw_h) // 2)
            if inp is not None and dep is not None:
                import numpy as np

                in_arr = np.asarray(
                    Image.fromarray(inp).resize((draw_w, draw_h), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )
                dep_arr = np.asarray(
                    Image.fromarray(dep).resize((draw_w, draw_h), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )
                split_x = max(1, min(draw_w - 1, round(draw_w * self._split_ratio)))
                merged = np.empty_like(in_arr)
                merged[:, :split_x] = in_arr[:, :split_x]
                merged[:, split_x:] = dep_arr[:, split_x:]
                composite.paste(Image.fromarray(merged), (ix, iy))
                self._render_rects.append((ix, iy, draw_w, draw_h))
                labels.append((ix, iy - 8, "Input  |  Depth (drag divider)"))
                split_line = (ix + split_x, iy, iy + draw_h)
            else:
                only = inp if inp is not None else dep
                assert only is not None
                only_img = Image.fromarray(only).resize((draw_w, draw_h), Image.Resampling.BILINEAR)
                composite.paste(only_img, (ix, iy))
                self._render_rects.append((ix, iy, draw_w, draw_h))
                labels.append(
                    (ix, iy - 8, "Input (awaiting depth)" if inp is not None else "Depth")
                )
        elif view == "depth":
            if dep is not None:
                draw_w, draw_h = _fit_size(dep.shape[1], dep.shape[0], cw - 24, ch - 44)
                ix = (cw - draw_w) // 2
                iy = 30 + max(0, (ch - 44 - draw_h) // 2)
                dep_img = Image.fromarray(dep).resize((draw_w, draw_h), Image.Resampling.BILINEAR)
                composite.paste(dep_img, (ix, iy))
                self._render_rects.append((ix, iy, draw_w, draw_h))
                labels.append((ix, iy - 8, "Depth"))
            else:
                self.canvas.create_text(
                    cw // 2,
                    ch // 2,
                    text="No depth map generated yet.\nClick 'Run Depth Estimation' to start.",
                    fill=PALETTE["muted"],
                    justify="center",
                )
                return
        else:  # "input"
            if inp is not None:
                draw_w, draw_h = _fit_size(inp.shape[1], inp.shape[0], cw - 24, ch - 44)
                ix = (cw - draw_w) // 2
                iy = 30 + max(0, (ch - 44 - draw_h) // 2)
                in_img = Image.fromarray(inp).resize((draw_w, draw_h), Image.Resampling.BILINEAR)
                composite.paste(in_img, (ix, iy))
                self._render_rects.append((ix, iy, draw_w, draw_h))
                labels.append((ix, iy - 8, "Input"))

        self._photo_image = ImageTk.PhotoImage(composite)
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo_image)

        if view == "side" and inp is not None and dep is None:
            slot_w = max(40, (cw - 36) // 2)
            slot_h = max(40, ch - 44)
            rx0 = 12 + slot_w + 12
            self.canvas.create_rectangle(
                rx0, 30, rx0 + slot_w, 30 + slot_h, outline=PALETTE["border"], dash=(4, 4)
            )
            self.canvas.create_text(
                rx0 + slot_w // 2,
                30 + slot_h // 2,
                text="Depth preview will appear here",
                fill=PALETTE["muted"],
                justify="center",
            )

        for lx, ly, text in labels:
            self.canvas.create_text(lx, ly, text=text, fill=PALETTE["muted"], anchor="sw")

        if split_line is not None:
            sx, sy0, sy1 = split_line
            self.canvas.create_line(sx, sy0, sx, sy1, fill="#ffffff", width=2)
            mid_y = (sy0 + sy1) // 2
            self.canvas.create_oval(
                sx - 6, mid_y - 6, sx + 6, mid_y + 6, fill=PALETTE["accent"], outline="#ffffff"
            )

    def _on_canvas_drag(self, event: tk.Event[Any]) -> None:
        if self.view_mode_var.get() != "split" or not self._render_rects:
            return
        if self._current_input_rgb is None or self._current_color_rgb is None:
            return
        x0, _, rw, _ = self._render_rects[0]
        if rw <= 0:
            return
        self._split_ratio = min(0.98, max(0.02, (event.x - x0) / rw))
        self._redraw_canvas()

    def _on_canvas_motion(self, event: tk.Event[Any]) -> None:
        if self._current_inverse_depth is None or not self._render_rects:
            self._reset_hover_text()
            return
        height, width = self._current_inverse_depth.shape[:2]
        mapped = map_canvas_to_pixel(event.x, event.y, self._render_rects, width, height)
        if mapped is None:
            self._reset_hover_text()
            return
        px, py = mapped
        inv_val = float(self._current_inverse_depth[py, px])
        self.hover_text_var.set(f"Pixel ({px}, {py})  \u2022  Depth: {format_depth_value(inv_val)}")

    def _reset_hover_text(self) -> None:
        if self._current_inverse_depth is None:
            self.hover_text_var.set("Hover over preview to inspect metric depth")
        else:
            self.hover_text_var.set("Hover over image to inspect metric depth")

    def _open_last_output(self) -> None:
        if self._last_output_path is not None and self._last_output_path.exists():
            try:
                open_path_in_os(self._last_output_path)
            except Exception as exc:
                messagebox.showerror("Open Failed", f"Could not open output:\n{exc}")

    def _open_output_folder(self) -> None:
        out_dir = Path(self.output_var.get().strip() or str(Path.cwd() / "out"))
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            open_path_in_os(out_dir)
        except Exception as exc:
            messagebox.showerror("Open Failed", f"Could not open folder:\n{exc}")

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _show_help_dialog(self) -> None:
        if self._help_window is not None and self._help_window.winfo_exists():
            self._help_window.deiconify()
            self._help_window.lift()
            self._help_window.focus_force()
            return

        win = tk.Toplevel(self.root)
        win.title("Depth Pro 2 \u2014 Parameter Guide")
        win.geometry("680x620")
        win.minsize(520, 420)
        win.configure(bg=PALETTE["bg"])
        self._help_window = win

        def close_help() -> None:
            if self._help_window is not None:
                try:
                    self._help_window.destroy()
                except Exception:
                    pass
                self._help_window = None

        win.protocol("WM_DELETE_WINDOW", close_help)
        win.bind("<Escape>", lambda _e: close_help())

        outer = ttk.Frame(win, style="App.TFrame", padding=10)
        outer.pack(fill="both", expand=True)

        text_frame = ttk.Frame(outer, style="Panel.TFrame")
        text_frame.pack(fill="both", expand=True)

        base_font = ("Segoe UI", 10) if sys.platform == "win32" else ("TkDefaultFont", 10)
        sec_font = (
            ("Segoe UI", 11, "bold")
            if sys.platform == "win32"
            else ("TkDefaultFont", 11, "bold")
        )
        param_font = (
            ("Segoe UI", 10, "bold")
            if sys.platform == "win32"
            else ("TkDefaultFont", 10, "bold")
        )

        text = tk.Text(
            text_frame,
            wrap="word",
            bg=PALETTE["canvas"],
            fg=PALETTE["text"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
            padx=12,
            pady=10,
            font=base_font,
        )
        scroll = ttk.Scrollbar(
            text_frame, orient="vertical", command=text.yview, style="Vertical.TScrollbar"
        )
        text.configure(yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        text.tag_configure("section", foreground=PALETTE["accent"], font=sec_font, spacing1=8)
        text.tag_configure("param", foreground="#ffffff", font=param_font, spacing1=6)
        text.tag_configure("body", foreground=PALETTE["text"], lmargin1=14, lmargin2=14)
        text.tag_configure("muted", foreground=PALETTE["muted"], spacing3=4)

        text.insert(
            "end",
            "Quick Reference \u2014 hover any control in the sidebar for a short tooltip.\n",
            "muted",
        )
        for section_title, items in PARAMETER_GUIDE_SECTIONS:
            text.insert("end", f"\n{section_title}\n", "section")
            for param_name, description in items:
                text.insert("end", f"{param_name}\n", "param")
                text.insert("end", f"{description}\n", "body")
        text.configure(state="disabled")

        btn_row = ttk.Frame(outer, style="App.TFrame", padding=(0, 8, 0, 0))
        btn_row.pack(fill="x")
        ttk.Button(
            btn_row, text="Close", style="Secondary.TButton", command=close_help
        ).pack(side="right")

    def _on_close(self) -> None:
        self._cancel_event.set()
        self._set_export_state(None)
        _cleanup_all_temp_dirs()
        for tip in self._tooltips:
            tip.hide()
        if self._help_window is not None:
            try:
                self._help_window.destroy()
            except Exception:
                pass
            self._help_window = None
        if self._poll_after_id is not None:
            try:
                self.root.after_cancel(self._poll_after_id)
            except Exception:
                pass
            self._poll_after_id = None
        if self._resize_after_id is not None:
            try:
                self.root.after_cancel(self._resize_after_id)
            except Exception:
                pass
        if self._colorbar_after_id is not None:
            try:
                self.root.after_cancel(self._colorbar_after_id)
            except Exception:
                pass
        logging.getLogger("depth_pro_2").removeHandler(self._log_handler)
        self.root.destroy()


def main_gui() -> None:
    """Launch the Depth Pro 2 desktop GUI."""
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    DepthPro2App(root)
    root.mainloop()


if __name__ == "__main__":
    main_gui()
