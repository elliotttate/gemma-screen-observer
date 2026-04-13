"""Screen capture module with Win32 PrintWindow for background window capture.

On Windows, uses the PrintWindow API with PW_RENDERFULLCONTENT to capture
windows even when minimized or behind other windows (same approach as OBS Studio).
Falls back to mss for non-Windows platforms or full-monitor capture.
"""

from __future__ import annotations

import base64
import io
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mss
from PIL import Image

from .config import CaptureConfig

logger = logging.getLogger(__name__)

# Win32 constants and types — loaded lazily on Windows only
_win32_ready = False
if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes as wt

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    kernel32 = ctypes.windll.kernel32

    PW_RENDERFULLCONTENT = 2
    DIB_RGB_COLORS = 0
    SRCCOPY = 0x00CC0020
    BI_RGB = 0
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wt.DWORD),
            ("biWidth", wt.LONG),
            ("biHeight", wt.LONG),
            ("biPlanes", wt.WORD),
            ("biBitCount", wt.WORD),
            ("biCompression", wt.DWORD),
            ("biSizeImage", wt.DWORD),
            ("biXPelsPerMeter", wt.LONG),
            ("biYPelsPerMeter", wt.LONG),
            ("biClrUsed", wt.DWORD),
            ("biClrImportant", wt.DWORD),
        ]

    class BITMAPINFO(ctypes.Structure):
        _fields_ = [
            ("bmiHeader", BITMAPINFOHEADER),
            ("bmiColors", wt.DWORD * 3),
        ]

    WNDENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

    # Set DPI awareness for accurate capture on high-DPI displays
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass

    _win32_ready = True


@dataclass
class WindowInfo:
    """Information about a discovered window."""

    hwnd: int
    title: str
    pid: int
    process_name: str

    def to_dict(self) -> dict:
        return {
            "hwnd": self.hwnd,
            "title": self.title,
            "pid": self.pid,
            "process_name": self.process_name,
        }


@dataclass
class Frame:
    """A captured screen frame."""

    image: Image.Image
    timestamp: float
    frame_number: int
    source: str  # "monitor:0", "window:Game Title", etc.

    @property
    def base64_jpeg(self) -> str:
        """Encode the frame as a base64 JPEG string."""
        buf = io.BytesIO()
        rgb = self.image.convert("RGB") if self.image.mode != "RGB" else self.image
        rgb.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @property
    def base64_png(self) -> str:
        """Encode the frame as a base64 PNG string."""
        buf = io.BytesIO()
        self.image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def compressed_base64(self, max_kb: int = 950, initial_quality: int = 85) -> str:
        """Progressively compress the image to fit within max_kb.

        Uses the same progressive strategy as WSLSnapit:
        1. JPEG at initial quality
        2. Reduce quality progressively
        3. Resize to 1280px wide if still too large
        4. Final resize to 800px at quality 50
        """
        rgb = self.image.convert("RGB") if self.image.mode != "RGB" else self.image
        max_bytes = max_kb * 1024

        # Step 1: Try at initial quality
        buf = io.BytesIO()
        img = rgb
        if img.width > 1920:
            ratio = 1920 / img.width
            img = img.resize((1920, int(img.height * ratio)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=initial_quality)
        if buf.tell() <= max_bytes:
            return base64.b64encode(buf.getvalue()).decode("ascii")

        # Step 2: Reduce quality progressively
        for quality in [80, 70, 60]:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            if buf.tell() <= max_bytes:
                return base64.b64encode(buf.getvalue()).decode("ascii")

        # Step 3: Resize to 1280px wide
        if img.width > 1280:
            ratio = 1280 / img.width
            img = img.resize((1280, int(img.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=60)
        if buf.tell() <= max_bytes:
            return base64.b64encode(buf.getvalue()).decode("ascii")

        # Step 4: Final fallback - 800px, quality 50
        ratio = 800 / img.width
        img = img.resize((800, int(img.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=50)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def save(self, directory: str | Path, fmt: str = "png") -> Path:
        """Save the frame to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"frame_{self.frame_number:06d}_{int(self.timestamp)}.{fmt}"
        path = directory / filename
        self.image.save(path)
        return path


# ---------------------------------------------------------------------------
# Win32 window enumeration and capture
# ---------------------------------------------------------------------------


def _get_process_name(pid: int) -> str:
    """Get the executable name for a process ID using Win32 API."""
    if not _win32_ready:
        return ""
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(512)
        size = wt.DWORD(512)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            # Return just the filename, not the full path
            return buf.value.rsplit("\\", 1)[-1]
        return ""
    finally:
        kernel32.CloseHandle(handle)


def enumerate_windows() -> list[WindowInfo]:
    """Enumerate all visible windows with titles."""
    if not _win32_ready:
        raise RuntimeError("Window enumeration requires Windows")

    windows: list[WindowInfo] = []

    def _callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        if not title.strip():
            return True
        pid = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        proc_name = _get_process_name(pid.value)
        windows.append(WindowInfo(hwnd=hwnd, title=title, pid=pid.value, process_name=proc_name))
        return True

    user32.EnumWindows(WNDENUMPROC(_callback), 0)
    return windows


def find_windows(
    title: str | None = None,
    process_name: str | None = None,
) -> list[WindowInfo]:
    """Find windows matching the given title substring or process name."""
    all_windows = enumerate_windows()
    matches = []
    for w in all_windows:
        if title and title.lower() not in w.title.lower():
            continue
        if process_name:
            target = process_name.lower().removesuffix(".exe")
            actual = w.process_name.lower().removesuffix(".exe")
            if target != actual:
                continue
        matches.append(w)
    return matches


def capture_window(hwnd: int) -> Image.Image:
    """Capture a window using PrintWindow with PW_RENDERFULLCONTENT.

    Works even if the window is minimized, hidden, or behind other windows.
    """
    if not _win32_ready:
        raise RuntimeError("Window capture requires Windows")

    rect = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    width = rect.right - rect.left
    height = rect.bottom - rect.top

    if width <= 0 or height <= 0:
        raise ValueError(f"Window has invalid dimensions: {width}x{height}")

    # Create a device context and bitmap to render into
    hwnd_dc = user32.GetWindowDC(hwnd)
    if not hwnd_dc:
        raise RuntimeError("Failed to get window device context")

    mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
    old_bitmap = gdi32.SelectObject(mem_dc, bitmap)

    try:
        # The key call: PrintWindow renders the window content to our DC
        # even if the window is minimized or behind other windows.
        # Flag 2 = PW_RENDERFULLCONTENT (includes non-client area, composited content)
        success = user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT)
        if not success:
            raise RuntimeError("PrintWindow failed for the target window")

        # Extract pixel data from the bitmap via GetDIBits
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = width
        bmi.bmiHeader.biHeight = -height  # Negative = top-down DIB
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB

        buffer_size = width * height * 4
        pixels = ctypes.create_string_buffer(buffer_size)
        gdi32.GetDIBits(mem_dc, bitmap, 0, height, pixels, ctypes.byref(bmi), DIB_RGB_COLORS)

        # Convert BGRA pixel data to a PIL Image
        image = Image.frombuffer("RGBA", (width, height), pixels, "raw", "BGRA", 0, 1)
        return image.convert("RGB")
    finally:
        gdi32.SelectObject(mem_dc, old_bitmap)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hwnd, hwnd_dc)


def capture_window_desktop(hwnd: int) -> Image.Image:
    """Capture a window by grabbing its screen region from the desktop.

    Unlike PrintWindow, this captures the DWM-composited output which includes
    DirectX/Vulkan/OpenGL rendered content. The window must be visible (not
    minimized) for this to work.
    """
    if not _win32_ready:
        raise RuntimeError("Window desktop capture requires Windows")

    rect = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    left, top = rect.left, rect.top
    width = rect.right - rect.left
    height = rect.bottom - rect.top

    if width <= 0 or height <= 0:
        raise ValueError(f"Window has invalid dimensions: {width}x{height}")

    with mss.mss() as sct:
        region = {"left": left, "top": top, "width": width, "height": height}
        screenshot = sct.grab(region)
        return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")


def _is_blank_frame(image: Image.Image, threshold: float = 0.95) -> bool:
    """Detect if a captured frame is blank (nearly all one color).

    PrintWindow returns blank white/black frames for DirectX/Vulkan windows.
    """
    w, h = image.size
    # Sample a grid of pixels instead of checking every pixel
    sample_count = 0
    dominant_count = 0
    step = max(1, w * h // 2000)  # ~2000 samples

    pixels = image.getdata()
    for i in range(0, len(pixels), step):
        r, g, b = pixels[i][:3]
        sample_count += 1
        if (r > 250 and g > 250 and b > 250) or (r < 5 and g < 5 and b < 5):
            dominant_count += 1

    return sample_count > 0 and (dominant_count / sample_count) > threshold


# ---------------------------------------------------------------------------
# Monitor capture (cross-platform fallback via mss)
# ---------------------------------------------------------------------------


def capture_monitor(monitor: int | str = "primary", region: tuple[int, int, int, int] | None = None) -> Image.Image:
    """Capture a monitor using mss."""
    with mss.mss() as sct:
        monitors = sct.monitors

        if region:
            left, top, width, height = region
            grab_area = {"left": left, "top": top, "width": width, "height": height}
        elif monitor == "all" or monitor == 0:
            grab_area = monitors[0]  # Full virtual screen
        elif monitor == "primary" or monitor == 1:
            grab_area = monitors[1] if len(monitors) > 1 else monitors[0]
        elif isinstance(monitor, int) and monitor < len(monitors):
            grab_area = monitors[monitor]
        else:
            grab_area = monitors[0]

        screenshot = sct.grab(grab_area)
        return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")


def list_monitors() -> list[dict]:
    """List available monitors."""
    with mss.mss() as sct:
        result = []
        for i, m in enumerate(sct.monitors):
            result.append({
                "index": i,
                "left": m["left"],
                "top": m["top"],
                "width": m["width"],
                "height": m["height"],
                "label": "all monitors" if i == 0 else f"monitor {i}",
            })
        return result


# ---------------------------------------------------------------------------
# Unified ScreenCapture interface
# ---------------------------------------------------------------------------


@dataclass
class ScreenCapture:
    """Unified screen/window capture with automatic backend selection.

    When window_title or process_name is configured, uses PrintWindow for
    background capture. Otherwise falls back to mss monitor capture.
    """

    config: CaptureConfig
    _frame_count: int = field(default=0, init=False)
    _previous_frame: Frame | None = field(default=None, init=False)
    _resolved_hwnd: int | None = field(default=None, init=False)
    _resolved_window: WindowInfo | None = field(default=None, init=False)

    def _resolve_window(self) -> int | None:
        """Resolve the target window handle from config, caching the result."""
        if not _win32_ready:
            return None
        if not (self.config.window_title or self.config.process_name):
            return None

        matches = find_windows(
            title=self.config.window_title,
            process_name=self.config.process_name,
        )

        if not matches:
            logger.warning(
                "No window found matching title=%r process=%r",
                self.config.window_title,
                self.config.process_name,
            )
            return None

        idx = (self.config.window_index or 1) - 1  # Convert to 0-based
        if idx >= len(matches):
            idx = 0

        self._resolved_window = matches[idx]
        self._resolved_hwnd = matches[idx].hwnd

        if len(matches) > 1:
            logger.info(
                "Multiple windows matched (%d), using index %d: %r",
                len(matches),
                idx + 1,
                self._resolved_window.title,
            )
        else:
            logger.info("Targeting window: %r (PID %d)", self._resolved_window.title, self._resolved_window.pid)

        return self._resolved_hwnd

    def capture(
        self,
        *,
        window_title: str | None = None,
        process_name: str | None = None,
        window_index: int | None = None,
        monitor: int | str | None = None,
    ) -> Frame:
        """Capture a frame.

        Parameters override the config for this single capture.
        When no overrides are given, uses the configured defaults.
        """
        # Determine capture source
        w_title = window_title or self.config.window_title
        p_name = process_name or self.config.process_name
        w_index = window_index or self.config.window_index
        mon = monitor if monitor is not None else self.config.monitor

        image: Image.Image
        source: str

        if w_title or p_name:
            # Window-specific capture via PrintWindow
            if not _win32_ready:
                raise RuntimeError(
                    "Window capture requires Windows. Use monitor capture on other platforms."
                )

            # Re-resolve if the override differs from cached or no cache
            if (
                self._resolved_hwnd is None
                or (w_title and w_title != self.config.window_title)
                or (p_name and p_name != self.config.process_name)
            ):
                matches = find_windows(title=w_title, process_name=p_name)
                if not matches:
                    raise ValueError(
                        f"No window found matching title={w_title!r} process={p_name!r}"
                    )
                idx = (w_index or 1) - 1
                if idx >= len(matches):
                    idx = 0
                self._resolved_window = matches[idx]
                self._resolved_hwnd = matches[idx].hwnd
            elif self._resolved_hwnd is None:
                self._resolve_window()
                if self._resolved_hwnd is None:
                    raise ValueError("Could not resolve target window")

            try:
                image = capture_window(self._resolved_hwnd)
                # PrintWindow returns blank frames for DirectX/Vulkan games.
                # Detect this and fall back to desktop-region capture.
                if _is_blank_frame(image):
                    logger.debug("PrintWindow returned blank frame, falling back to desktop capture")
                    image = capture_window_desktop(self._resolved_hwnd)
            except RuntimeError:
                # Window may have closed or PrintWindow failed — try desktop capture, then re-resolve
                try:
                    image = capture_window_desktop(self._resolved_hwnd)
                except Exception:
                    logger.warning("Capture failed, re-resolving window...")
                    self._resolved_hwnd = None
                    self._resolve_window()
                    if self._resolved_hwnd is None:
                        raise ValueError("Target window no longer available")
                    image = capture_window_desktop(self._resolved_hwnd)

            source = f"window:{self._resolved_window.title if self._resolved_window else 'unknown'}"
        else:
            # Monitor capture via mss
            image = capture_monitor(mon, self.config.region)
            source = f"monitor:{mon}"

        # Resize for inference
        if self.config.resize:
            tw, th = self.config.resize
            if image.size != (tw, th):
                image = image.resize((tw, th), Image.LANCZOS)

        self._frame_count += 1
        frame = Frame(
            image=image,
            timestamp=time.time(),
            frame_number=self._frame_count,
            source=source,
        )

        if self.config.save_screenshots:
            frame.save(self.config.screenshot_dir)

        self._previous_frame = frame
        return frame

    @property
    def previous_frame(self) -> Frame | None:
        return self._previous_frame

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def target_window(self) -> WindowInfo | None:
        return self._resolved_window

    def refresh_window(self) -> WindowInfo | None:
        """Force re-resolution of the target window."""
        self._resolved_hwnd = None
        self._resolved_window = None
        self._resolve_window()
        return self._resolved_window
