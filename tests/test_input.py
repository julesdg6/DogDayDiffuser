"""
tests/test_input.py — Unit tests for the DogDayDiffuser input abstraction.

These tests do not require any hardware (no webcam, no USB device).  They
use temporary directories and mock objects so they can run in CI.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


def _make_files(base: Path, names: List[str]) -> List[Path]:
    """Create empty files under *base* and return their paths."""
    paths = []
    for name in names:
        p = base / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        paths.append(p)
    return paths


class _MinimalConfig:
    """Minimal stub that satisfies SourceManager's attribute access."""

    camera: int = 0
    width: int = 320
    height: int = 240
    prefer_camera: bool = True
    usb_mount_roots: Optional[List[str]] = None
    video_extensions: Optional[List[str]] = None
    usb_video_loop: bool = True
    usb_video_shuffle: bool = False
    rescan_on_source_failure: bool = True


# ---------------------------------------------------------------------------
# usb_media — extension filtering
# ---------------------------------------------------------------------------


class TestFindUsbVideoFiles:
    """Tests for input.usb_media.find_usb_video_files."""

    def test_finds_supported_extensions(self, tmp_path: Path) -> None:
        from input.usb_media import find_usb_video_files

        _make_files(
            tmp_path,
            [
                "clip.mp4",
                "loop.avi",
                "photo.jpg",
                "document.pdf",
                "video.mkv",
                "movie.mov",
            ],
        )

        result = find_usb_video_files(mount_roots=[str(tmp_path)])
        names = [p.name for p in result]

        assert "clip.mp4" in names
        assert "loop.avi" in names
        assert "video.mkv" in names
        assert "movie.mov" in names
        assert "photo.jpg" not in names
        assert "document.pdf" not in names

    def test_ignores_hidden_files(self, tmp_path: Path) -> None:
        from input.usb_media import find_usb_video_files

        _make_files(tmp_path, [".hidden.mp4", "visible.mp4"])

        result = find_usb_video_files(mount_roots=[str(tmp_path)])
        names = [p.name for p in result]

        assert "visible.mp4" in names
        assert ".hidden.mp4" not in names

    def test_ignores_hidden_directories(self, tmp_path: Path) -> None:
        from input.usb_media import find_usb_video_files

        hidden_dir = tmp_path / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "secret.mp4").touch()
        (tmp_path / "public.mp4").touch()

        result = find_usb_video_files(mount_roots=[str(tmp_path)])
        paths_str = [str(p) for p in result]

        assert not any(".hidden" in s for s in paths_str)
        assert any(s.endswith("public.mp4") for s in paths_str)

    def test_returns_sorted_results(self, tmp_path: Path) -> None:
        from input.usb_media import find_usb_video_files

        _make_files(tmp_path, ["c.mp4", "a.mp4", "b.mp4"])

        result = find_usb_video_files(mount_roots=[str(tmp_path)])
        assert result == sorted(result)

    def test_nonexistent_root_is_skipped(self, tmp_path: Path) -> None:
        from input.usb_media import find_usb_video_files

        result = find_usb_video_files(
            mount_roots=[str(tmp_path / "does_not_exist")]
        )
        assert result == []

    def test_custom_extensions(self, tmp_path: Path) -> None:
        from input.usb_media import find_usb_video_files

        _make_files(tmp_path, ["video.mp4", "custom.xyz", "other.avi"])

        result = find_usb_video_files(
            mount_roots=[str(tmp_path)], extensions=[".xyz"]
        )
        names = [p.name for p in result]
        assert names == ["custom.xyz"]

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        from input.usb_media import find_usb_video_files

        result = find_usb_video_files(mount_roots=[str(tmp_path)])
        assert result == []

    def test_multiple_mount_roots(self, tmp_path: Path) -> None:
        from input.usb_media import find_usb_video_files

        root1 = tmp_path / "media"
        root2 = tmp_path / "mnt"
        root1.mkdir()
        root2.mkdir()
        (root1 / "a.mp4").touch()
        (root2 / "b.mp4").touch()

        result = find_usb_video_files(mount_roots=[str(root1), str(root2)])
        names = [p.name for p in result]
        assert "a.mp4" in names
        assert "b.mp4" in names


# ---------------------------------------------------------------------------
# CameraSource
# ---------------------------------------------------------------------------


class TestCameraSource:
    """Tests for input.camera_source.CameraSource."""

    def test_open_fails_when_capture_cannot_open(self) -> None:
        from input.camera_source import CameraSource

        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_cls.return_value = mock_cap

            source = CameraSource(index=99)
            assert source.open() is False

    def test_open_fails_when_test_frame_fails(self) -> None:
        from input.camera_source import CameraSource

        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (False, None)
            mock_cap_cls.return_value = mock_cap

            source = CameraSource(index=0)
            assert source.open() is False

    def test_open_succeeds(self) -> None:
        from input.camera_source import CameraSource

        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, dummy_frame)
            mock_cap.get.return_value = 320
            mock_cap_cls.return_value = mock_cap

            source = CameraSource(index=0, width=320, height=240)
            assert source.open() is True
            assert source.is_live is True
            assert "WEBCAM" in source.source_name

    def test_read_returns_false_when_not_open(self) -> None:
        from input.camera_source import CameraSource

        source = CameraSource()
        ok, frame = source.read()
        assert ok is False
        assert frame is None

    def test_read_returns_frame(self) -> None:
        from input.camera_source import CameraSource

        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.side_effect = [
                (True, dummy_frame),  # test-frame read during open()
                (True, dummy_frame),  # actual read
            ]
            mock_cap.get.return_value = 320
            mock_cap_cls.return_value = mock_cap

            source = CameraSource(index=0, width=320, height=240)
            source.open()

            ok, frame = source.read()
            assert ok is True
            assert frame is not None

    def test_release_clears_cap(self) -> None:
        from input.camera_source import CameraSource

        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, dummy_frame)
            mock_cap.get.return_value = 320
            mock_cap_cls.return_value = mock_cap

            source = CameraSource(index=0, width=320, height=240)
            source.open()
            source.release()
            assert source._cap is None


# ---------------------------------------------------------------------------
# VideoFileSource
# ---------------------------------------------------------------------------


class TestVideoFileSource:
    """Tests for input.video_file_source.VideoFileSource."""

    def test_requires_non_empty_playlist(self) -> None:
        from input.video_file_source import VideoFileSource

        with pytest.raises(ValueError):
            VideoFileSource(playlist=[])

    def test_open_fails_when_file_cannot_be_read(self, tmp_path: Path) -> None:
        from input.video_file_source import VideoFileSource

        fake = tmp_path / "fake.mp4"
        fake.touch()

        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_cls.return_value = mock_cap

            source = VideoFileSource(playlist=[fake])
            assert source.open() is False

    def test_open_succeeds_with_valid_file(self, tmp_path: Path) -> None:
        from input.video_file_source import VideoFileSource

        fake = tmp_path / "video.mp4"
        fake.touch()
        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, dummy_frame)
            mock_cap_cls.return_value = mock_cap

            source = VideoFileSource(playlist=[fake])
            assert source.open() is True
            assert source.is_live is False
            assert "USB VIDEO" in source.source_name

    def test_read_stops_when_loop_false(self, tmp_path: Path) -> None:
        from input.video_file_source import VideoFileSource

        fake = tmp_path / "once.mp4"
        fake.touch()
        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.side_effect = [
                (True, dummy_frame),  # test frame during open()
                (False, None),         # EOF on first read
            ]
            mock_cap_cls.return_value = mock_cap

            source = VideoFileSource(playlist=[fake], loop=False)
            source.open()

            ok, frame = source.read()
            assert ok is False
            assert frame is None

    def test_read_loops_single_file(self, tmp_path: Path) -> None:
        from input.video_file_source import VideoFileSource

        fake = tmp_path / "loop.mp4"
        fake.touch()
        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            # open: test-frame OK; first read: EOF; loop-rewind read: OK
            mock_cap.read.side_effect = [
                (True, dummy_frame),   # test frame during open()
                (False, None),          # EOF
                (True, dummy_frame),   # after rewind
            ]
            mock_cap_cls.return_value = mock_cap

            source = VideoFileSource(playlist=[fake], loop=True)
            source.open()

            ok, frame = source.read()  # triggers loop
            assert ok is True
            assert frame is not None
            # Verify that seek-to-start was called (i.e. the file was rewound).
            import cv2 as _cv2
            mock_cap.set.assert_called_with(_cv2.CAP_PROP_POS_FRAMES, 0)


# ---------------------------------------------------------------------------
# SourceManager — source selection and fallback logic
# ---------------------------------------------------------------------------


class TestSourceManager:
    """Tests for input.source_manager.SourceManager."""

    def _cfg(self, **overrides) -> _MinimalConfig:
        cfg = _MinimalConfig()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def test_selects_camera_when_available(self) -> None:
        from input.source_manager import SourceManager
        from input.camera_source import CameraSource

        with patch.object(CameraSource, "open", return_value=True):
            mgr = SourceManager(self._cfg())
            result = mgr.initialize()
            assert result is True
            assert isinstance(mgr._source, CameraSource)

    def test_falls_back_to_usb_when_camera_fails(self, tmp_path: Path) -> None:
        from input.source_manager import SourceManager
        from input.camera_source import CameraSource
        from input.video_file_source import VideoFileSource

        video = tmp_path / "clip.mp4"
        video.touch()

        with (
            patch.object(CameraSource, "open", return_value=False),
            patch(
                "input.source_manager.find_usb_video_files",
                return_value=[video],
            ),
            patch.object(VideoFileSource, "open", return_value=True),
        ):
            mgr = SourceManager(self._cfg(usb_mount_roots=[str(tmp_path)]))
            result = mgr.initialize()
            assert result is True
            assert isinstance(mgr._source, VideoFileSource)

    def test_fails_when_no_source_available(self, tmp_path: Path) -> None:
        from input.source_manager import SourceManager
        from input.camera_source import CameraSource

        with (
            patch.object(CameraSource, "open", return_value=False),
            patch(
                "input.source_manager.find_usb_video_files",
                return_value=[],
            ),
        ):
            mgr = SourceManager(self._cfg())
            result = mgr.initialize()
            assert result is False
            assert mgr._source is None

    def test_skips_camera_when_prefer_camera_false(self, tmp_path: Path) -> None:
        from input.source_manager import SourceManager
        from input.camera_source import CameraSource
        from input.video_file_source import VideoFileSource

        video = tmp_path / "clip.mp4"
        video.touch()

        with (
            patch.object(CameraSource, "open", return_value=True) as mock_cam_open,
            patch(
                "input.source_manager.find_usb_video_files",
                return_value=[video],
            ),
            patch.object(VideoFileSource, "open", return_value=True),
        ):
            mgr = SourceManager(self._cfg(prefer_camera=False))
            mgr.initialize()
            mock_cam_open.assert_not_called()
            assert isinstance(mgr._source, VideoFileSource)

    def test_get_status_no_source(self) -> None:
        from input.source_manager import SourceManager

        mgr = SourceManager(self._cfg())
        status = mgr.get_status()
        assert status["active"] is False
        assert status["source_name"] == "NO INPUT"

    def test_get_status_with_camera(self) -> None:
        from input.source_manager import SourceManager
        from input.camera_source import CameraSource

        with patch.object(CameraSource, "open", return_value=True):
            mgr = SourceManager(self._cfg())
            mgr.initialize()
            status = mgr.get_status()
            assert status["active"] is True
            assert "WEBCAM" in status["source_name"]
            assert status["is_live"] is True

    def test_read_returns_false_when_no_source(self) -> None:
        from input.source_manager import SourceManager

        mgr = SourceManager(self._cfg())
        ok, frame = mgr.read()
        assert ok is False
        assert frame is None

    def test_runtime_fallback_after_camera_failure(self, tmp_path: Path) -> None:
        """After _MAX_CONSECUTIVE_FAILURES bad reads, switch to USB video."""
        from input.source_manager import SourceManager
        from input.camera_source import CameraSource
        from input.video_file_source import VideoFileSource

        video = tmp_path / "clip.mp4"
        video.touch()

        with (
            patch.object(CameraSource, "open", return_value=True),
            patch.object(CameraSource, "read", return_value=(False, None)),
            patch.object(CameraSource, "release"),
            patch(
                "input.source_manager.find_usb_video_files",
                return_value=[video],
            ),
            patch.object(VideoFileSource, "open", return_value=True),
        ):
            mgr = SourceManager(self._cfg())
            mgr.initialize()
            assert isinstance(mgr._source, CameraSource)

            for _ in range(SourceManager._MAX_CONSECUTIVE_FAILURES):
                mgr.read()

            assert isinstance(mgr._source, VideoFileSource)

    def test_release_clears_source(self) -> None:
        from input.source_manager import SourceManager
        from input.camera_source import CameraSource

        with patch.object(CameraSource, "open", return_value=True):
            mgr = SourceManager(self._cfg())
            mgr.initialize()
            mgr.release()
            assert mgr._source is None
