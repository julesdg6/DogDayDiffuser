"""
midi/midi_manager.py — Korg nanoKONTROL2 auto-detection and MIDI I/O.

Usage::

    mgr = MidiManager()
    mgr.open()              # detects device, starts background reader
    state = mgr.get_state() # returns a MidiState snapshot
    mgr.close()             # stops reader and closes port

If ``mido`` / ``python-rtmidi`` are not installed, or no matching device
is found, :meth:`open` raises and the caller should disable MIDI gracefully.

The background thread reads MIDI messages at low latency without blocking
the video render loop.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional

from .mapping import (
    DEFAULT_CC_MAP,
    TRANSPORT_CC_ACTIONS,
    MidiState,
)
from . import nanokontrol2 as nk2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _cc_to_float(value: int) -> float:
    """Convert MIDI CC value (0–127) to normalised float (0.0–1.0)."""
    return max(0.0, min(1.0, value / 127.0))


# ---------------------------------------------------------------------------
# Main manager class
# ---------------------------------------------------------------------------


class MidiManager:
    """Manages MIDI input from a Korg nanoKONTROL2 (or compatible device).

    Args:
        device_name: Substring to match against available port names.
                     Defaults to ``nanoKONTROL2``.
        cc_map: CC-number → MidiState-attribute mapping.
                Defaults to :data:`~midi.mapping.DEFAULT_CC_MAP`.
    """

    def __init__(
        self,
        device_name: str = nk2.DEVICE_NAME_SUBSTRING,
        cc_map: Optional[dict] = None,
    ) -> None:
        self._device_name = device_name
        self._cc_map = cc_map if cc_map is not None else DEFAULT_CC_MAP

        self._state = MidiState()
        self._lock = threading.Lock()

        self._port = None       # mido input port
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Device discovery
    # ------------------------------------------------------------------

    @staticmethod
    def list_ports() -> List[str]:
        """Return all available MIDI input port names.

        Raises:
            ImportError: if ``mido`` is not installed.
        """
        import mido  # type: ignore
        return mido.get_input_names()

    def find_device(self) -> Optional[str]:
        """Return the first port whose name contains *device_name*, or None."""
        try:
            ports = self.list_ports()
        except ImportError:
            raise
        except Exception as exc:
            logger.warning("Could not query MIDI ports: %s", exc)
            return None

        for port in ports:
            if self._device_name.lower() in port.lower():
                return port
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, port_name: Optional[str] = None) -> None:
        """Open the MIDI port and start the background reader thread.

        Args:
            port_name: Exact port name to open.  If *None*, auto-detects
                       the first port matching :attr:`device_name`.

        Raises:
            ImportError:  ``mido`` / ``python-rtmidi`` not installed.
            RuntimeError: No matching port found, or port open failed.
        """
        import mido  # type: ignore  # noqa: F401 — validates import early

        if port_name is None:
            port_name = self.find_device()
            if port_name is None:
                raise RuntimeError(
                    f"No MIDI device matching '{self._device_name}' found. "
                    "Check USB connection and run midi.MidiManager.list_ports()."
                )

        logger.info("Opening MIDI port: %s", port_name)
        try:
            import mido
            self._port = mido.open_input(port_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to open MIDI port '{port_name}': {exc}") from exc

        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="midi-reader",
            daemon=True,
        )
        self._thread.start()
        logger.info("MIDI reader started for port: %s", port_name)

    def close(self) -> None:
        """Stop the reader thread and close the MIDI port."""
        self._running = False
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass
            self._port = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("MIDI manager closed")

    # ------------------------------------------------------------------
    # Background reader
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        """Background thread: reads MIDI messages and updates state."""
        try:
            for message in self._port:
                if not self._running:
                    break
                self._handle_message(message)
        except Exception as exc:
            if self._running:
                logger.warning("MIDI reader error: %s", exc)
        logger.debug("MIDI reader loop exited")

    def _handle_message(self, message) -> None:  # type: ignore[type-arg]
        """Dispatch a single mido Message to the appropriate handler."""
        msg_type = message.type

        if msg_type == "control_change":
            self._handle_cc(message.control, message.value)

        elif msg_type in ("note_on", "note_off"):
            # nanoKONTROL2 sends note_on with velocity 127 for press,
            # note_on with velocity 0 (or note_off) for release.
            pressed = (msg_type == "note_on" and message.velocity > 0)
            self._handle_note(message.note, pressed)

    def _handle_cc(self, cc: int, value: int) -> None:
        """Process a CC message."""
        # Continuous parameters (faders, knobs)
        if cc in self._cc_map:
            param = self._cc_map[cc]
            normalised = _cc_to_float(value)
            with self._lock:
                setattr(self._state, param, normalised)
            logger.debug("CC %d → %s = %.3f", cc, param, normalised)
            return

        # Transport CC buttons
        if cc in TRANSPORT_CC_ACTIONS:
            attr, action = TRANSPORT_CC_ACTIONS[cc]
            # Transport buttons fire on press (value 127) only
            if value == 127:
                with self._lock:
                    if action == "set_true":
                        setattr(self._state, attr, True)
                    elif action == "set_false":
                        setattr(self._state, attr, False)
                    else:  # toggle
                        setattr(self._state, attr, not getattr(self._state, attr))
                logger.debug("Transport CC %d → %s %s", cc, attr, action)
            return

        # Solo / Mute / Rec strip buttons (CC mode, factory default)
        for strip_idx, solo_cc in enumerate(nk2.SOLO_CC):
            if cc == solo_cc:
                if value == 127:
                    with self._lock:
                        prev = self._state.solo_active.get(strip_idx, False)
                        self._state.solo_active[strip_idx] = not prev
                    logger.debug("Solo %d toggled", strip_idx)
                return

        for strip_idx, mute_cc in enumerate(nk2.MUTE_CC):
            if cc == mute_cc:
                if value == 127:
                    with self._lock:
                        prev = self._state.mute_active.get(strip_idx, False)
                        self._state.mute_active[strip_idx] = not prev
                    logger.debug("Mute %d toggled", strip_idx)
                return

        for strip_idx, rec_cc in enumerate(nk2.REC_CC):
            if cc == rec_cc:
                if value == 127:
                    with self._lock:
                        prev = self._state.rec_active.get(strip_idx, False)
                        self._state.rec_active[strip_idx] = not prev
                    logger.debug("Rec %d toggled", strip_idx)
                return

    def _handle_note(self, note: int, pressed: bool) -> None:
        """Process a Note On/Off message (for button-mode configurations)."""
        # nanoKONTROL2 factory default uses CC, not notes, but some users
        # reprogram it to send notes for buttons.  Log and ignore for now.
        logger.debug("Note %d pressed=%s (unhandled)", note, pressed)

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_state(self) -> MidiState:
        """Return a thread-safe snapshot of the current MIDI parameter state."""
        with self._lock:
            s = self._state
            snap = MidiState(
                master_intensity=s.master_intensity,
                warp_amount=s.warp_amount,
                feedback_decay=s.feedback_decay,
                zoom=s.zoom,
                symmetry=s.symmetry,
                color_shift=s.color_shift,
                glow=s.glow,
                mix=s.mix,
                warp_speed=s.warp_speed,
                rotation=s.rotation,
                noise_amount=s.noise_amount,
                edge_gain=s.edge_gain,
                audio_sensitivity=s.audio_sensitivity,
                palette_shift=s.palette_shift,
                trail_length=s.trail_length,
                brightness=s.brightness,
                solo_active=dict(s.solo_active),
                mute_active=dict(s.mute_active),
                rec_active=dict(s.rec_active),
                playing=s.playing,
                recording=s.recording,
            )
        return snap
