"""Render an Dragonoff project through Surge XT and explicit bundled drum samples.

This is an optional local backend. It does not evaluate compositions or invent
notes. Every hit and note must already be present in the editable project.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import (
    Pedalboard, BrickwallLimiter, Chorus, Compressor, Delay, Gain,
    HighpassFilter, HighShelfFilter, LowShelfFilter, PeakFilter, PitchShift, Reverb,
)
from scipy import signal

from plugin_render import render_vst, write_midi

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "assets/samples/manifest.json").read_text())['samples']
DRUM_ROLES = {"kick", "hat", "hat_16", "open_hat", "clap", "snare", "perc", "ride"}

# Surge XT raw-parameter maps are intentionally small and role-specific. They
# produce a controlled analog-style source, not a hidden auto-selected preset.
PATCHES = {
    "bass": {
        # Clean, centered analog sub with just enough upper harmonic to read
        # on small speakers. The low kick remains the transient anchor.
        "a_osc_1_type": 0.09090909, "a_osc_2_type": 0.0,
        "a_osc_1_volume": 0.88, "a_osc_2_volume": 0.20,
        "a_osc_1_unison_voices": 0.0, "a_osc_1_unison_detune": 0.0,
        "a_filter_1_type": 0.06060606, "a_filter_1_cutoff": 0.29,
        "a_filter_1_resonance": 0.045, "a_filter_1_feg_mod_amount": 0.12,
        "a_filter_eg_attack": 0.0, "a_filter_eg_decay": 0.24,
        "a_filter_eg_sustain": 0.08, "a_filter_eg_release": 0.16,
        "a_amp_eg_attack": 0.005, "a_amp_eg_decay": 0.26,
        "a_amp_eg_sustain": 0.38, "a_amp_eg_release": 0.14,
        "a_pre_filter_gain": 0.43, "a_waveshaper_drive": 0.16,
    },
    "lead": {
        # A subtractive techno voice: saw carries the fundamental and a
        # restrained square layer adds odd harmonics. Keep FM/triangle out of
        # the lead default so short riffs do not bloom like a doorbell.
        "a_osc_1_type": 0.0, "a_osc_2_type": 0.27272728,
        "a_osc_1_volume": 0.72, "a_osc_2_volume": 0.22,
        "a_osc_1_unison_voices": 0.0, "a_osc_1_unison_detune": 0.0,
        "a_filter_1_type": 0.06060606, "a_filter_1_cutoff": 0.32,
        "a_filter_1_resonance": 0.045, "a_filter_1_feg_mod_amount": 0.27,
        "a_filter_eg_attack": 0.0, "a_filter_eg_decay": 0.25,
        "a_filter_eg_sustain": 0.08, "a_filter_eg_release": 0.13,
        "a_amp_eg_attack": 0.005, "a_amp_eg_decay": 0.28,
        "a_amp_eg_sustain": 0.31, "a_amp_eg_release": 0.12,
        "a_pre_filter_gain": 0.50, "a_waveshaper_drive": 0.58,
    },
    "pad": {
        # Slow, dark analog/string blend: sustained harmonic support, not a
        # bright pluck or arpeggiator-like part.
        "a_osc_1_type": 0.0, "a_osc_2_type": 0.81818181,
        "a_osc_1_volume": 0.76, "a_osc_2_volume": 0.24,
        "a_osc_1_unison_voices": 0.06666667, "a_osc_1_unison_detune": 0.045,
        "a_filter_1_type": 0.06060606, "a_filter_1_cutoff": 0.27,
        "a_filter_1_resonance": 0.025, "a_filter_1_feg_mod_amount": 0.04,
        "a_filter_eg_attack": 0.32, "a_filter_eg_decay": 0.64,
        "a_filter_eg_sustain": 0.46, "a_filter_eg_release": 0.75,
        "a_amp_eg_attack": 0.34, "a_amp_eg_decay": 0.54,
        "a_amp_eg_sustain": 0.62, "a_amp_eg_release": 0.82,
    },
    "chord": {
        "a_osc_1_type": 0.0, "a_osc_2_type": 0.09090909,
        "a_osc_1_volume": 0.72, "a_osc_2_volume": 0.28,
        "a_filter_1_type": 0.06060606, "a_filter_1_cutoff": 0.44,
        "a_filter_1_resonance": 0.035, "a_filter_1_feg_mod_amount": 0.12,
        "a_filter_eg_attack": 0.0, "a_filter_eg_decay": 0.28,
        "a_filter_eg_sustain": 0.10, "a_filter_eg_release": 0.14,
        "a_amp_eg_attack": 0.008, "a_amp_eg_decay": 0.24,
        "a_amp_eg_sustain": 0.16, "a_amp_eg_release": 0.12,
        "a_pre_filter_gain": 0.50, "a_waveshaper_drive": 0.50,
    },
    "airy": {
        "a_osc_1_type": 0.81818181, "a_osc_2_type": 0.09090909,
        "a_osc_1_volume": 0.62, "a_osc_2_volume": 0.38,
        "a_osc_1_unison_voices": 0.0, "a_osc_1_unison_detune": 0.0,
        "a_filter_1_type": 0.33333334, "a_filter_1_cutoff": 0.64,
        "a_filter_1_resonance": 0.025, "a_filter_1_feg_mod_amount": 0.06,
        "a_amp_eg_attack": 0.018, "a_amp_eg_decay": 0.52,
        "a_amp_eg_sustain": 0.40, "a_amp_eg_release": 0.72,
    },
}


def db(value: float) -> float:
    return 10.0 ** (value / 20.0)


def total_bars(song: dict) -> int:
    section_end = max((s["start_bar"] + s["bars"] - 1 for s in song.get("sections", [])), default=0)
    clip_end = max((c["start_bar"] + c["bars"] - 1 for t in song.get("tracks", []) for c in t.get("clips", [])), default=0)
    return max(section_end, clip_end, 1)


def notes_for(track: dict, song: dict) -> list[dict]:
    beats_per_bar = song.get("numerator", 4) * 4.0 / song.get("denominator", 4)
    patterns = {p["id"]: p for p in track.get("patterns", [])}
    out = []
    for clip in track.get("clips", []):
        pattern = patterns.get(clip["pattern_id"])
        if not pattern:
            continue
        plen = max(int(pattern.get("bars", 1)), 1)
        for offset in range(int(clip["bars"])):
            pat_bar = offset % plen + 1
            for note in pattern.get("notes", []):
                if int(note["bar"]) != pat_bar:
                    continue
                onset = (int(clip["start_bar"]) - 1 + offset) * beats_per_bar
                onset += (float(note["beat"]) - 1.0 + float(note.get("tick", 0)) / 480.0) * 4.0 / song.get("denominator", 4)
                pitch = int(note["pitch"]) + int(clip.get("transpose", 0))
                out.append({
                    "onset_beats": onset,
                    "duration_beats": float(note["length_beats"]),
                    "pitch": max(0, min(127, pitch)),
                    "velocity": int(note.get("velocity", 90)),
                })
    return sorted(out, key=lambda n: (n["onset_beats"], n["pitch"]))


def read_sample(sample_id: str, sample_rate: int, sample_path: str | None = None) -> np.ndarray:
    item = MANIFEST.get(sample_id)
    if sample_path:
        path = Path(sample_path)
        if not path.is_absolute():
            path = ROOT / (path if path.parts and path.parts[0] == "assets" else Path("assets") / path)
    elif item is not None:
        path = ROOT / item["path"]
    else:
        raise ValueError(f"sample ID is not installed and has no sample_path: {sample_id}")
    if not path.is_file():
        raise ValueError(f"sample file not found for {sample_id}: {path}")
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if rate != sample_rate:
        divisor = math.gcd(int(rate), int(sample_rate))
        audio = signal.resample_poly(audio, sample_rate // divisor, rate // divisor, axis=0).astype("float32")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio


def drum_stem(track: dict, song: dict, path: Path, bars: int) -> list[tuple[int, float]]:
    sr = int(song["sample_rate"])
    bpm = float(song["bpm"])
    beats_per_bar = song.get("numerator", 4) * 4.0 / song.get("denominator", 4)
    frames = round(bars * beats_per_bar * 60.0 / bpm * sr)
    out = np.zeros((frames, 2), dtype="float32")
    ids = [track.get("sample_id"), track.get("sample_alt_id")]
    paths = [track.get("sample_path"), track.get("sample_alt_path")]
    has_alternate = bool(ids[1] or paths[1])
    source_keys = [sid or f"__external_{track['id']}_primary" for sid in ids]
    samples = {
        key: read_sample(sample_id or key, sr, sample_path)
        for key, sample_id, sample_path in zip(source_keys, ids, paths)
        if sample_id or sample_path
    }
    kick_times = []
    for index, note in enumerate(notes_for(track, song)):
        slot = index % 2 if has_alternate else 0
        sample_id = source_keys[slot]
        if sample_id not in samples:
            raise ValueError(f"{track['id']}: missing explicit drum sample or sample_path")
        onset_sec = note["onset_beats"] * 60.0 / bpm
        start = round(onset_sec * sr)
        if track["role"] == "kick":
            kick_times.append((start, onset_sec))
        source = samples[sample_id]
        tune = float(track.get("sample_tune_semitones", 0.0))
        if tune:
            new_len = max(1, round(len(source) / 2.0 ** (tune / 12.0)))
            source = signal.resample(source, new_len, axis=0).astype("float32")
        limit_ms = float(track.get("drum", {}).get("decay_ms", 0.0))
        if limit_ms > 0:
            source = source[:round(limit_ms * sr / 1000.0)]
        end = min(frames, start + len(source))
        if end <= start:
            continue
        velocity = (note["velocity"] / 100.0) ** 1.1
        gain = db(float(track.get("gain_db", 0.0))) * velocity
        pan = max(-1.0, min(1.0, float(track.get("pan", 0.0))))
        gains = np.array([math.sqrt((1.0 - pan) * 0.5), math.sqrt((1.0 + pan) * 0.5)], dtype="float32") * math.sqrt(2.0)
        out[start:end] += source[:end - start] * (gain * gains)[None, :]
    sf.write(str(path), out, sr, subtype="FLOAT")
    del out
    return kick_times


def _norm_log(value: float, low: float, high: float) -> float:
    value = min(high, max(low, float(value)))
    return min(1.0, max(0.0, math.log(value / low) / math.log(high / low)))


def _surge_wave(wave: str) -> float:
    # Normalized oscillator choices in Surge XT's Classic oscillator.
    return {
        "saw": 0.0, "sine": 0.09090909, "triangle": 0.18181819,
        "square": 0.27272728, "fm": 0.36363637, "brass": 0.45454547,
        "choir": 0.63636363, "noise": 0.90909090,
    }.get(str(wave).lower(), 0.0)


def patch_for(track: dict) -> dict:
    """Resolve the base voice, then apply the track's actual Dragonoff sound patch."""
    role, track_id = track.get("role", "lead"), track.get("id", "")
    if track_id == "hook_response":
        base = PATCHES["airy"]
    elif role in ("bass", "sub", "sf2_bass", "pipe_bass", "fm_bass", "brass_bass"):
        base = PATCHES["bass"]
    elif role == "sf2_chords":
        base = PATCHES["chord"]
    elif role in ("pad", "chords"):
        base = PATCHES["pad"]
    else:
        base = PATCHES["lead"]
    patch = dict(base)
    sound = track.get("sound", {})
    layers = sound.get("layers", [])
    # Model defaults mean "inherit the role patch". Only non-default top-level
    # settings replace it; this prevents serialized defaults from flattening a
    # deliberately dark bass or a slow pad into one generic open saw patch.
    if sound.get("waveform", "saw") != "saw":
        patch["a_osc_1_type"] = _surge_wave(sound["waveform"])
    if int(sound.get("unison", 1)) != 1:
        patch["a_osc_1_unison_voices"] = min(0.46666667, max(0.0, (int(sound["unison"]) - 1) / 15.0))
    if float(sound.get("detune_cents", 0.0)) > 0:
        patch["a_osc_1_unison_detune"] = min(0.2, max(0.0, float(sound["detune_cents"]) / 500.0))
    if float(sound.get("cutoff_hz", 20_000)) < 19_999:
        patch["a_filter_1_cutoff"] = _norm_log(sound["cutoff_hz"], 20, 20_000)
    if float(sound.get("filter_resonance", 0.0)) > 0:
        patch["a_filter_1_resonance"] = min(0.95, max(0.0, float(sound["filter_resonance"])))
    if float(sound.get("filter_env_amount", 0.0)) != 0:
        patch["a_filter_1_feg_mod_amount"] = min(1.0, max(0.0, abs(float(sound["filter_env_amount"])) / 4.0))
    for field, default, parameter in (
        ("attack_ms", 8.0, "a_amp_eg_attack"),
        ("decay_ms", 180.0, "a_amp_eg_decay"),
        ("sustain", 0.72, "a_amp_eg_sustain"),
        ("release_ms", 100.0, "a_amp_eg_release"),
    ):
        value = float(sound.get(field, default))
        if abs(value - default) > 1e-6:
            patch[parameter] = min(1.0, value if field == "sustain" else value / 700.0)
    if layers:
        for slot, layer in enumerate(layers[:2], start=1):
            patch[f"a_osc_{slot}_type"] = _surge_wave(layer.get("waveform", sound.get("waveform", "saw")))
            patch[f"a_osc_{slot}_volume"] = min(1.0, max(0.0, float(layer.get("gain", 1.0))))
            if slot == 1:
                patch["a_osc_1_unison_voices"] = min(0.46666667, max(0.0, (int(layer.get("unison", sound.get("unison", 1))) - 1) / 15.0))
                patch["a_osc_1_unison_detune"] = min(0.2, max(0.0, float(layer.get("detune_cents", sound.get("detune_cents", 0))) / 500.0))
    # Explicit VST raw parameters are the escape hatch from role presets. The
    # plugin catalog exposes exact IDs; render_vst validates IDs and ranges.
    for key, value in track.get("plugin_parameters", {}).items():
        patch[str(key)] = float(value)
    return patch


def track_board(track: dict, song: dict, apply_track_gain: bool = True) -> tuple[Pedalboard, dict]:
    fx = track.get("effects", {})
    automated = {p.get("parameter") for p in track.get("automation", [])}
    chain = []
    controls = {}
    # Honor the per-track filter and EQ controls that the former Surge path
    # silently skipped. Import optional pedalboard processors only when used.
    if fx.get("highpass_hz"):
        from pedalboard import HighpassFilter
        chain.append(HighpassFilter(cutoff_frequency_hz=float(fx["highpass_hz"])))
    if fx.get("lowpass_hz") or "lowpass_hz" in automated:
        from pedalboard import LowpassFilter
        lowpass = LowpassFilter(cutoff_frequency_hz=float(fx.get("lowpass_hz") or 20000.0))
        chain.append(lowpass)
        controls["lowpass"] = lowpass
    if fx.get("eq"):
        from pedalboard import PeakFilter
        for band in fx["eq"]:
            if band.get("kind", "peak") not in ("peak", "bell"):
                continue
            chain.append(PeakFilter(
                cutoff_frequency_hz=float(band["frequency_hz"]),
                gain_db=float(band.get("gain_db", 0.0)),
                q=max(0.1, float(band.get("q", 0.8))),
            ))
    saturation = min(1.0, max(0.0, float(fx.get("saturation", 0.0))))
    if saturation > 0.001:
        from pedalboard import Distortion
        chain.append(Distortion(drive_db=saturation * 18.0))
    if float(fx.get("chorus_depth", 0.0)) > 0.02 or "chorus_depth" in automated:
        chorus = Chorus()
        chorus.rate_hz = max(0.05, float(fx.get("chorus_rate_hz", 0.35)))
        chorus.depth = min(0.8, float(fx["chorus_depth"]))
        chorus.mix = min(0.35, float(fx["chorus_depth"]) * 0.28)
        chain.append(chorus)
        controls["chorus"] = chorus
    if float(fx.get("delay_wet", 0.0)) > 0.01 or "delay_wet" in automated:
        delay = Delay()
        delay.delay_seconds = (float(fx.get("delay_ms", 0.0)) / 1000.0
                               if float(fx.get("delay_ms", 0.0)) > 0.0
                               else float(fx.get("delay_beats") or 0.75) * 60.0 / float(song["bpm"]))
        delay.feedback = min(0.7, float(fx.get("delay_feedback", 0.0)))
        delay.mix = min(0.4, float(fx["delay_wet"]))
        chain.append(delay)
        controls["delay"] = delay
    send = float(fx.get("reverb_send", 0.0))
    if send > 0.01 or "reverb_send" in automated:
        reverb = Reverb()
        reverb.room_size = 0.90 if track.get("role") == "pad" else (0.86 if track.get("id") == "hook_response" else (0.80 if track.get("role") == "sf2_lead" else 0.62))
        reverb.damping = 0.38 if track.get("role") == "pad" else (0.42 if track.get("id") == "hook_response" else 0.48)
        reverb.wet_level = min(0.60, send * 0.75)
        reverb.dry_level = 1.0
        reverb.width = 1.0
        chain.append(reverb)
        controls["reverb"] = reverb
    if float(fx.get("compressor_ratio", 1.0)) > 1.001:
        compressor = Compressor(
            threshold_db=float(fx.get("compressor_threshold_db", -6.0)),
            ratio=float(fx.get("compressor_ratio", 1.0)),
            attack_ms=float(fx.get("compressor_attack_ms", 10.0)),
            release_ms=float(fx.get("compressor_release_ms", 120.0)),
        )
        chain.append(compressor)
    if apply_track_gain:
        chain.append(Gain(gain_db=float(track.get("gain_db", 0.0)) + float(track.get("sound", {}).get("output_gain_db", 0.0))))
    return Pedalboard(chain), controls


def automation_value(track: dict, parameter: str, position_beats: float, fallback: float, song: dict) -> float:
    beats_per_bar = song.get("numerator", 4) * 4.0 / song.get("denominator", 4)
    points = []
    for point in track.get("automation", []):
        if point.get("parameter") == parameter:
            position = (int(point["bar"]) - 1) * beats_per_bar
            position += (float(point.get("beat", 1.0)) - 1.0) * 4.0 / song.get("denominator", 4)
            points.append((position, float(point["value"])))
    points.sort()
    if not points or position_beats <= points[0][0]:
        return points[0][1] if points else fallback
    for (left, lv), (right, rv) in zip(points, points[1:]):
        if left <= position_beats <= right:
            fraction = (position_beats - left) / max(right - left, 1e-9)
            return lv + (rv - lv) * fraction
    return points[-1][1]



def stereo_image_state(track: dict, sample_rate: int) -> dict:
    """State for a mono-compatible, high-passed M/S widener and ITD pan."""
    role = track.get("role", "")
    settings = {
        "sf2_bass": (0.85, 0.00, 7.0),
        "sf2_lead": (1.45, 0.36, 15.0),
        "sf2_chords": (1.50, 0.28, 13.0),
        "pad": (1.95, 0.48, 24.0),
    }
    width, decorrelation, delay_ms = settings.get(role, (1.0, 0.0, 8.0))
    if track.get("id") == "hook_response":
        width, decorrelation, delay_ms = (1.65, 0.42, 22.0)
    delay_samples = max(1, round(sample_rate * delay_ms / 1000.0))
    b, a = signal.butter(1, 180.0, btype="highpass", fs=sample_rate)
    itd = float(track.get("sound", {}).get("binaural_itd_ms", 0.0))
    itd_samples = max(0, round(sample_rate * itd / 1000.0))
    return {
        "width": width,
        "decorrelation": decorrelation,
        "delay_history": np.zeros(delay_samples, dtype="float32"),
        "b": b,
        "a": a,
        "filter_state": np.zeros(max(len(a), len(b)) - 1, dtype="float32"),
        "itd_samples": itd_samples,
        "itd_history": np.zeros(itd_samples, dtype="float32"),
    }


def apply_stereo_image(audio: np.ndarray, track: dict, state: dict) -> np.ndarray:
    """Apply per-track pan, side width and a small far-ear delay."""
    mid = (audio[:, 0] + audio[:, 1]) * 0.5
    side = (audio[:, 0] - audio[:, 1]) * 0.5 * state["width"]
    history = state["delay_history"]
    extended = np.concatenate((history, mid))
    delayed = extended[:len(mid)]
    state["delay_history"] = extended[-len(history):]
    decorrelated, state["filter_state"] = signal.lfilter(
        state["b"], state["a"], mid - delayed, zi=state["filter_state"]
    )
    side += decorrelated * state["decorrelation"]
    left, right = mid + side, mid - side

    pan = max(-1.0, min(1.0, float(track.get("pan", 0.0))))
    left *= math.sqrt(1.0 - pan)
    right *= math.sqrt(1.0 + pan)
    itd_samples = state["itd_samples"]
    if itd_samples and pan:
        history = state["itd_history"]
        if pan < 0:
            extended = np.concatenate((history, right))
            right = extended[:len(right)]
        else:
            extended = np.concatenate((history, left))
            left = extended[:len(left)]
        state["itd_history"] = extended[-itd_samples:]
    return np.column_stack((left, right)).astype("float32", copy=False)


def process_sample_track(source_path: Path, output_path: Path, track: dict, song: dict) -> None:
    """Apply configured track effects to one-shot stems without applying gain/pan twice."""
    sr = int(song["sample_rate"])
    board, controls = track_board(track, song, apply_track_gain=False)
    image_track = dict(track)
    image_track["pan"] = 0.0  # drum_stem already applied the track pan and gain per hit
    image_state = stereo_image_state(image_track, sr)
    with sf.SoundFile(str(source_path)) as source, sf.SoundFile(
        str(output_path), "w", sr, 2, subtype="FLOAT"
    ) as dest:
        first = True
        while True:
            offset = source.tell()
            block = source.read(4096, dtype="float32", always_2d=True)
            if len(block) == 0:
                break
            beat_position = offset / sr * float(song["bpm"]) / 60.0
            if "lowpass" in controls:
                cutoff = automation_value(
                    track, "lowpass_hz", beat_position,
                    float(track.get("effects", {}).get("lowpass_hz") or 20000.0), song,
                )
                controls["lowpass"].cutoff_frequency_hz = min(20000.0, max(20.0, cutoff))
            if "chorus" in controls:
                depth = automation_value(
                    track, "chorus_depth", beat_position,
                    float(track.get("effects", {}).get("chorus_depth", 0.0)), song,
                )
                controls["chorus"].depth = min(0.8, max(0.0, depth))
                controls["chorus"].mix = min(0.35, max(0.0, depth) * 0.28)
            if "delay" in controls:
                wet = automation_value(
                    track, "delay_wet", beat_position,
                    float(track.get("effects", {}).get("delay_wet", 0.0)), song,
                )
                controls["delay"].mix = min(0.4, max(0.0, wet))
            if "reverb" in controls:
                send = automation_value(
                    track, "reverb_send", beat_position,
                    float(track.get("effects", {}).get("reverb_send", 0.0)), song,
                )
                controls["reverb"].wet_level = min(0.35, max(0.0, send) * 0.75)
            audio = board(block.T.copy(), sr, reset=first).T
            audio = apply_stereo_image(audio, image_track, image_state)
            first = False
            dest.write(audio)


def master_stereo_width(audio: np.ndarray, width: float, low_band_width: float, state: dict) -> np.ndarray:
    """Expand the side image above the low-frequency mono crossover."""
    mid = (audio[:, 0] + audio[:, 1]) * 0.5
    side = (audio[:, 0] - audio[:, 1]) * 0.5
    low_side, state["zi"] = signal.sosfilt(state["sos"], side, zi=state["zi"])
    side = low_side * low_band_width + (side - low_side) * width
    return np.column_stack((mid + side, mid - side)).astype("float32", copy=False)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("project", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--plugin", type=Path, required=True)
    args = ap.parse_args()
    song = json.loads(args.project.read_text())
    bars, sr, bpm = total_bars(song), int(song["sample_rate"]), float(song["bpm"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="surge-render-", dir=args.output.parent) as temp_name:
        temp = Path(temp_name)
        rendered = []
        kick_events = []
        for track in song.get("tracks", []):
            if track.get("muted") or not track.get("clips"):
                continue
            notes = notes_for(track, song)
            if not notes:
                continue
            stem = temp / f"{track['id']}.wav"
            if (track.get("source_type") != "synth" and track.get("role") in DRUM_ROLES
                    and (track.get("sample_id") or track.get("sample_path"))):
                events = drum_stem(track, song, stem, bars)
                if track["role"] == "kick":
                    kick_events = events
                processed = temp / f"{track['id']}-processed.wav"
                process_sample_track(stem, processed, track, song)
                stem = processed
            else:
                vst_track = {
                    "id": track["id"], "name": track["name"], "source": "vst3",
                    "notes": notes, "patch": patch_for(track),
                    "automation": {},
                }
                session = {"sample_rate": sr, "bpm": bpm, "bars": bars}
                render_vst(vst_track, session, stem, args.plugin)
                board, controls = track_board(track, song)
                image_state = stereo_image_state(track, sr)
                # Surge's VST path does not consume the model's native pitch-LFO
                # controls. Build a parallel, wet-only echo for tracks that ask
                # for pitch motion so their tails can sweep downward without
                # bending the written notes themselves.
                sound = track.get("sound", {})
                pitch_echo_enabled = sound.get("lfo_target") == "pitch"
                pitch_echo_delay = pitch_echo = None
                if pitch_echo_enabled:
                    fx = track.get("effects", {})
                    pitch_echo_delay = Delay()
                    pitch_echo_delay.delay_seconds = (
                        float(fx.get("delay_ms", 0.0)) / 1000.0
                        if float(fx.get("delay_ms", 0.0)) > 0.0
                        else float(fx.get("delay_beats") or 0.75) * 60.0 / bpm
                    )
                    pitch_echo_delay.feedback = min(0.78, float(fx.get("delay_feedback", 0.0)))
                    pitch_echo_delay.mix = 1.0  # wet-only; the dry synth remains untouched
                    pitch_echo = PitchShift(semitones=0.0)
                processed = temp / f"{track['id']}-processed.wav"
                with sf.SoundFile(str(stem)) as source, sf.SoundFile(str(processed), "w", sr, 2, subtype="FLOAT") as dest:
                    first = True
                    while True:
                        offset = source.tell()
                        block = source.read(4096, dtype="float32", always_2d=True)
                        if len(block) == 0:
                            break
                        beat_position = offset / sr * bpm / 60.0
                        if "chorus" in controls:
                            depth = automation_value(track, "chorus_depth", beat_position, float(track.get("effects", {}).get("chorus_depth", 0.0)), song)
                            controls["chorus"].depth = min(0.8, max(0.0, depth))
                            controls["chorus"].mix = min(0.35, max(0.0, depth) * 0.28)
                        if "delay" in controls:
                            wet = automation_value(track, "delay_wet", beat_position, float(track.get("effects", {}).get("delay_wet", 0.0)), song)
                            controls["delay"].mix = min(0.4, max(0.0, wet))
                        if "reverb" in controls:
                            send = automation_value(track, "reverb_send", beat_position, float(track.get("effects", {}).get("reverb_send", 0.0)), song)
                            controls["reverb"].wet_level = min(0.35, max(0.0, send) * 0.75)
                        audio = board(block.T.copy(), sr, reset=first).T
                        if pitch_echo_enabled:
                            depth = automation_value(track, "lfo_depth", beat_position, float(sound.get("lfo_depth", 2.4)), song)
                            rate = automation_value(track, "lfo_rate_hz", beat_position, float(sound.get("lfo_rate_hz", 0.14)), song)
                            bias = automation_value(track, "lfo_bias", beat_position, float(sound.get("lfo_bias", -0.3)), song)
                            phase = 2.0 * math.pi * max(0.01, rate) * (offset / sr)
                            # A slow cyclic dive with a negative bias: the echo
                            # falls several semitones, then eases back before
                            # the next descent.
                            fall = (1.0 - math.sin(phase)) * 0.5
                            semitones = -max(0.0, depth) * (fall + max(0.0, min(1.0, -bias)) * 0.25)
                            pitch_echo.semitones = max(-12.0, min(0.0, semitones))
                            echo = pitch_echo_delay(audio.T.copy(), sr, reset=first).T
                            echo = pitch_echo(echo.T.copy(), sr, reset=first).T
                            if len(echo) < len(audio):
                                echo = np.pad(echo, ((0, len(audio) - len(echo)), (0, 0)))
                            elif len(echo) > len(audio):
                                echo = echo[:len(audio)]
                            send = automation_value(
                                track, "delay_wet", beat_position,
                                float(track.get("effects", {}).get("delay_wet", 0.0)), song,
                            )
                            audio += echo * min(0.48, max(0.0, send) * 0.85)
                        audio = apply_stereo_image(audio, track, image_state)
                        first = False
                        dest.write(audio)
                stem = processed
            rendered.append((track, stem))

        frames = round(bars * 4 * 60.0 / bpm * sr)
        duck = np.ones(frames, dtype="float32")
        # Use the actual written kick lane to duck only tracks that request it.
        amount = max((float(t.get("effects", {}).get("sidechain_amount", 0.0)) for t in song.get("tracks", []) if t.get("effects", {}).get("sidechain")), default=0.0)
        if kick_events and amount > 0:
            release = round(0.16 * sr)
            curve = 1.0 - amount * np.exp(-np.arange(release, dtype="float32") / (0.045 * sr))
            for start, _ in kick_events:
                end = min(frames, start + release)
                if end > start:
                    duck[start:end] = np.minimum(duck[start:end], curve[:end - start])

        mastering = song.get("master", {})
        # Keep the Surge backend aligned with the editable project's master
        # strip. These controls used to be silently replaced by fixed values.
        master_eq = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=float(mastering.get("eq_highpass_hz", 24.0))),
            LowShelfFilter(cutoff_frequency_hz=110.0,
                           gain_db=float(mastering.get("eq_low_shelf_db", 0.0)), q=0.7071),
            PeakFilter(cutoff_frequency_hz=float(mastering.get("eq_low_mid_hz", 240.0)),
                       gain_db=float(mastering.get("eq_low_mid_db", 0.0)),
                       q=float(mastering.get("eq_low_mid_q", 0.85))),
            PeakFilter(cutoff_frequency_hz=float(mastering.get("eq_presence_hz", 2800.0)),
                       gain_db=float(mastering.get("eq_presence_db", 0.0)),
                       q=float(mastering.get("eq_presence_q", 0.75))),
            HighShelfFilter(cutoff_frequency_hz=11000.0,
                            gain_db=float(mastering.get("eq_high_shelf_db", 0.0)), q=0.7071),
            Compressor(
                threshold_db=float(mastering.get("compressor_threshold_db", -12.0)),
                ratio=float(mastering.get("compressor_ratio", 1.5)),
                attack_ms=float(mastering.get("compressor_attack_ms", 20.0)),
                release_ms=float(mastering.get("compressor_release_ms", 90.0)),
            ),
        ])
        output_stage = Pedalboard([
            # A gentle soft clipper adds density before the true-peak limiter.
            # The drive is normalized so full-scale level stays predictable.
            Gain(gain_db=-2.0 + float(mastering.get("output_gain_db", 0.0))),
            BrickwallLimiter(
                ceiling_db=float(mastering.get("limiter_ceiling_db", -1.2)),
                release_ms=float(mastering.get("limiter_release_ms", 100.0)),
                lookahead_ms=float(mastering.get("limiter_lookahead_ms", 5.0)),
                true_peak=True,
            ),
        ])
        clipper_drive = max(0.0, float(mastering.get("clipper_drive_db", 0.0)))
        clipper_gain = 10.0 ** (clipper_drive / 20.0)
        width_sos = signal.butter(2, 150.0, btype="lowpass", fs=sr, output="sos")
        width_state = {"sos": width_sos, "zi": np.zeros((len(width_sos), 2), dtype="float32")}
        stereo_width = float(mastering.get("stereo_width", 1.0))
        low_band_width = float(mastering.get("low_band_width", 0.0))
        peak = 0.0
        squares = 0.0
        count = 0
        with sf.SoundFile(str(args.output), "w", sr, 2, subtype="PCM_24") as dest:
            first = True
            for offset in range(0, frames, 16384):
                size = min(16384, frames - offset)
                mix = np.zeros((size, 2), dtype="float32")
                for track, path in rendered:
                    with sf.SoundFile(str(path)) as source:
                        source.seek(offset)
                        block = source.read(size, dtype="float32", always_2d=True)
                    if len(block) < size:
                        block = np.pad(block, ((0, size - len(block)), (0, 0)))
                    if any(p.get("parameter") == "gain_db" for p in track.get("automation", [])):
                        # The source stem already contains the track's static
                        # gain. Apply the saved gain lane as an offset from it,
                        # matching the native renderer's absolute gain points.
                        base_gain = float(track.get("gain_db", 0.0))
                        start_beat = offset / sr * float(song["bpm"]) / 60.0
                        end_beat = (offset + size) / sr * float(song["bpm"]) / 60.0
                        start_gain = automation_value(track, "gain_db", start_beat, base_gain, song)
                        end_gain = automation_value(track, "gain_db", end_beat, base_gain, song)
                        gain_db_curve = np.linspace(
                            start_gain - base_gain,
                            end_gain - base_gain,
                            size,
                            endpoint=False,
                            dtype="float32",
                        )
                        block *= (10.0 ** (gain_db_curve / 20.0))[:, None]
                    if track.get("effects", {}).get("sidechain"):
                        block *= duck[offset:offset + size, None]
                    mix += block
                mix = master_stereo_width(mix, stereo_width, low_band_width, width_state)
                mix = master_eq(mix.T.copy(), sr, reset=first).T
                if clipper_drive > 0.0:
                    norm = np.tanh(clipper_gain)
                    mix = np.tanh(mix * clipper_gain) / max(norm, 1e-6)
                mix = output_stage(mix.T.copy(), sr, reset=first).T
                first = False
                peak = max(peak, float(np.max(np.abs(mix))))
                squares += float(np.sum(mix.astype("float64") ** 2))
                count += mix.size
                dest.write(mix)
        report = {
            "output": str(args.output), "renderer": "Surge XT VST3 + explicit bundled samples",
            "plugin": str(args.plugin), "sample_rate": sr, "bars": bars,
            "duration_seconds": frames / sr, "peak": peak,
            "rms": math.sqrt(squares / max(count, 1)),
            "tracks": [t["id"] for t, _ in rendered],
            "musical_quality": "not evaluated automatically",
        }
        args.output.with_suffix(args.output.suffix + ".render.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
