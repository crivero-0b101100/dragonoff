#!/usr/bin/env python3
"""Optional pretrained source-separation and note-transcription pass.

Requires the optional packages listed in requirements-track-analysis.txt.
The script writes separated stems and MIDI/CSV transcriptions under --output,
then prints a JSON analysis report to stdout.
"""

import argparse
import bisect
import contextlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path


PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _relative_envelope(samples, count):
    import numpy as np

    values = [
        float(np.sqrt(np.mean(chunk * chunk))) if len(chunk) else 0.0
        for chunk in np.array_split(samples, count)
    ]
    peak = max(values, default=0.0)
    if peak <= 1e-12:
        return [0.0] * count
    return [round(min(1.0, value / peak), 4) for value in values]


def _spectral_summary(mono, sample_rate):
    import numpy as np

    if len(mono) < 16:
        return {"centroid_hz": None, "flatness": None, "energy_share_low_mid_high": [0.0, 0.0, 0.0]}
    window = np.hanning(len(mono))
    power = np.abs(np.fft.rfft(mono * window)) ** 2
    frequencies = np.fft.rfftfreq(len(mono), 1.0 / sample_rate)
    total = float(power.sum()) or 1e-20
    shares = [
        float(power[(frequencies >= low) & (frequencies < high)].sum() / total)
        for low, high in ((20.0, 180.0), (180.0, 4_000.0), (4_000.0, sample_rate / 2.0))
    ]
    flatness = float(np.exp(np.mean(np.log(power + 1e-20))) / (float(power.mean()) + 1e-20))
    return {
        "centroid_hz": round(float((frequencies * power).sum() / total), 1),
        "flatness": round(flatness, 5),
        "energy_share_low_mid_high": [round(value, 5) for value in shares],
    }


def read_audio(path, mono=True):
    import librosa
    import numpy as np

    audio, sample_rate = librosa.load(path, sr=None, mono=mono)
    if len(audio) == 0:
        raise ValueError(f"empty audio file: {path}")
    if not mono:
        if audio.ndim == 1:
            audio = audio[:, None]
        else:
            audio = audio.T
        return audio.astype(np.float32, copy=False), sample_rate
    return audio.astype(np.float32, copy=False), sample_rate


def event_sound_observation(audio, sample_rate, start, end, pitch_midi=None):
    """Measure a note's audible shape; this is evidence, not a recovered patch."""
    import numpy as np

    frames = len(audio)
    left = max(0, min(frames, int(start * sample_rate)))
    right = max(left + 1, min(frames, int(end * sample_rate)))
    segment = audio[left:right]
    mono = segment.mean(axis=1)
    rms = float(np.sqrt(np.mean(mono * mono))) if len(mono) else 0.0
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    if len(mono) < 32:
        return {"rms_dbfs": -120.0, "peak_dbfs": -120.0, "measurement_status": "window_too_short"}
    envelope_shape = _relative_envelope(mono, 12)
    if rms < 1e-6:
        return {
            "rms_dbfs": -120.0,
            "peak_dbfs": -120.0,
            "spectral_centroid_hz": 0.0,
            "spectral_flatness": 0.0,
            "harmonic_energy_ratio_at_pitch": 0.0 if pitch_midi is not None else None,
            "energy_share_low_mid_high": [0.0, 0.0, 0.0],
            "attack_to_80_percent_peak_ms_estimate": None,
            "post_note_tail_to_event_rms_db_estimate": -120.0,
            "stereo_side_to_mid_db": -120.0,
            "stereo_channel_correlation": 1.0,
            "relative_amplitude_envelope_12_points": envelope_shape,
            "harmonic_energy_share_partial_1_to_8": [0.0] * 8 if pitch_midi is not None else None,
            "measurement_status": "silent_window_no_timbre_estimate",
        }

    window = np.hanning(len(mono))
    power = np.abs(np.fft.rfft(mono * window)) ** 2
    frequencies = np.fft.rfftfreq(len(mono), 1.0 / sample_rate)
    total_power = float(power.sum()) or 1e-20
    centroid = float((frequencies * power).sum() / total_power)
    geometric = float(np.exp(np.mean(np.log(power + 1e-20))))
    flatness = geometric / (float(power.mean()) + 1e-20)
    bands = []
    for low, high in ((20.0, 180.0), (180.0, 4_000.0), (4_000.0, sample_rate / 2.0)):
        bands.append(float(power[(frequencies >= low) & (frequencies < high)].sum() / total_power))
    harmonic_ratio = None
    harmonic_shares = None
    if pitch_midi is not None:
        fundamental = 440.0 * 2.0 ** ((int(pitch_midi) - 69) / 12.0)
        harmonic_energy = 0.0
        harmonic_parts = []
        for harmonic in range(1, int((sample_rate / 2.0) / fundamental) + 1):
            target = fundamental * harmonic
            width = max(1, int(round(target * len(mono) / sample_rate * 0.015)))
            center = int(round(target * len(mono) / sample_rate))
            part_energy = float(power[max(0, center - width) : min(len(power), center + width + 1)].sum())
            harmonic_energy += part_energy
            if harmonic <= 8:
                harmonic_parts.append(part_energy)
        harmonic_ratio = round(min(1.0, harmonic_energy / total_power), 5)
        harmonic_shares = [round(min(1.0, value / total_power), 5) for value in harmonic_parts]
        harmonic_shares.extend([0.0] * (8 - len(harmonic_shares)))

    # Estimate attack from the event's short-window RMS envelope.
    window_size = max(1, int(sample_rate * 0.01))
    attack_region = mono[: min(len(mono), int(sample_rate * 0.15))]
    envelope = [
        float(np.sqrt(np.mean(chunk * chunk)))
        for chunk in (attack_region[i : i + window_size] for i in range(0, len(attack_region), window_size))
        if len(chunk)
    ]
    envelope_peak = max(envelope, default=0.0)
    attack_ms = next(
        (index * 10 for index, value in enumerate(envelope) if value >= envelope_peak * 0.8),
        None,
    ) if envelope_peak > 0 else None

    tail_start = max(0, min(frames, int(end * sample_rate)))
    tail_end = min(frames, tail_start + int(sample_rate * 0.25))
    tail = audio[tail_start:tail_end].mean(axis=1) if tail_end > tail_start else np.zeros(0)
    tail_rms = float(np.sqrt(np.mean(tail * tail))) if len(tail) else 0.0
    tail_envelope = _relative_envelope(tail, 12) if len(tail) else [0.0] * 12
    release_index = next((i for i, value in enumerate(tail_envelope) if value <= 0.2), None)
    stage_profiles = {}
    for name, first, last in (("attack", 0.0, 0.2), ("body", 0.2, 0.8), ("late", 0.8, 1.0)):
        left_index = min(len(mono) - 1, int(len(mono) * first))
        right_index = max(left_index + 1, min(len(mono), int(len(mono) * last)))
        stage_profiles[name] = _spectral_summary(mono[left_index:right_index], sample_rate)
    decay_index = next((i for i, value in enumerate(envelope_shape) if value >= 0.8), 0)
    decay_crossing = next(
        (i for i in range(decay_index + 1, len(envelope_shape)) if envelope_shape[i] <= envelope_shape[decay_index] * 0.6),
        None,
    )
    decay_ms = round((decay_crossing - decay_index) * len(mono) / 12 / sample_rate * 1000.0, 1) if decay_crossing is not None else None
    sustain_level = round(float(np.median(envelope_shape[6:])), 4)
    mid = (segment[:, 0] + segment[:, -1]) * 0.5
    side = (segment[:, 0] - segment[:, -1]) * 0.5
    mid_rms = float(np.sqrt(np.mean(mid * mid))) if len(mid) else 0.0
    side_rms = float(np.sqrt(np.mean(side * side))) if len(side) else 0.0
    correlation = float(np.corrcoef(segment[:, 0], segment[:, -1])[0, 1]) if segment.shape[1] > 1 else 1.0
    if not math.isfinite(correlation):
        correlation = 1.0

    return {
        "rms_dbfs": round(20.0 * math.log10(max(rms, 1e-6)), 2),
        "peak_dbfs": round(20.0 * math.log10(max(peak, 1e-6)), 2),
        "spectral_centroid_hz": round(centroid, 1),
        "spectral_flatness": round(float(flatness), 5),
        "harmonic_energy_ratio_at_pitch": harmonic_ratio,
        "energy_share_low_mid_high": [round(value, 5) for value in bands],
        "attack_to_80_percent_peak_ms_estimate": attack_ms,
        "decay_60_percent_ms_estimate": decay_ms,
        "sustain_level_relative_estimate": sustain_level,
        "release_to_20_percent_tail_ms_estimate": round((release_index + 1) * 0.25 / 12 * 1000.0, 1) if release_index is not None else None,
        "relative_amplitude_envelope_12_points": envelope_shape,
        "post_note_tail_envelope_12_points": tail_envelope,
        "spectral_profile_by_event_stage": stage_profiles,
        "post_note_tail_to_event_rms_db_estimate": round(20.0 * math.log10(max(tail_rms, 1e-6) / max(rms, 1e-6)), 2),
        "stereo_side_to_mid_db": round(20.0 * math.log10(max(side_rms, 1e-6) / max(mid_rms, 1e-6)), 2) if side_rms > 1e-6 else -120.0,
        "stereo_channel_correlation": round(correlation, 5),
        "harmonic_energy_share_partial_1_to_8": harmonic_shares,
        "measurement_status": "measured_audio_window; overlap_and_reverb_tails_can_affect_estimates",
    }


def classify_drum_event(audio, sample_rate, time_seconds):
    import numpy as np

    start = max(0, int((time_seconds - 0.035) * sample_rate))
    end = min(len(audio), int((time_seconds + 0.065) * sample_rate))
    segment = audio[start:end]
    if len(segment) < 16:
        # Keep the classifier's return shape stable for events at the file
        # edges, where the analysis window may contain only a few samples.
        return "snare/percussion-like", [0.0, 1.0, 0.0]
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment)))) ** 2
    frequencies = np.fft.rfftfreq(len(segment), 1 / sample_rate)
    bands = [
        float(spectrum[(frequencies >= 25) & (frequencies < 180)].sum()),
        float(spectrum[(frequencies >= 180) & (frequencies < 4_000)].sum()),
        float(spectrum[frequencies >= 4_000].sum()),
    ]
    label = ["kick-like", "snare/percussion-like", "hat/cymbal-like"][int(np.argmax(bands))]
    total = sum(bands) or 1.0
    return label, [value / total for value in bands]


def drum_hit_observation(audio, sample_rate, time_seconds, band_share, label):
    """Capture the transient, decay, and pitch sweep of one separated drum hit."""
    import numpy as np

    onset = max(0, int(time_seconds * sample_rate))
    tail_end = min(len(audio), onset + int(sample_rate * 0.32))
    hit = audio[onset:tail_end]
    if len(hit) < 16:
        return {"measurement_status": "hit_window_too_short", "synth_control_estimate": {}}

    envelope = _relative_envelope(hit, 24)
    peak_index = max(range(len(envelope)), key=envelope.__getitem__)
    decay_index = next((i for i in range(peak_index + 1, len(envelope)) if envelope[i] <= 0.2), None)
    decay_ms = round((decay_index - peak_index) * len(hit) / 24 / sample_rate * 1000.0, 1) if decay_index is not None else 240.0
    frame_length = min(4096, max(1024, int(0.05 * sample_rate)))
    frame_length += frame_length % 2
    hop = max(1, int(0.0125 * sample_rate))
    trajectory = []
    for start in range(0, min(len(hit), int(sample_rate * 0.16)), hop):
        end = min(len(hit), start + frame_length)
        frame = hit[start:end]
        if len(frame) < 256:
            break
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame)), n=4096)) ** 2
        frequencies = np.fft.rfftfreq(4096, 1.0 / sample_rate)
        low = np.flatnonzero((frequencies >= 25.0) & (frequencies <= 220.0))
        if len(low) and float(spectrum[low].sum()) > float(spectrum.sum()) * 0.015:
            peak_bin = int(low[np.argmax(spectrum[low])])
            trajectory.append({"offset_ms": round(start / sample_rate * 1000.0, 1), "frequency_hz": round(float(frequencies[peak_bin]), 1)})

    stage_profiles = {}
    mono = hit
    for name, first, last in (("transient", 0.0, 0.04), ("early_decay", 0.04, 0.12), ("body", 0.12, 0.24), ("tail", 0.24, 0.32)):
        left = min(len(mono) - 1, int(sample_rate * first))
        right = max(left + 1, min(len(mono), int(sample_rate * last)))
        stage_profiles[name] = _spectral_summary(mono[left:right], sample_rate)

    start_pitch = next((point["frequency_hz"] for point in trajectory[:3] if point["frequency_hz"] > 0), None)
    end_pitch = next((point["frequency_hz"] for point in reversed(trajectory[-4:]) if point["frequency_hz"] > 0), None)
    pitch_drop = None
    pitch_decay_ms = None
    base_pitch_midi = None
    if start_pitch and end_pitch and start_pitch > end_pitch:
        pitch_drop = round(min(36.0, 12.0 * math.log2(start_pitch / end_pitch)), 2)
        pitch_decay_ms = round(max(5.0, min(500.0, (trajectory[-1]["offset_ms"] or 40.0) / 2.0)), 1)
    if end_pitch:
        base_pitch_midi = int(max(24, min(84, round(69.0 + 12.0 * math.log2(end_pitch / 440.0)))))
    transient = stage_profiles["transient"]
    flatness = transient.get("flatness") or 0.0
    high_share = transient["energy_share_low_mid_high"][2]
    noise_amount = round(max(0.0, min(1.0, 0.75 * flatness + 0.35 * high_share)), 3)
    peak_margin = round(max(band_share) - sorted(band_share)[-2], 5)
    controls = {
        "waveform": "sine" if label == "kick-like" else "noise",
        "pitch_envelope_semitones": pitch_drop if label == "kick-like" else 0.0,
        "pitch_decay_ms": pitch_decay_ms or 55.0,
        "attack_ms": round(max(1.0, min(40.0, 2.0 + peak_index * len(hit) / 24 / sample_rate * 1000.0)), 1),
        "decay_ms": round(max(8.0, min(900.0, decay_ms)), 1),
        "sustain": 0.0,
        "release_ms": round(max(10.0, min(500.0, decay_ms * 0.45)), 1),
        "cutoff_hz": round(max(80.0, min(18_000.0, (transient.get("centroid_hz") or 2000.0) * 1.8)), 1),
        "filter_env_amount": 0.0,
        "noise_amount": noise_amount,
    }
    return {
        "measurement_status": "measured_separated_drum_stem; overlapping_hits_and_separation_leakage_can_affect_estimates",
        "window_duration_ms": round(len(hit) / sample_rate * 1000.0, 1),
        "relative_amplitude_envelope_24_points": envelope,
        "decay_to_20_percent_ms_estimate": round(decay_ms, 1),
        "spectral_profile_by_hit_stage": stage_profiles,
        "low_frequency_peak_trajectory": trajectory,
        "pitch_start_hz_estimate": start_pitch,
        "pitch_late_hz_estimate": end_pitch,
        "synth_base_pitch_midi_estimate": base_pitch_midi,
        "pitch_drop_semitones_estimate": pitch_drop,
        "noise_amount_estimate": noise_amount,
        "band_class_margin_uncalibrated": peak_margin,
        "band_class_shares": {"low": round(band_share[0], 5), "mid": round(band_share[1], 5), "high": round(band_share[2], 5)},
        "synth_control_estimate": controls,
    }


def estimate_drum_events(drum_path, tempo_hint):
    import librosa
    import numpy as np

    audio, sample_rate = read_audio(drum_path)
    hop = 512
    envelope = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=envelope, sr=sample_rate, hop_length=hop, backtrack=False, units="frames"
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=hop)
    beat_times = np.array([], dtype=float)
    try:
        _, beat_frames = librosa.beat.beat_track(
            onset_envelope=envelope,
            sr=sample_rate,
            hop_length=hop,
            bpm=tempo_hint,
            units="frames",
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop)
    except Exception:
        beat_times = np.array([], dtype=float)
    beat_period = 60.0 / tempo_hint
    strength_scale = float(np.percentile(envelope, 95)) if len(envelope) else 1.0
    result = []
    for onset_frame, time_seconds in zip(onset_frames, onset_times):
        label, band_share = classify_drum_event(audio, sample_rate, float(time_seconds))
        hit_observation = drum_hit_observation(audio, sample_rate, float(time_seconds), band_share, label)
        if len(beat_times):
            insertion = int(np.searchsorted(beat_times, time_seconds))
            candidates = [i for i in (insertion - 1, insertion) if 0 <= i < len(beat_times)]
            nearest = min(candidates, key=lambda i: abs(beat_times[i] - time_seconds))
            beat_time = float(beat_times[nearest])
            beat = nearest + (float(time_seconds) - beat_time) / beat_period
        else:
            beat = float(time_seconds) / beat_period
            nearest = max(0, int(round(beat)))
            beat_time = nearest * beat_period
        fraction = (beat - np.floor(beat)) % 1.0
        subdivision = min(range(4), key=lambda i: abs(fraction - i / 4.0))
        result.append(
            {
                "time_seconds": round(float(time_seconds), 4),
                "nearest_beat_zero_based": nearest,
                "bar_one_based": nearest // 4 + 1,
                "beat_in_bar_one_based": nearest % 4 + 1,
                "sixteenth_position": subdivision * 4 + 1,
                "offset_from_nearest_beat_seconds": round(float(time_seconds) - beat_time, 4),
                "character_estimate": label,
                "character_candidates_with_uncalibrated_band_scores": {
                    "kick-like": round(float(band_share[0]), 5),
                    "snare/percussion-like": round(float(band_share[1]), 5),
                    "hat/cymbal-like": round(float(band_share[2]), 5),
                },
                "band_energy_share_low_mid_high": [round(x, 5) for x in band_share],
                "onset_strength_relative": round(float(envelope[onset_frame] / max(strength_scale, 1e-12)), 5),
                "synthesis_role": {
                    "kick-like": "low_synth_percussion",
                    "snare/percussion-like": "noisy_mid_synth_percussion",
                    "hat/cymbal-like": "bright_noise_synth_percussion",
                }.get(label, "unclassified_synth_percussion"),
                "sound_observation": hit_observation,
            }
        )
    return result, [round(float(value), 4) for value in beat_times]


def align_note_to_grid(start, end, beat_times, tempo):
    beat_period = 60.0 / tempo
    if len(beat_times) < 2:
        beat_float = start / beat_period
        beat_index = max(0, int(round(beat_float)))
        beat_time = beat_index * beat_period
    else:
        insertion = bisect.bisect_left(beat_times, start)
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(beat_times)]
        beat_index = min(candidates, key=lambda index: abs(beat_times[index] - start))
        beat_time = beat_times[beat_index]
        beat_float = beat_index + (start - beat_time) / beat_period
    nearest = max(0, int(round(beat_float)))
    return {
        "absolute_beat_position_zero_based_estimate": round(float(beat_float), 5),
        "bar_one_based": nearest // 4 + 1,
        "beat_in_bar_one_based": nearest % 4 + 1,
        "nearest_beat_zero_based": nearest,
        "offset_from_nearest_beat_seconds": round(start - beat_time, 4),
        "duration_in_beats_estimate": round((end - start) / beat_period, 3),
        "sixteenth_position_estimate": (int(round((beat_float - math.floor(beat_float)) * 4)) % 4) * 4 + 1,
    }


def _bass_note_synth_controls(observation, pitch_curve):
    attack = observation.get("attack_to_80_percent_peak_ms_estimate")
    decay = observation.get("decay_60_percent_ms_estimate")
    sustain = observation.get("sustain_level_relative_estimate")
    release = observation.get("release_to_20_percent_tail_ms_estimate")
    centroid = observation.get("spectral_centroid_hz") or 1800.0
    flatness = observation.get("spectral_flatness") or 0.0
    controls = {
        "attack_ms": round(max(1.0, min(1200.0, attack if attack is not None else 12.0)), 1),
        "decay_ms": round(max(10.0, min(2400.0, decay if decay is not None else 180.0)), 1),
        "sustain": round(max(0.05, min(0.95, sustain if sustain is not None else 0.65)), 3),
        "release_ms": round(max(10.0, min(1600.0, release if release is not None else 120.0)), 1),
        "cutoff_hz": round(max(80.0, min(18_000.0, centroid * 2.0)), 1),
        "noise_amount": round(max(0.0, min(0.35, flatness * 0.35)), 3),
    }
    if len(pitch_curve) >= 3 and max(pitch_curve) - min(pitch_curve) >= 0.5:
        # The native synth supports an exponential per-note pitch envelope.
        # Fit its launch amount and decay from the early part of the measured
        # Basic Pitch bend curve; retain the full curve alongside this fit.
        launch = pitch_curve[0]
        target = pitch_curve[-1]
        controls["pitch_envelope_semitones"] = round(max(-24.0, min(48.0, launch - target)), 2)
        controls["pitch_decay_ms"] = 120.0
    return controls


def note_json(events, beat_times, tempo, audio=None, sample_rate=None, stem_role=None):
    import numpy as np
    import pretty_midi

    output = []
    for start, end, pitch, amplitude, pitch_bend in sorted(events, key=lambda event: (event[0], event[2])):
        item = {
            "start_seconds": round(float(start), 4),
            "end_seconds": round(float(end), 4),
            "pitch_midi": int(pitch),
            "pitch_name": pretty_midi.note_number_to_name(int(pitch)),
            "amplitude_estimate": round(float(amplitude), 4),
            "timing_on_beat_grid": align_note_to_grid(float(start), float(end), beat_times, tempo),
        }
        if pitch_bend:
            item["pitch_bend_offset_semitones_estimate"] = [round(float(value) / 3.0, 2) for value in pitch_bend]
        if audio is not None and sample_rate:
            item["sound_observation"] = event_sound_observation(audio, sample_rate, float(start), float(end), int(pitch))
            if stem_role == "bass":
                bend_curve = [float(value) / 3.0 for value in pitch_bend] if pitch_bend is not None else []
                if bend_curve:
                    indexes = np.linspace(0, len(bend_curve) - 1, 8).round().astype(int)
                    item["pitch_trajectory_relative_semitones_8_points"] = [round(bend_curve[index], 3) for index in indexes]
                item["synth_overrides_estimate"] = _bass_note_synth_controls(item["sound_observation"], bend_curve)
        output.append(item)
    return output


def _dragonoff_note(start, end, pitch, velocity, tempo, bar_count, metadata=None, synth_overrides=None):
    beat = max(0.0, start * tempo / 60.0)
    bar_zero = int(beat // 4.0)
    if bar_zero >= bar_count:
        return None
    note = {
        "bar": bar_zero + 1,
        "beat": round((beat - bar_zero * 4.0) + 1.0, 5),
        "tick": 0,
        "pitch": int(max(0, min(127, pitch))),
        "length_beats": round(max(0.04, (end - start) * tempo / 60.0), 5),
        "velocity": int(max(1, min(127, velocity))),
        "channel": 1,
        "source_time_seconds": round(float(start), 5),
    }
    if metadata:
        note["source_observation"] = metadata
    if synth_overrides:
        note["synth_overrides"] = synth_overrides
    return note


def _part_sound_profile(stem, notes, duration, tempo, audio=None, sample_rate=None):
    import numpy as np

    by_bar = {}
    for event in notes:
        timing = event["timing_on_beat_grid"]
        by_bar.setdefault(timing["bar_one_based"], []).append(event)
    bars = int(math.ceil(duration * tempo / 240.0))
    profiles = []
    for bar in range(1, bars + 1):
        events = by_bar.get(bar, [])
        observations = [event.get("sound_observation", {}) for event in events]
        centroid = [item["spectral_centroid_hz"] for item in observations if "spectral_centroid_hz" in item]
        flatness = [item["spectral_flatness"] for item in observations if "spectral_flatness" in item]
        attacks = [item["attack_to_80_percent_peak_ms_estimate"] for item in observations if item.get("attack_to_80_percent_peak_ms_estimate") is not None]
        levels = [item["rms_dbfs"] for item in observations if "rms_dbfs" in item]
        row = {
                "bar": bar,
                "note_count": len(events),
                "pitch_range_midi": [min((e["pitch_midi"] for e in events), default=None), max((e["pitch_midi"] for e in events), default=None)],
                "median_note_duration_beats": round(float(np.median([e["timing_on_beat_grid"]["duration_in_beats_estimate"] for e in events])), 4) if events else None,
                "median_note_level_dbfs": round(float(np.median(levels)), 2) if levels else None,
                "median_spectral_centroid_hz": round(float(np.median(centroid)), 1) if centroid else None,
                "median_spectral_flatness": round(float(np.median(flatness)), 5) if flatness else None,
                "median_attack_ms_estimate": round(float(np.median(attacks)), 1) if attacks else None,
            }
        if audio is not None and sample_rate:
            start = (bar - 1) * 240.0 / tempo
            end = min(duration, bar * 240.0 / tempo)
            row["audio_observation"] = event_sound_observation(audio, sample_rate, start, end)
        profiles.append(row)
    return profiles


def _synth_patch_guess(stem, notes):
    import numpy as np

    observations = [event.get("sound_observation", {}) for event in notes]
    centroids = [item["spectral_centroid_hz"] for item in observations if "spectral_centroid_hz" in item]
    flatness = [item["spectral_flatness"] for item in observations if "spectral_flatness" in item]
    attacks = [item["attack_to_80_percent_peak_ms_estimate"] for item in observations if item.get("attack_to_80_percent_peak_ms_estimate") is not None]
    side_width = [item["stereo_side_to_mid_db"] for item in observations if "stereo_side_to_mid_db" in item]
    median_flatness = float(np.median(flatness)) if flatness else 0.1
    harmonicity = [item["harmonic_energy_ratio_at_pitch"] for item in observations if item.get("harmonic_energy_ratio_at_pitch") is not None]
    median_harmonicity = float(np.median(harmonicity)) if harmonicity else 0.0
    waveform = "noise" if median_flatness >= 0.42 else ("sine" if median_flatness <= 0.1 and median_harmonicity >= 0.72 else "saw")
    role = {"bass": "bass", "other": "lead", "vocals": "pad"}.get(stem, "texture")
    attack = float(np.median(attacks)) if attacks else 20.0
    centroid = float(np.median(centroids)) if centroids else 2500.0
    width_db = float(np.median(side_width)) if side_width else -60.0
    sound = {
        "waveform": waveform,
        "unison": 1,
        "cutoff_hz": round(max(250.0, min(18_000.0, centroid * 2.2)), 1),
        "attack_ms": round(max(1.0, min(1200.0, attack)), 1),
        "decay_ms": 180.0,
        "sustain": 0.68,
        "release_ms": 180.0,
        "stereo_spread": round(max(0.0, min(1.0, 10.0 ** (width_db / 20.0))), 3),
        "filter_mode": "lowpass",
    }
    if stem == "bass":
        partial_rows = [
            item.get("harmonic_energy_share_partial_1_to_8")
            for item in observations
            if item.get("harmonic_energy_share_partial_1_to_8")
        ]
        partials = np.median(np.asarray(partial_rows, dtype=float), axis=0) if partial_rows else np.zeros(8)
        fundamental_share = float(partials[0]) if len(partials) else 0.0
        upper_share = float(np.sum(partials[1:])) if len(partials) > 1 else 0.0
        sub_gain = round(max(0.08, min(0.65, 0.18 + 0.6 * fundamental_share)), 3)
        body_gain = round(max(0.35, min(1.25, 0.55 + 0.8 * upper_share)), 3)
        bass_layers = [
            {"waveform": "sine", "octave": -1, "gain": sub_gain, "noise_amount": 0.0},
            {"waveform": waveform if waveform != "noise" else "saw", "octave": 0, "gain": body_gain, "noise_amount": round(min(0.2, median_flatness * 0.2), 3)},
        ]
        if median_flatness >= 0.18:
            bass_layers.append({"waveform": "noise", "octave": 0, "gain": 0.16, "noise_amount": round(min(0.65, median_flatness), 3), "cutoff_hz": round(max(500.0, min(12_000.0, centroid * 2.8)), 1)})
        sound["layers"] = bass_layers
    return {
        "source_type": "synth",
        "role": role,
        "sound": sound,
        "estimate_method": "measured_note_envelope_spectral_harmonic_and_stereo_features_mapped_to_native_oscillator_layers",
        "estimate_status": "initial_synth_patch_guess_unvalidated",
    }


def build_synth_reconstruction_spec(audio, sample_rate, duration, tempo, stem_reports, drum_events, chord_bars):
    import numpy as np

    bar_count = int(math.ceil(duration * tempo / 240.0))
    parts = []
    for stem_report in stem_reports:
        stem = stem_report["stem"]
        notes = stem_report["note_events"]
        amplitudes = [max(0.0, event.get("amplitude_estimate", 0.0)) for event in notes]
        velocity_scale = float(np.percentile(amplitudes, 95)) if amplitudes else 1.0
        synth_events = []
        for event in notes:
            start = event["start_seconds"]
            end = event["end_seconds"]
            level = max(0.0, event.get("amplitude_estimate", 0.0))
            velocity = round(36 + 90 * min(1.0, level / max(velocity_scale, 1e-9)) ** 0.5)
            timing = event["timing_on_beat_grid"]
            controls = event.get("synth_overrides_estimate") if stem == "bass" else None
            note = _dragonoff_note(
                start,
                end,
                event["pitch_midi"],
                velocity,
                tempo,
                bar_count,
                synth_overrides=controls,
            )
            if note is not None:
                synth_events.append(note)
        part_id = {"bass": "synth_bass", "other": "synth_pitched_layer", "vocals": "synth_sustained_layer"}.get(stem, f"synth_{stem}")
        profiles = stem_report.get("bar_sound_profiles", [])
        bar_levels = [
            profile.get("audio_observation", {}).get("rms_dbfs", -120.0)
            for profile in profiles
        ]
        reference_level = max(bar_levels, default=-120.0)
        automation = []
        for profile in profiles:
            observed = profile.get("audio_observation", {})
            automation.extend(
                [
                    {
                        "parameter": "gain_db",
                        "bar": profile["bar"],
                        "beat": 1.0,
                        "value": round(max(-60.0, min(0.0, observed.get("rms_dbfs", -120.0) - reference_level)), 2),
                    },
                    {
                        "parameter": "lowpass_hz",
                        "bar": profile["bar"],
                        "beat": 1.0,
                        "value": round(max(250.0, min(18_000.0, observed.get("spectral_centroid_hz", 2_000.0) * 2.2)), 1),
                    },
                ]
            )
        parts.append(
            {
                "id": part_id,
                "name": {"bass": "Synth Bass", "other": "Synth Pitched Layer", "vocals": "Synth Sustained Layer"}.get(stem, f"Synth {stem.title()}"),
                "source_stem": stem,
                **_synth_patch_guess(stem, notes),
                "pattern": {"id": f"{part_id}_events", "bars": bar_count, "notes": synth_events},
                "source_note_event_count": len(notes),
                "source_note_event_path": f"note_transcriptions[stem={stem}].note_events",
                "bar_sound_profiles": profiles or _part_sound_profile(stem, notes, duration, tempo),
                "automation": automation,
            }
        )

    drum_groups = {"kick-like": [], "snare/percussion-like": [], "hat/cymbal-like": []}
    for event in drum_events:
        label = event.get("character_estimate", "snare/percussion-like")
        if label not in drum_groups:
            label = "snare/percussion-like"
        observation = event.get("sound_observation", {})
        pitch = observation.get("synth_base_pitch_midi_estimate") if label == "kick-like" else None
        pitch = pitch or {"kick-like": 36, "snare/percussion-like": 48, "hat/cymbal-like": 72}[label]
        start = event["time_seconds"]
        controls = observation.get("synth_control_estimate", {})
        # The event gate follows its measured decay; each hit also carries
        # per-note envelope, filter, noise, and pitch-sweep controls.
        hit_duration = max(0.035, min(0.65, (controls.get("decay_ms", 90.0) + controls.get("release_ms", 45.0)) / 1000.0))
        end = start + hit_duration
        velocity = round(42 + 78 * min(1.0, event.get("onset_strength_relative", 0.5)))
        note = _dragonoff_note(
            start,
            end,
            pitch,
            velocity,
            tempo,
            bar_count,
            metadata={
                "pitch_start_hz_estimate": observation.get("pitch_start_hz_estimate"),
                "pitch_late_hz_estimate": observation.get("pitch_late_hz_estimate"),
                "decay_to_20_percent_ms_estimate": observation.get("decay_to_20_percent_ms_estimate"),
                "band_class_margin_uncalibrated": observation.get("band_class_margin_uncalibrated"),
            },
            synth_overrides=controls,
        )
        if note is not None:
            drum_groups[label].append(note)
    for label, part_id, name, waveform, cutoff in (
        ("kick-like", "synth_kick", "Synth Kick", "sine", 180.0),
        ("snare/percussion-like", "synth_mid_percussion", "Synth Mid Percussion", "noise", 5_000.0),
        ("hat/cymbal-like", "synth_bright_percussion", "Synth Bright Percussion", "noise", 12_000.0),
    ):
        source_events = [
            event for event in drum_events
            if event.get("character_estimate") == label
        ]
        observations = [event.get("sound_observation", {}) for event in source_events]

        def median_control(name, fallback):
            values = [
                item.get("synth_control_estimate", {}).get(name)
                for item in observations
                if item.get("synth_control_estimate", {}).get(name) is not None
            ]
            return round(float(np.median(values)), 2) if values else fallback

        median_attack = median_control("attack_ms", 1.0)
        median_decay = median_control("decay_ms", 90.0)
        median_release = median_control("release_ms", 45.0)
        median_cutoff = median_control("cutoff_hz", cutoff)
        median_pitch_sweep = median_control("pitch_envelope_semitones", 18.0 if label == "kick-like" else 0.0)
        median_pitch_decay = median_control("pitch_decay_ms", 55.0)
        median_noise = median_control("noise_amount", 0.0 if label == "kick-like" else 0.6)
        drum_sound = {
            "waveform": waveform,
            "cutoff_hz": median_cutoff,
            "attack_ms": median_attack,
            "decay_ms": median_decay,
            "sustain": 0.0,
            "release_ms": median_release,
            "pitch_envelope_semitones": median_pitch_sweep,
            "pitch_decay_ms": median_pitch_decay,
            "layers": [
                {"waveform": waveform, "gain": 1.0, "noise_amount": median_noise, "cutoff_hz": median_cutoff}
            ],
        }
        parts.append(
            {
                "id": part_id,
                "name": name,
                "source_stem": "drums",
                "source_type": "synth",
                "role": "kick" if label == "kick-like" else "perc",
                "sound": drum_sound,
                "estimate_method": "per_hit_transient_envelope_spectral_and_low_frequency_pitch_trajectory_fit",
                "estimate_status": "measured_controls_with_unvalidated_synth_mapping",
                "pattern": {"id": f"{part_id}_events", "bars": bar_count, "notes": drum_groups[label]},
                "source_note_event_count": len(drum_groups[label]),
            }
        )

    chord_notes = []
    root_map = {name: index for index, name in enumerate(PITCH_NAMES)}
    for item in chord_bars:
        parts_of_name = item["chord_candidate"].split()
        root = root_map.get(parts_of_name[0], 0)
        quality = parts_of_name[1] if len(parts_of_name) > 1 else "minor"
        third = 4 if quality == "major" else 3
        pitches = [48 + root, 48 + root + third, 48 + root + 7]
        for pitch in pitches:
            chord_notes.append(
                {
                    "bar": item["bar"], "beat": 1.0, "tick": 0, "pitch": pitch,
                    "length_beats": 4.0, "velocity": 68, "channel": 1,
                    "source_chord_candidate": item["chord_candidate"],
                    "source_chord_confidence_uncalibrated": item["confidence"],
                }
            )
    parts.append(
        {
            "id": "synth_harmonic_bed", "name": "Synth Harmonic Bed", "source_stem": "estimated_bar_harmony",
            "source_type": "synth", "role": "chords",
            "sound": {"waveform": "triangle", "unison": 3, "detune_cents": 7.0, "cutoff_hz": 4_200.0, "attack_ms": 90.0, "decay_ms": 420.0, "sustain": 0.64, "release_ms": 480.0, "stereo_spread": 0.45},
        "estimate_method": "one_sustained_triad_voicing_per_bar_from_estimated_chord_labels",
            "estimate_status": "harmonic_placeholder_bar_level_only_not_a_transcribed_chord_rhythm",
            "pattern": {"id": "synth_harmonic_bed_events", "bars": bar_count, "notes": chord_notes},
            "source_note_event_count": len(chord_notes),
        }
    )

    mix_targets = []
    for bar in range(1, bar_count + 1):
        start = (bar - 1) * 240.0 / tempo
        end = min(duration, bar * 240.0 / tempo)
        features = event_sound_observation(audio, sample_rate, start, end)
        mix_targets.append({"bar": bar, "start_seconds": round(start, 4), "end_seconds": round(end, 4), **features})

    chord_roots = {}
    for item in chord_bars:
        root_name, _, quality = (item["chord_candidate"].partition(" "))
        entry = chord_roots.setdefault(root_name, {"major": 0, "minor": 0})
        entry[quality if quality in ("major", "minor") else "minor"] += 1
    key_name = max(chord_roots, key=lambda root: sum(chord_roots[root].values())) if chord_roots else "C"
    key_scale = max(chord_roots.get(key_name, {"minor": 1}), key=chord_roots.get(key_name, {"minor": 1}).get)

    section_bars = []
    for start_bar in range(1, bar_count + 1, 4):
        rows = mix_targets[start_bar - 1 : min(bar_count, start_bar + 3)]
        energies = [10.0 ** (row["rms_dbfs"] / 20.0) for row in rows]
        section_bars.append({"name": f"analysis_{start_bar:03d}", "start_bar": start_bar, "bars": len(rows), "energy": round(float(np.mean(energies)), 5)})
    max_section_energy = max((section["energy"] for section in section_bars), default=1.0) or 1.0
    for section in section_bars:
        section["energy"] = round(section["energy"] / max_section_energy, 5)

    return {
        "spec_schema_version": "2.0",
        "target_renderer": "dragonoff_native_synth",
        "all_parts_source_type": "synth",
        "instrument_identity_required": False,
        "event_synthesis_controls": {
            "consumer": "Dragonoff Note.synth_overrides; applied per rendered MIDI event",
            "supported_per_note_fields": [
                "waveform", "pitch_envelope_semitones", "pitch_decay_ms", "attack_ms",
                "decay_ms", "sustain", "release_ms", "cutoff_hz", "filter_env_amount", "noise_amount",
            ],
            "bass_note_path": "parts[id=synth_bass].pattern.notes[*].synth_overrides",
            "drum_hit_path": "parts[id=synth_kick|synth_mid_percussion|synth_bright_percussion].pattern.notes[*].synth_overrides",
            "fit_status": "measured_event_shape_mapped_to_supported_controls; heuristic_not_original_patch_recovery",
        },
        "tempo_bpm": tempo,
        "key": key_name,
        "scale": key_scale,
        "meter": "4/4_assumption",
        "bar_count": bar_count,
        "sections": section_bars,
        "parts": parts,
        "mix_targets_by_bar": mix_targets,
        "render_instructions": [
            "Create every part with source_type=synth; these roles describe audible function, not real instrument identity.",
            "Use each patch estimate as an initial Dragonoff SoundSettings object and preserve every explicit note event and its synth_overrides.",
            "Bass notes carry measured envelope, harmonic, stereo, and pitch-bend evidence; synth_overrides fit per-note envelope, filter, noise, and supported pitch-envelope controls.",
            "Drum hits carry transient, decay, band-energy, and low-frequency pitch-trajectory measurements; apply each hit's synth_overrides to vary the native synth per hit.",
            "Use note_transcriptions and bar_sound_profiles as the timing and timbre evidence; pattern notes are the Dragonoff beat-grid conversion.",
            "Match mix_targets_by_bar for relative level, spectral balance, and stereo width while rendering.",
            "The chord part is a one-chord-per-bar placeholder because this analysis does not yet transcribe intra-bar chord rhythm or voicing.",
        ],
        "confidence_and_limits": "Patch values are heuristic initial settings, not recovered original presets. Polyphonic overlap, source separation leakage, and the ambiguity of a mixed recording affect the estimates.",
    }


def estimate_chords(other_path, tempo, duration):
    import librosa
    import numpy as np

    audio, sample_rate = read_audio(other_path)
    hop = 2048
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate, hop_length=hop)
    bar_seconds = 4.0 * 60.0 / tempo
    bars = []
    count = int(np.ceil(duration / bar_seconds))
    for index in range(count):
        start = index * bar_seconds
        end = min(duration, start + bar_seconds)
        first = max(0, int(start * sample_rate / hop))
        last = min(chroma.shape[1], max(first + 1, int(end * sample_rate / hop)))
        profile = np.mean(chroma[:, first:last], axis=1)
        total = float(np.sum(profile)) or 1.0
        scores = []
        for root in range(12):
            for quality, third in (("major", 4), ("minor", 3)):
                score = float((profile[root] + profile[(root + third) % 12] + profile[(root + 7) % 12]) / total)
                scores.append((score, root, quality))
        scores.sort(reverse=True)
        best = scores[0]
        confidence = (best[0] - scores[1][0]) / max(best[0], 1e-12)
        candidates = [
            {
                "chord": f"{PITCH_NAMES[root]} {quality}",
                "triad_energy_share": round(float(score), 4),
                "rank_margin": round(float((score - scores[index + 1][0]) / max(score, 1e-12)), 4)
                if index + 1 < len(scores)
                else 0.0,
            }
            for index, (score, root, quality) in enumerate(scores[:3])
        ]
        bars.append(
            {
                "bar": index + 1,
                "start_seconds": round(start, 4),
                "end_seconds": round(end, 4),
                "chord_candidate": f"{PITCH_NAMES[best[1]]} {best[2]}",
                "confidence": round(float(max(0.0, min(1.0, confidence))), 4),
                "confidence_method": "uncalibrated top-versus-runner-up triad-energy margin",
                "interpretation": "ambiguous" if confidence < 0.12 else "candidate",
                "top_candidates": candidates,
                "strongest_pitch_classes": [PITCH_NAMES[i] for i in np.argsort(profile)[-4:][::-1]],
                "pitch_class_energy_share": {
                    PITCH_NAMES[i]: round(float(profile[i] / total), 4)
                    for i in np.argsort(profile)[-6:][::-1]
                },
            }
        )
    return bars


def merge_chord_bars(bars):
    merged = []
    for bar in bars:
        chord = bar["chord_candidate"]
        if merged and merged[-1]["chord"] == chord:
            item = merged[-1]
            n = item["end_bar"] - item["start_bar"] + 1
            item["confidence"] = round((item["confidence"] * n + bar["confidence"]) / (n + 1), 4)
            item["end_bar"] = bar["bar"]
            item["end_seconds"] = bar["end_seconds"]
        else:
            merged.append(
                {
                    "chord": chord,
                    "start_bar": bar["bar"],
                    "end_bar": bar["bar"],
                    "start_seconds": bar["start_seconds"],
                    "end_seconds": bar["end_seconds"],
                    "confidence": bar["confidence"],
                }
            )
    return merged


def build_bar_event_map(chord_bars, drum_events, stem_reports, duration, tempo):
    bar_seconds = 4.0 * 60.0 / tempo
    event_bars = [
        event["timing_on_beat_grid"]["bar_one_based"]
        for stem in stem_reports
        for event in stem["note_events"]
    ]
    event_bars.extend(event["bar_one_based"] for event in drum_events)
    count = max(int(math.ceil(duration / bar_seconds)), len(chord_bars), max(event_bars, default=0))
    chord_by_bar = {item["bar"]: item for item in chord_bars}
    drums_by_bar = {}
    for event in drum_events:
        drums_by_bar.setdefault(event["bar_one_based"], []).append(event)
    notes_by_stem_and_bar = {}
    for stem in stem_reports:
        for event in stem["note_events"]:
            bar = event["timing_on_beat_grid"]["bar_one_based"]
            notes_by_stem_and_bar.setdefault((stem["stem"], bar), []).append(event)

    output = []
    for bar_number in range(1, count + 1):
        chord = chord_by_bar.get(bar_number)
        layer_counts = {
            stem["stem"]: len(notes_by_stem_and_bar.get((stem["stem"], bar_number), []))
            for stem in stem_reports
        }
        layer_counts["drums"] = len(drums_by_bar.get(bar_number, []))
        pitch_ranges = {}
        for stem in stem_reports:
            events = notes_by_stem_and_bar.get((stem["stem"], bar_number), [])
            pitches = [event["pitch_midi"] for event in events]
            if pitches:
                pitch_ranges[stem["stem"]] = {"lowest_midi": min(pitches), "highest_midi": max(pitches)}
        output.append(
            {
                "bar": bar_number,
                "start_seconds_estimate": round((bar_number - 1) * bar_seconds, 4),
                "end_seconds_estimate": round(min(duration, bar_number * bar_seconds), 4),
                "chord_candidate": chord["chord_candidate"] if chord else None,
                "chord_interpretation": chord["interpretation"] if chord else "unavailable",
                "chord_top_candidates": chord["top_candidates"] if chord else [],
                "note_event_count_by_layer": layer_counts,
                "pitch_range_by_layer_midi": pitch_ranges,
            }
        )
    return output


def run(args):
    # Imports are delayed so missing optional packages produce a direct error.
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, predict, save_note_events

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    audio_path = Path(args.audio).resolve()
    model_name = args.demucs_model
    demucs_root = output / "stems"
    expected_dir = demucs_root / model_name / audio_path.stem
    required_stems = ["drums", "bass", "other", "vocals"]
    if not all((expected_dir / f"{stem}.wav").is_file() for stem in required_stems):
        command = [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n",
            model_name,
            "--shifts",
            str(args.shifts),
            "-o",
            str(demucs_root),
            str(audio_path),
        ]
        subprocess.run(command, check=True, stdout=sys.stderr)

    audio, sample_rate = read_audio(audio_path)
    stereo_audio, stereo_sample_rate = read_audio(audio_path, mono=False)
    if stereo_sample_rate != sample_rate:
        raise ValueError("mono and stereo audio decoders returned inconsistent sample rates")
    duration = len(audio) / sample_rate
    drum_events, beat_times = estimate_drum_events(expected_dir / "drums.wav", args.tempo)
    for item in drum_events:
        item["timing_on_beat_grid"] = {
            "bar_one_based": item["bar_one_based"],
            "beat_in_bar_one_based": item["beat_in_bar_one_based"],
            "sixteenth_position_estimate": item["sixteenth_position"],
        }
    stem_reports = []
    midi_dir = output / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    model = Model(ICASSP_2022_MODEL_PATH)
    for stem in ("bass", "other", "vocals"):
        wav_path = expected_dir / f"{stem}.wav"
        stem_audio, stem_sample_rate = read_audio(wav_path, mono=False)
        # Basic Pitch prints CoreML input diagnostics to stdout; keep the tool's
        # stdout reserved for its JSON result.
        with contextlib.redirect_stdout(sys.stderr):
            _raw, midi, events = predict(wav_path, model)
        midi_path = midi_dir / f"{stem}.mid"
        csv_path = midi_dir / f"{stem}.csv"
        midi.write(str(midi_path))
        save_note_events(events, csv_path)
        notes = note_json(events, beat_times, args.tempo, stem_audio, stem_sample_rate, stem_role=stem)
        bar_sound_profiles = _part_sound_profile(stem, notes, duration, args.tempo, stem_audio, stem_sample_rate)
        stem_reports.append(
            {
                "stem": stem,
                "audio_path": str(wav_path),
                "midi_path": str(midi_path),
                "note_events_csv": str(csv_path),
                "note_event_count": len(notes),
                "note_event_order": "ascending start_seconds, then pitch_midi",
                "note_events": notes,
                "bar_sound_profiles": bar_sound_profiles,
            }
        )

    chord_bars = estimate_chords(expected_dir / "other.wav", args.tempo, duration)
    bar_event_map = build_bar_event_map(chord_bars, drum_events, stem_reports, duration, args.tempo)
    synth_reconstruction_spec = build_synth_reconstruction_spec(
        stereo_audio,
        sample_rate,
        duration,
        args.tempo,
        stem_reports,
        drum_events,
        chord_bars,
    )
    report = {
        "analysis_schema_version": "4.0",
        "input_audio": str(audio_path),
        "duration_seconds": round(duration, 4),
        "models": {
            "source_separation": model_name,
            "note_transcription": "Spotify Basic Pitch ICASSP 2022 pretrained model",
            "bass_synth_control_estimation": "per-note envelope and harmonic observations with Basic Pitch bend-curve fit",
            "drum_synth_control_estimation": "per-hit transient decay and low-frequency pitch-trajectory fit",
            "synth_patch_estimation": "measured event profiles mapped heuristically to supported Dragonoff synth controls",
            "training_performed": False,
        },
        "limitations": "Source separation can leak sounds between stems. Note events and sound-shape measurements are model estimates and are not manually corrected; amplitude_estimate is not calibrated confidence or MIDI velocity. Bass per-note synth controls and drum hit roles, pitch sweeps, and envelopes are evidence-based heuristic fits, not recovered original settings. Drum role scores derive from frequency-band energy and are uncalibrated. Chord confidence is an uncalibrated score margin, not a probability. Chord labels are estimates from the separated accompaniment; the synth harmonic bed uses one sustained triad per bar and does not recover chord rhythm or voicing.",
        "stems": [
            {"stem": name, "audio_path": str(expected_dir / f"{name}.wav")}
            for name in required_stems
        ],
        "beat_grid": {
            "tempo_bpm_hint": args.tempo,
            "beat_times_seconds_from_drum_stem": beat_times,
            "meter_assumption": "4/4",
        },
        "chords": {
            "per_bar": chord_bars,
            "progression": merge_chord_bars(chord_bars),
        },
        "bar_event_map": bar_event_map,
        "synth_reconstruction_spec": synth_reconstruction_spec,
        "reconstruction_spec": {
            "intent": "synth-first event-level reconstruction specification from an analyzed mix",
            "consumer_guidance": [
                "Treat synth_reconstruction_spec as the concrete project-building specification, not tonal guidance.",
                "Render all parts with Dragonoff's native synth source; real instrument classification is not required.",
                "Use the timed event arrays and bar_event_map as the primary arrangement and performance reference; do not reduce this data to key/chord guidance alone.",
                "Use separated stems as listening references for each layer and MIDI/CSV files as editable candidate transcriptions.",
                "Seconds are the authoritative event positions. Beat/bar assignments and chord labels are estimates; preserve their uncertainty during reconstruction.",
                "Do not interpret estimated MIDI events as recovered synth patches, automation, effects, or original session settings.",
            ],
            "layer_event_paths": {
                "drums": "drum_events",
                "bass": "note_transcriptions[stem=bass].note_events",
                "other": "note_transcriptions[stem=other].note_events",
                "vocals": "note_transcriptions[stem=vocals].note_events",
                "synth_parts": "synth_reconstruction_spec.parts",
                "chords_as_synth_events": "synth_reconstruction_spec.parts[id=synth_harmonic_bed].pattern.notes",
            },
            "synth_reconstruction_spec_path": "synth_reconstruction_spec",
            "timing": {
                "event_time_unit": "seconds",
                "beat_grid_source": "librosa beat tracker on separated drums stem, guided by tempo estimate",
                "tempo_bpm_hint": args.tempo,
                "meter_assumption": "4/4",
                "sixteenth_positions_are_estimates": True,
            },
        },
        "drum_events": drum_events,
        "note_transcriptions": stem_reports,
        "source_separation_note": "The Demucs 'other' stem may still contain several sounds. The synth reconstruction spec preserves its measured sound behavior and timed candidate notes without requiring real instrument classification.",
    }
    report_path = output / "track_analysis_ml.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, separators=(",", ":")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tempo", type=float, required=True)
    parser.add_argument("--demucs-model", default="htdemucs")
    parser.add_argument("--shifts", type=int, default=1)
    args = parser.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        run(args)
    except Exception as error:
        print(f"track analysis ML pipeline failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
