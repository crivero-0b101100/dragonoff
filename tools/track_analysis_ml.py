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


def read_audio(path):
    import librosa
    import numpy as np

    audio, sample_rate = librosa.load(path, sr=None, mono=True)
    if len(audio) == 0:
        raise ValueError(f"empty audio file: {path}")
    return audio.astype(np.float32, copy=False), sample_rate


def classify_drum_event(audio, sample_rate, time_seconds):
    import numpy as np

    start = max(0, int((time_seconds - 0.035) * sample_rate))
    end = min(len(audio), int((time_seconds + 0.065) * sample_rate))
    segment = audio[start:end]
    if len(segment) < 16:
        return "percussive"
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
    result = []
    for time_seconds in onset_times:
        label, band_share = classify_drum_event(audio, sample_rate, float(time_seconds))
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
                "band_energy_share_low_mid_high": [round(x, 5) for x in band_share],
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
        "bar_one_based": nearest // 4 + 1,
        "beat_in_bar_one_based": nearest % 4 + 1,
        "nearest_beat_zero_based": nearest,
        "offset_from_nearest_beat_seconds": round(start - beat_time, 4),
        "duration_in_beats_estimate": round((end - start) / beat_period, 3),
        "sixteenth_position_estimate": (int(round((beat_float - math.floor(beat_float)) * 4)) % 4) * 4 + 1,
    }


def note_json(events, beat_times, tempo):
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
        output.append(item)
    return output


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
        # Basic Pitch prints CoreML input diagnostics to stdout; keep the tool's
        # stdout reserved for its JSON result.
        with contextlib.redirect_stdout(sys.stderr):
            _raw, midi, events = predict(wav_path, model)
        midi_path = midi_dir / f"{stem}.mid"
        csv_path = midi_dir / f"{stem}.csv"
        midi.write(str(midi_path))
        save_note_events(events, csv_path)
        notes = note_json(events, beat_times, args.tempo)
        stem_reports.append(
            {
                "stem": stem,
                "audio_path": str(wav_path),
                "midi_path": str(midi_path),
                "note_events_csv": str(csv_path),
                "note_event_count": len(notes),
                "note_event_order": "ascending start_seconds, then pitch_midi",
                "note_events": notes,
            }
        )

    chord_bars = estimate_chords(expected_dir / "other.wav", args.tempo, duration)
    bar_event_map = build_bar_event_map(chord_bars, drum_events, stem_reports, duration, args.tempo)
    report = {
        "analysis_schema_version": "2.0",
        "input_audio": str(audio_path),
        "duration_seconds": round(duration, 4),
        "models": {
            "source_separation": model_name,
            "note_transcription": "Spotify Basic Pitch ICASSP 2022 pretrained model",
            "training_performed": False,
        },
        "limitations": "Source separation can leak sounds between stems. Basic Pitch note events are model estimates and are not manually corrected; amplitude_estimate is not calibrated confidence or MIDI velocity. Drum labels are frequency-based guesses. Chord confidence is an uncalibrated score margin, not a probability. Chord labels are estimates from the separated accompaniment.",
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
        "reconstruction_spec": {
            "intent": "event-level reconstruction reference from an analyzed mix",
            "consumer_guidance": [
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
            },
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
        "source_separation_note": "The Demucs 'other' stem may still contain several instruments; note transcription is therefore a candidate event list, not a definitive score.",
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
