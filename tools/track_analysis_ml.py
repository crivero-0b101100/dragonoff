#!/usr/bin/env python3
"""Optional pretrained source-separation and note-transcription pass.

Requires the optional packages listed in requirements-track-analysis.txt.
The script writes separated stems and MIDI/CSV transcriptions under --output,
then prints a JSON analysis report to stdout.
"""

import argparse
import contextlib
import json
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
    beat_offset = 0.0
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
        if len(beat_times):
            beat_offset = float(beat_times[0])
    except Exception:
        beat_times = np.array([], dtype=float)
    beat_period = 60.0 / tempo_hint
    result = []
    for time_seconds in onset_times:
        label, band_share = classify_drum_event(audio, sample_rate, float(time_seconds))
        beat = (float(time_seconds) - beat_offset) / beat_period
        nearest = max(0, int(round(beat)))
        fraction = (beat - round(beat)) % 1.0
        subdivision = min(range(4), key=lambda i: abs(fraction - i / 4.0))
        result.append(
            {
                "time_seconds": round(float(time_seconds), 4),
                "nearest_beat_zero_based": nearest,
                "bar_one_based": nearest // 4 + 1,
                "beat_in_bar_one_based": nearest % 4 + 1,
                "sixteenth_position": subdivision * 4 + 1,
                "character_estimate": label,
                "band_energy_share_low_mid_high": [round(x, 5) for x in band_share],
            }
        )
    return result, [round(float(value), 4) for value in beat_times]


def note_json(events):
    import pretty_midi

    output = []
    for start, end, pitch, amplitude, _pitch_bend in events:
        output.append(
            {
                "start_seconds": round(float(start), 4),
                "end_seconds": round(float(end), 4),
                "pitch_midi": int(pitch),
                "pitch_name": pretty_midi.note_number_to_name(int(pitch)),
                "amplitude_estimate": round(float(amplitude), 4),
            }
        )
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
        bars.append(
            {
                "bar": index + 1,
                "start_seconds": round(start, 4),
                "end_seconds": round(end, 4),
                "chord_candidate": f"{PITCH_NAMES[best[1]]} {best[2]}",
                "confidence": round(float(max(0.0, min(1.0, confidence))), 4),
                "strongest_pitch_classes": [PITCH_NAMES[i] for i in np.argsort(profile)[-4:][::-1]],
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
    stem_reports = []
    midi_dir = output / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    note_events_by_stem = {}
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
        notes = note_json(events)
        note_events_by_stem[stem] = notes
        stem_reports.append(
            {
                "stem": stem,
                "audio_path": str(wav_path),
                "midi_path": str(midi_path),
                "note_events_csv": str(csv_path),
                "note_event_count": len(notes),
                "note_events": notes,
            }
        )

    drum_events, beat_times = estimate_drum_events(expected_dir / "drums.wav", args.tempo)
    chord_bars = estimate_chords(expected_dir / "other.wav", args.tempo, duration)
    report = {
        "input_audio": str(audio_path),
        "duration_seconds": round(duration, 4),
        "models": {
            "source_separation": model_name,
            "note_transcription": "Spotify Basic Pitch ICASSP 2022 pretrained model",
            "training_performed": False,
        },
        "limitations": "Source separation can leak sounds between stems. Basic Pitch note events are model estimates and are not manually corrected. Drum labels are frequency-based guesses. Chord labels are estimates from the separated accompaniment.",
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
