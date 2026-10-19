// Copyright (C) 2026 OpenAI
// This program is free software under the GNU GPL v2 or (at your option) any later version.

//! Lightweight, dependency-free analysis of a finished stereo or mono WAV mix.
//!
//! These estimates describe audible properties of the mix. Optional ML output
//! can propose synth settings from the observed sound, but cannot identify a
//! unique original multitrack session or patch.

use serde::Serialize;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Serialize)]
pub struct TrackAnalysis {
    pub source: String,
    pub duration_seconds: f64,
    pub sample_rate: u32,
    pub tempo_bpm: f64,
    pub tempo_bpm_detected: f64,
    pub tempo_confidence: f64,
    pub meter: String,
    pub beat_interval_seconds: f64,
    pub key: String,
    pub key_confidence: f64,
    pub spectral_balance: SpectralBalance,
    pub stereo: StereoCharacter,
    pub sections: Vec<EnergySection>,
    pub bars: Vec<BarAnalysis>,
    pub chord_progression: Vec<ChordSegment>,
    pub transient_events: Vec<TransientEvent>,
    pub sound_character: SoundCharacter,
    pub limitations: &'static str,
}

#[derive(Debug, Serialize)]
pub struct SpectralBalance {
    pub low_below_180_hz: f64,
    pub mid_180_to_4000_hz: f64,
    pub high_above_4000_hz: f64,
}

#[derive(Debug, Serialize)]
pub struct StereoCharacter {
    pub rms_dbfs: f64,
    pub peak_dbfs: f64,
    pub channel_correlation: f64,
    pub side_to_mid_db: f64,
}

#[derive(Debug, Serialize)]
pub struct ReconstructionComparison {
    pub reference: String,
    pub reconstruction: String,
    pub duration_difference_seconds: f64,
    pub tempo_difference_bpm: f64,
    pub detected_reference_tempo_bpm: f64,
    pub detected_reconstruction_tempo_bpm: f64,
    pub comparison_grid_tempo_bpm: f64,
    pub comparison_grid_source: String,
    pub reconstruction_alignment_offset_seconds: f64,
    pub alignment_confidence: f64,
    pub alignment_method: String,
    pub key_matches: bool,
    pub reference_key_confidence: f64,
    pub reconstruction_key_confidence: f64,
    pub chord_match_fraction: f64,
    pub chord_match_fraction_method: String,
    pub confident_chord_match_fraction: f64,
    pub chord_confidence_gate_method: String,
    pub harmony_chroma_similarity: f64,
    pub bar_energy_correlation: f64,
    pub bar_rms_mae_db: f64,
    pub rms_difference_db: f64,
    pub peak_difference_db: f64,
    pub spectral_balance_difference: SpectralBalance,
    pub stereo_correlation_difference: f64,
    pub side_to_mid_difference_db: f64,
    pub per_bar: Vec<BarComparison>,
    pub recommended_adjustments: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct BarComparison {
    pub bar: usize,
    pub reference_energy: f64,
    pub reconstruction_energy: f64,
    pub reference_rms_dbfs: f64,
    pub reconstruction_rms_dbfs: f64,
    pub reference_chord: String,
    pub reconstruction_chord: String,
    pub chord_matches: bool,
    pub reference_chord_confidence: f64,
    pub reconstruction_chord_confidence: f64,
    pub reference_top_chord_candidates: Vec<ChordCandidate>,
    pub reconstruction_top_chord_candidates: Vec<ChordCandidate>,
    pub harmony_chroma_similarity: f64,
}

#[derive(Debug, Serialize)]
pub struct EnergySection {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub rms_dbfs: f64,
    pub relative_energy: f64,
}

#[derive(Debug, Serialize)]
pub struct BarAnalysis {
    pub bar: usize,
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub chord: String,
    pub chord_confidence: f64,
    pub chord_confidence_method: String,
    pub top_chord_candidates: Vec<ChordCandidate>,
    pub strongest_pitch_classes: Vec<String>,
    pub pitch_class_profile: [f64; 12],
    pub relative_energy: f64,
    pub rms_dbfs: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChordCandidate {
    pub chord: String,
    pub template_score: f64,
}

#[derive(Debug, Serialize)]
pub struct ChordSegment {
    pub chord: String,
    pub start_bar: usize,
    pub end_bar: usize,
    pub confidence: f64,
}

#[derive(Debug, Serialize)]
pub struct TransientEvent {
    pub time_seconds: f64,
    pub nearest_beat: usize,
    pub beat_subdivision: String,
    pub character: String,
    pub strength: f64,
}

#[derive(Debug, Serialize)]
pub struct SoundCharacter {
    pub description: String,
    pub notes: Vec<&'static str>,
}

struct ChromaFrame {
    time_seconds: f64,
    values: [f64; 12],
    energy: f64,
}

struct Wav {
    samples: Vec<f64>,
    channels: usize,
    sample_rate: u32,
}

pub fn analyze_wav(path: &Path) -> Result<TrackAnalysis, String> {
    analyze_wav_on_grid(path, None, 0.0)
}

fn analyze_wav_on_grid(
    path: &Path,
    grid_tempo_bpm: Option<f64>,
    timeline_offset_seconds: f64,
) -> Result<TrackAnalysis, String> {
    let wav = read_wav(path)?;
    let frames = wav.samples.len() / wav.channels;
    let duration = frames as f64 / wav.sample_rate as f64;
    if duration < 0.5 {
        return Err("track analysis requires at least 0.5 seconds of audio".into());
    }
    let mono: Vec<f64> = wav
        .samples
        .chunks_exact(wav.channels)
        .map(|frame| frame.iter().sum::<f64>() / wav.channels as f64)
        .collect();

    let (detected_tempo_bpm, tempo_confidence) = estimate_tempo(&mono, wav.sample_rate);
    let tempo_bpm = grid_tempo_bpm.unwrap_or(detected_tempo_bpm);
    let beat_interval_seconds = 60.0 / tempo_bpm;
    let section_len = (wav.sample_rate as f64 * beat_interval_seconds * 16.0).round() as usize;
    let mut sections = Vec::new();
    let mut section_energy = Vec::new();
    for (index, chunk) in mono.chunks(section_len).enumerate() {
        let mean_square = chunk.iter().map(|x| x * x).sum::<f64>() / chunk.len() as f64;
        section_energy.push(mean_square);
        sections.push(EnergySection {
            start_seconds: index as f64 * section_len as f64 / wav.sample_rate as f64,
            end_seconds: ((index + 1) * section_len).min(frames) as f64 / wav.sample_rate as f64,
            rms_dbfs: 10.0 * mean_square.max(1e-20).log10(),
            relative_energy: 0.0,
        });
    }
    let max_energy = section_energy
        .iter()
        .copied()
        .fold(0.0, f64::max)
        .max(1e-20);
    for (section, energy) in sections.iter_mut().zip(section_energy) {
        section.relative_energy = (energy / max_energy).sqrt().clamp(0.0, 1.0);
    }

    let (chroma, band_energy, chroma_frames) = spectral_features(&mono, wav.sample_rate);
    let (key, key_confidence) = estimate_key(&chroma);
    let total = band_energy.iter().sum::<f64>().max(1e-20);
    let bars = analyze_bars(
        &chroma_frames,
        duration,
        beat_interval_seconds,
        timeline_offset_seconds,
        &mono,
        wav.sample_rate,
    );
    let chord_progression = merge_chords(&bars);
    let transient_events = detect_transients(&mono, wav.sample_rate, beat_interval_seconds);
    let sound_character = characterize_sound(&band_energy, total, &sections);
    let stereo = stereo_character(&wav);

    Ok(TrackAnalysis {
        source: path.display().to_string(),
        duration_seconds: duration,
        sample_rate: wav.sample_rate,
        tempo_bpm,
        tempo_bpm_detected: detected_tempo_bpm,
        tempo_confidence,
        meter: "4/4 assumed".into(),
        beat_interval_seconds,
        key,
        key_confidence,
        spectral_balance: SpectralBalance {
            low_below_180_hz: band_energy[0] / total,
            mid_180_to_4000_hz: band_energy[1] / total,
            high_above_4000_hz: band_energy[2] / total,
        },
        stereo,
        sections,
        bars,
        chord_progression,
        transient_events,
        sound_character,
        limitations: "Bars assume a 4/4 meter and begin on the file start; tempo, chords, key and transient labels are estimates from the mixed audio. Transients are not confirmed drum hits. The built-in pass does not separate stems or transcribe notes. The optional ML pass can provide candidate stems, note events, sound observations and heuristic synth settings, but cannot uniquely recover original instruments, patches, or effects.",
    })
}

pub fn compare_wavs_with_grid(
    reference_path: &Path,
    reconstruction_path: &Path,
    expected_tempo_bpm: Option<f64>,
) -> Result<ReconstructionComparison, String> {
    let reference_detected = analyze_wav(reference_path)?;
    let reconstruction_detected = analyze_wav(reconstruction_path)?;
    let (grid_tempo, grid_source) =
        if let Some(tempo) = expected_tempo_bpm.filter(|value| (40.0..=240.0).contains(value)) {
            (tempo, "analysis_spec")
        } else {
            let candidates = [
                reference_detected.tempo_bpm * 0.5,
                reference_detected.tempo_bpm,
                reference_detected.tempo_bpm * 2.0,
            ];
            let chosen = candidates
                .into_iter()
                .filter(|value| (40.0..=240.0).contains(value))
                .min_by(|left, right| {
                    (left / reconstruction_detected.tempo_bpm)
                        .ln()
                        .abs()
                        .total_cmp(&(right / reconstruction_detected.tempo_bpm).ln().abs())
                })
                .unwrap_or(reference_detected.tempo_bpm);
            (chosen, "nearest_half_double_tempo_candidate")
        };
    let reference_wav = read_wav(reference_path)?;
    let reconstruction_wav = read_wav(reconstruction_path)?;
    let (alignment_offset, alignment_confidence) =
        estimate_envelope_alignment(&reference_wav, &reconstruction_wav);
    let reference = analyze_wav_on_grid(reference_path, Some(grid_tempo), 0.0)?;
    let reconstruction =
        analyze_wav_on_grid(reconstruction_path, Some(grid_tempo), alignment_offset)?;
    let bar_count = reference.bars.len().min(reconstruction.bars.len());
    let per_bar: Vec<BarComparison> = reference
        .bars
        .iter()
        .zip(&reconstruction.bars)
        .take(bar_count)
        .map(|(left, right)| BarComparison {
            bar: left.bar,
            reference_energy: left.relative_energy,
            reconstruction_energy: right.relative_energy,
            reference_rms_dbfs: left.rms_dbfs,
            reconstruction_rms_dbfs: right.rms_dbfs,
            reference_chord: left.chord.clone(),
            reconstruction_chord: right.chord.clone(),
            chord_matches: left.chord == right.chord,
            reference_chord_confidence: left.chord_confidence,
            reconstruction_chord_confidence: right.chord_confidence,
            reference_top_chord_candidates: left.top_chord_candidates.clone(),
            reconstruction_top_chord_candidates: right.top_chord_candidates.clone(),
            harmony_chroma_similarity: cosine_similarity(
                &left.pitch_class_profile,
                &right.pitch_class_profile,
            ),
        })
        .collect();
    let chord_match_fraction = if bar_count == 0 {
        0.0
    } else {
        per_bar.iter().filter(|bar| bar.chord_matches).count() as f64 / bar_count as f64
    };
    let confident_bar_count = per_bar
        .iter()
        .filter(|bar| {
            bar.reference_chord_confidence >= 0.05 && bar.reconstruction_chord_confidence >= 0.05
        })
        .count();
    let confident_chord_match_fraction = if confident_bar_count == 0 {
        0.0
    } else {
        per_bar
            .iter()
            .filter(|bar| {
                bar.reference_chord_confidence >= 0.05
                    && bar.reconstruction_chord_confidence >= 0.05
                    && bar.chord_matches
            })
            .count() as f64
            / confident_bar_count as f64
    };
    let harmony_chroma_similarity = if bar_count == 0 {
        0.0
    } else {
        per_bar
            .iter()
            .map(|bar| bar.harmony_chroma_similarity)
            .sum::<f64>()
            / bar_count as f64
    };
    let bar_rms_mae_db = if bar_count == 0 {
        0.0
    } else {
        per_bar
            .iter()
            .map(|bar| (bar.reference_rms_dbfs - bar.reconstruction_rms_dbfs).abs())
            .sum::<f64>()
            / bar_count as f64
    };
    let bar_energy_correlation = pearson(
        &per_bar
            .iter()
            .map(|bar| bar.reference_energy)
            .collect::<Vec<_>>(),
        &per_bar
            .iter()
            .map(|bar| bar.reconstruction_energy)
            .collect::<Vec<_>>(),
    );
    let rms_difference_db = reconstruction.stereo.rms_dbfs - reference.stereo.rms_dbfs;
    let peak_difference_db = reconstruction.stereo.peak_dbfs - reference.stereo.peak_dbfs;
    let low_difference = reconstruction.spectral_balance.low_below_180_hz
        - reference.spectral_balance.low_below_180_hz;
    let mid_difference = reconstruction.spectral_balance.mid_180_to_4000_hz
        - reference.spectral_balance.mid_180_to_4000_hz;
    let high_difference = reconstruction.spectral_balance.high_above_4000_hz
        - reference.spectral_balance.high_above_4000_hz;
    let stereo_correlation_difference =
        reconstruction.stereo.channel_correlation - reference.stereo.channel_correlation;
    let side_to_mid_difference_db =
        reconstruction.stereo.side_to_mid_db - reference.stereo.side_to_mid_db;
    let mut recommended_adjustments = Vec::new();
    let duration_difference_seconds = reconstruction.duration_seconds - reference.duration_seconds;
    if duration_difference_seconds.abs() > 0.05 {
        recommended_adjustments.push(format!(
            "Trim or extend the final render by {:.3} seconds to match the source duration.",
            -duration_difference_seconds
        ));
    }
    if rms_difference_db.abs() > 1.0 {
        recommended_adjustments.push(format!(
            "Adjust overall track gain by approximately {:.1} dB.",
            -rms_difference_db
        ));
    }
    if side_to_mid_difference_db.abs() > 3.0 {
        recommended_adjustments.push(format!(
            "Adjust stereo width; the reconstruction side-to-mid level differs by {:.1} dB.",
            side_to_mid_difference_db
        ));
    }
    if low_difference.abs() > 0.08 {
        recommended_adjustments.push(format!("Rebalance low-band energy; the reconstruction differs by {:.3} of total spectral energy.", low_difference));
    }
    if mid_difference.abs() > 0.08 {
        recommended_adjustments.push(format!("Rebalance mid-band energy; the reconstruction differs by {:.3} of total spectral energy.", mid_difference));
    }
    if high_difference.abs() > 0.04 {
        recommended_adjustments.push(format!("Rebalance high-band energy; the reconstruction differs by {:.3} of total spectral energy.", high_difference));
    }
    if expected_tempo_bpm.is_none()
        && (reconstruction_detected.tempo_bpm - reference_detected.tempo_bpm).abs() > 0.5
    {
        recommended_adjustments.push(format!(
            "Tempo detectors disagree ({:.2} vs {:.2} BPM); verify half/double-time and beat phase before changing project tempo.",
            reference_detected.tempo_bpm, reconstruction_detected.tempo_bpm
        ));
    }
    if harmony_chroma_similarity < 0.8 && confident_bar_count > 0 {
        recommended_adjustments.push(
            "Review aligned harmonic regions with sufficient chord evidence; low-confidence chord labels are not ground truth."
                .into(),
        );
    }
    if bar_energy_correlation < 0.8 {
        recommended_adjustments.push("Adjust per-section and per-bar layer levels; the energy contour differs from the reference.".into());
    }
    if bar_rms_mae_db > 3.0 {
        recommended_adjustments.push(format!(
            "Per-bar absolute RMS differs by {:.1} dB on average; compare level targets separately from the normalized energy shape.",
            bar_rms_mae_db
        ));
    }
    if alignment_confidence < 0.1 {
        recommended_adjustments.push(
            "Timeline alignment is ambiguous; inspect the offset before interpreting bar-level comparisons."
                .into(),
        );
    }
    Ok(ReconstructionComparison {
        reference: reference.source,
        reconstruction: reconstruction.source,
        duration_difference_seconds,
        tempo_difference_bpm: reconstruction_detected.tempo_bpm - reference_detected.tempo_bpm,
        detected_reference_tempo_bpm: reference_detected.tempo_bpm,
        detected_reconstruction_tempo_bpm: reconstruction_detected.tempo_bpm,
        comparison_grid_tempo_bpm: grid_tempo,
        comparison_grid_source: grid_source.into(),
        reconstruction_alignment_offset_seconds: alignment_offset,
        alignment_confidence,
        alignment_method: "20 Hz log-RMS envelope cross-correlation; +/-4 seconds; confidence is peak separation from non-neighboring offsets".into(),
        key_matches: reference.key == reconstruction.key,
        reference_key_confidence: reference.key_confidence,
        reconstruction_key_confidence: reconstruction.key_confidence,
        chord_match_fraction,
        chord_match_fraction_method: "exact chord-label string match across aligned bars; diagnostic only".into(),
        confident_chord_match_fraction,
        chord_confidence_gate_method: "uncalibrated heuristic gate: candidate margin >= 0.05; use chroma similarity as the primary harmony metric".into(),
        harmony_chroma_similarity,
        bar_energy_correlation,
        bar_rms_mae_db,
        rms_difference_db,
        peak_difference_db,
        spectral_balance_difference: SpectralBalance {
            low_below_180_hz: low_difference,
            mid_180_to_4000_hz: mid_difference,
            high_above_4000_hz: high_difference,
        },
        stereo_correlation_difference,
        side_to_mid_difference_db,
        per_bar,
        recommended_adjustments,
    })
}

fn cosine_similarity(left: &[f64; 12], right: &[f64; 12]) -> f64 {
    let dot = left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>();
    let left_norm = left.iter().map(|value| value * value).sum::<f64>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f64>().sqrt();
    if left_norm <= 1e-12 || right_norm <= 1e-12 {
        0.0
    } else {
        (dot / (left_norm * right_norm)).clamp(0.0, 1.0)
    }
}

/// Estimate a small timeline offset from coarse RMS envelopes. Positive values
/// mean that rendered timestamps should move later to align with the reference.
fn estimate_envelope_alignment(reference: &Wav, reconstruction: &Wav) -> (f64, f64) {
    const RATE_HZ: usize = 20;
    const MAX_SHIFT: isize = 80; // four seconds in either direction
    fn envelope(wav: &Wav) -> Vec<f64> {
        let hop = (wav.sample_rate as usize / RATE_HZ).max(1);
        wav.samples
            .chunks(hop * wav.channels)
            .map(|chunk| {
                let mean_square = chunk.iter().map(|sample| sample * sample).sum::<f64>()
                    / chunk.len().max(1) as f64;
                mean_square.sqrt().max(1e-8).ln()
            })
            .collect()
    }
    let left = envelope(reference);
    let right = envelope(reconstruction);
    let mut scores = Vec::new();
    for shift in -MAX_SHIFT..=MAX_SHIFT {
        let start = 0isize.max(shift);
        let end = (left.len() as isize).min(right.len() as isize + shift);
        if end - start < (RATE_HZ * 8) as isize {
            continue;
        }
        let mut a = Vec::with_capacity((end - start) as usize);
        let mut b = Vec::with_capacity((end - start) as usize);
        for index in start..end {
            a.push(left[index as usize]);
            b.push(right[(index - shift) as usize]);
        }
        scores.push((shift, pearson(&a, &b)));
    }
    scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    let Some((best_shift, best_score)) = scores.first().copied() else {
        return (0.0, 0.0);
    };
    let competing_score = scores
        .iter()
        .find(|(shift, _)| (shift - best_shift).abs() >= 3)
        .map(|(_, score)| *score)
        .unwrap_or(-1.0);
    let confidence =
        ((best_score - competing_score) / (1.0 - competing_score).max(1e-9)).clamp(0.0, 1.0);
    (best_shift as f64 / RATE_HZ as f64, confidence)
}

fn pearson(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.len() < 2 {
        return 0.0;
    }
    let mean_left = left.iter().sum::<f64>() / left.len() as f64;
    let mean_right = right.iter().sum::<f64>() / right.len() as f64;
    let covariance = left
        .iter()
        .zip(right)
        .map(|(a, b)| (a - mean_left) * (b - mean_right))
        .sum::<f64>();
    let variance_left = left
        .iter()
        .map(|value| (value - mean_left).powi(2))
        .sum::<f64>();
    let variance_right = right
        .iter()
        .map(|value| (value - mean_right).powi(2))
        .sum::<f64>();
    covariance / (variance_left * variance_right).sqrt().max(1e-15)
}

fn stereo_character(wav: &Wav) -> StereoCharacter {
    let mut squared_sum = 0.0;
    let mut peak: f64 = 0.0;
    let mut left_right = 0.0;
    let mut left_squared = 0.0;
    let mut right_squared = 0.0;
    let mut mid_squared = 0.0;
    let mut side_squared = 0.0;
    let mut count = 0usize;
    for frame in wav.samples.chunks_exact(wav.channels) {
        for sample in frame {
            squared_sum += sample * sample;
            peak = peak.max(sample.abs());
        }
        let left = frame[0];
        let right = *frame.get(1).unwrap_or(&left);
        left_right += left * right;
        left_squared += left * left;
        right_squared += right * right;
        let mid = (left + right) * 0.5;
        let side = (left - right) * 0.5;
        mid_squared += mid * mid;
        side_squared += side * side;
        count += 1;
    }
    let sample_count = (count * wav.channels).max(1) as f64;
    let rms = (squared_sum / sample_count).sqrt();
    let correlation = left_right / (left_squared * right_squared).sqrt().max(1e-15);
    let width_db = 10.0 * (side_squared / mid_squared.max(1e-15)).max(1e-15).log10();
    StereoCharacter {
        rms_dbfs: 20.0 * rms.max(1e-15).log10(),
        peak_dbfs: 20.0 * peak.max(1e-15).log10(),
        channel_correlation: if wav.channels < 2 { 1.0 } else { correlation },
        side_to_mid_db: if wav.channels < 2 { -120.0 } else { width_db },
    }
}

fn read_wav(path: &Path) -> Result<Wav, String> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|e| format!("cannot read WAV: {e}"))?
        .read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read WAV: {e}"))?;
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("track analysis supports RIFF/WAVE files only".into());
    }
    let mut cursor = 12usize;
    let (mut format, mut channels, mut sample_rate, mut bits) = (0u16, 0usize, 0u32, 0u16);
    let mut data: Option<&[u8]> = None;
    while cursor + 8 <= bytes.len() {
        let chunk_len =
            u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        let start = cursor + 8;
        let end = start
            .checked_add(chunk_len)
            .ok_or("invalid WAV chunk size")?;
        if end > bytes.len() {
            return Err("truncated WAV chunk".into());
        }
        match &bytes[cursor..cursor + 4] {
            b"fmt " if chunk_len >= 16 => {
                format = u16::from_le_bytes(bytes[start..start + 2].try_into().unwrap());
                channels =
                    u16::from_le_bytes(bytes[start + 2..start + 4].try_into().unwrap()) as usize;
                sample_rate = u32::from_le_bytes(bytes[start + 4..start + 8].try_into().unwrap());
                bits = u16::from_le_bytes(bytes[start + 14..start + 16].try_into().unwrap());
                if format == 0xfffe && chunk_len >= 40 {
                    format = u16::from_le_bytes(bytes[start + 24..start + 26].try_into().unwrap());
                }
            }
            b"data" => data = Some(&bytes[start..end]),
            _ => {}
        }
        cursor = end + (chunk_len & 1);
    }
    if channels == 0 || channels > 8 || !(8_000..=384_000).contains(&sample_rate) {
        return Err(
            "WAV must have 1–8 channels and a sample rate between 8 kHz and 384 kHz".into(),
        );
    }
    let bytes_per_sample = (bits / 8) as usize;
    if !matches!((format, bits), (1, 16 | 24 | 32) | (3, 32)) || bytes_per_sample == 0 {
        return Err("track analysis supports PCM16/24/32 or float32 WAV".into());
    }
    let data = data.ok_or("WAV is missing its data chunk")?;
    let frame_bytes = bytes_per_sample * channels;
    let mut samples = Vec::with_capacity(data.len() / bytes_per_sample);
    for sample in data.chunks_exact(bytes_per_sample) {
        let value = match (format, bits) {
            (3, 32) => f32::from_le_bytes(sample.try_into().unwrap()) as f64,
            (_, 16) => i16::from_le_bytes(sample.try_into().unwrap()) as f64 / 32768.0,
            (_, 24) => {
                let raw =
                    (sample[0] as i32) | ((sample[1] as i32) << 8) | ((sample[2] as i32) << 16);
                ((raw << 8) >> 8) as f64 / 8_388_608.0
            }
            (_, 32) => i32::from_le_bytes(sample.try_into().unwrap()) as f64 / 2_147_483_648.0,
            _ => unreachable!(),
        };
        if !value.is_finite() {
            return Err("WAV contains non-finite samples".into());
        }
        samples.push(value);
    }
    if samples.len() < frame_bytes / bytes_per_sample || samples.len() % channels != 0 {
        return Err("WAV contains no complete audio frames".into());
    }
    Ok(Wav {
        samples,
        channels,
        sample_rate,
    })
}

fn estimate_tempo(samples: &[f64], sample_rate: u32) -> (f64, f64) {
    // Build a 100 Hz positive energy-change envelope, then find the strongest
    // periodicity in the usual 60–180 BPM range.
    let hop = (sample_rate as usize / 100).max(1);
    let envelope: Vec<f64> = samples
        .chunks(hop)
        .map(|chunk| (chunk.iter().map(|x| x * x).sum::<f64>() / chunk.len() as f64).sqrt())
        .collect();
    let novelty: Vec<f64> = envelope
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).max(0.0))
        .collect();
    if novelty.len() < 100 {
        return (120.0, 0.0);
    }
    let mean = novelty.iter().sum::<f64>() / novelty.len() as f64;
    let centered: Vec<f64> = novelty.iter().map(|x| x - mean).collect();
    let min_lag = 33usize; // about 182 BPM
    let max_lag = 100usize; // 60 BPM
    let mut scores = Vec::with_capacity(max_lag - min_lag + 1);
    for lag in min_lag..=max_lag {
        let left = &centered[..centered.len() - lag];
        let right = &centered[lag..];
        let cross = left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>();
        let left_energy = left.iter().map(|x| x * x).sum::<f64>();
        let right_energy = right.iter().map(|x| x * x).sum::<f64>();
        let correlation = cross / (left_energy * right_energy).sqrt().max(1e-20);
        scores.push((lag, correlation));
    }
    let best_index = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.1.total_cmp(&b.1.1))
        .map(|(index, _)| index)
        .unwrap_or(0);
    let (best_lag, peak) = scores[best_index];
    // Refine the selected autocorrelation peak so an integer 100 Hz frame lag
    // does not quantize common tempos (for example, 123 BPM to 122 BPM).
    let fractional_lag = if best_index > 0 && best_index + 1 < scores.len() {
        let before = scores[best_index - 1].1;
        let after = scores[best_index + 1].1;
        let denominator = before - 2.0 * peak + after;
        if denominator.abs() > 1e-12 {
            best_lag as f64 + (0.5 * (before - after) / denominator).clamp(-0.5, 0.5)
        } else {
            best_lag as f64
        }
    } else {
        best_lag as f64
    };
    // Compare with the baseline of the full tempo range, not the adjacent BPM
    // bin; neighboring tempos share almost the same rounded 100 Hz lag.
    let mut baseline: Vec<f64> = scores.iter().map(|(_, score)| *score).collect();
    baseline.sort_by(f64::total_cmp);
    let median = baseline[baseline.len() / 2].max(0.0);
    let confidence = ((peak - median) / (1.0 - median).max(1e-20)).clamp(0.0, 1.0);
    let bpm = 60.0 * 100.0 / fractional_lag;
    (bpm.clamp(60.0, 180.0), confidence)
}

fn spectral_features(samples: &[f64], sample_rate: u32) -> ([f64; 12], [f64; 3], Vec<ChromaFrame>) {
    const N: usize = 8192;
    let hop = 4096usize;
    let mut chroma = [0.0; 12];
    let mut bands = [0.0; 3];
    let mut frames = Vec::new();
    if samples.len() < N {
        return (chroma, bands, frames);
    }
    let mut real = vec![0.0; N];
    let mut imag = vec![0.0; N];
    for start in (0..=samples.len() - N).step_by(hop) {
        for i in 0..N {
            let window = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / N as f64).cos();
            real[i] = samples[start + i] * window;
            imag[i] = 0.0;
        }
        fft(&mut real, &mut imag);
        let mut frame_chroma = [0.0; 12];
        for bin in 1..N / 2 {
            let hz = bin as f64 * sample_rate as f64 / N as f64;
            let power = real[bin] * real[bin] + imag[bin] * imag[bin];
            if hz < 180.0 {
                bands[0] += power;
            } else if hz < 4_000.0 {
                bands[1] += power;
            } else {
                bands[2] += power;
            }
            if (40.0..=5_000.0).contains(&hz) {
                let midi = 69.0 + 12.0 * (hz / 440.0).log2();
                // Magnitude weighting and per-frame normalization prevent a
                // handful of loud bass bins or sections from dominating key.
                frame_chroma[(midi.round() as i32).rem_euclid(12) as usize] += power.sqrt();
            }
        }
        let frame_total = frame_chroma.iter().sum::<f64>().max(1e-20);
        let mut normalized_chroma = [0.0; 12];
        for pitch_class in 0..12 {
            normalized_chroma[pitch_class] = frame_chroma[pitch_class] / frame_total;
            chroma[pitch_class] += normalized_chroma[pitch_class];
        }
        frames.push(ChromaFrame {
            time_seconds: (start + N / 2) as f64 / sample_rate as f64,
            values: normalized_chroma,
            energy: frame_chroma.iter().sum(),
        });
    }
    (chroma, bands, frames)
}

fn analyze_bars(
    frames: &[ChromaFrame],
    duration: f64,
    beat_interval_seconds: f64,
    timeline_offset_seconds: f64,
    samples: &[f64],
    sample_rate: u32,
) -> Vec<BarAnalysis> {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let bar_seconds = beat_interval_seconds * 4.0;
    let bar_count = ((duration + timeline_offset_seconds).max(0.0) / bar_seconds).ceil() as usize;
    let mut bars = Vec::with_capacity(bar_count);
    let mut max_energy: f64 = 0.0;
    let mut aggregate = Vec::with_capacity(bar_count);
    for bar in 0..bar_count {
        let start = bar as f64 * bar_seconds;
        let end = start + bar_seconds;
        let selected: Vec<&ChromaFrame> = frames
            .iter()
            .filter(|frame| {
                let aligned_time = frame.time_seconds + timeline_offset_seconds;
                aligned_time >= start && aligned_time < end
            })
            .collect();
        let mut chroma = [0.0; 12];
        let mut energy = 0.0;
        for frame in &selected {
            for (chroma_value, frame_value) in chroma.iter_mut().zip(frame.values) {
                *chroma_value += frame_value;
            }
            energy += frame.energy;
        }
        if !selected.is_empty() {
            for value in &mut chroma {
                *value /= selected.len() as f64;
            }
            energy /= selected.len() as f64;
        }
        max_energy = max_energy.max(energy);
        let (chord, confidence, top_candidates) = estimate_chord(&chroma);
        let audio_start =
            ((start - timeline_offset_seconds).max(0.0) * sample_rate as f64).floor() as usize;
        let audio_end =
            ((end - timeline_offset_seconds).max(0.0) * sample_rate as f64).ceil() as usize;
        let audio_end = audio_end.min(samples.len());
        let audio_start = audio_start.min(audio_end);
        let mean_square = if audio_end > audio_start {
            samples[audio_start..audio_end]
                .iter()
                .map(|sample| sample * sample)
                .sum::<f64>()
                / (audio_end - audio_start) as f64
        } else {
            0.0
        };
        let rms_dbfs = 10.0 * mean_square.max(1e-20).log10();
        let mut ranked: Vec<(usize, f64)> = chroma.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        aggregate.push((
            chord,
            confidence,
            top_candidates,
            chroma,
            ranked,
            energy,
            start,
            end,
            rms_dbfs,
        ));
    }
    for (
        index,
        (chord, confidence, top_candidates, chroma, ranked, energy, start, end, rms_dbfs),
    ) in aggregate.into_iter().enumerate()
    {
        bars.push(BarAnalysis {
            bar: index + 1,
            start_seconds: start,
            end_seconds: end,
            chord,
            chord_confidence: confidence,
            chord_confidence_method: "uncalibrated_major_minor_triad_template_margin".into(),
            top_chord_candidates: top_candidates,
            strongest_pitch_classes: ranked
                .into_iter()
                .take(3)
                .map(|(pitch, _)| NAMES[pitch].to_string())
                .collect(),
            pitch_class_profile: chroma,
            relative_energy: (energy / max_energy.max(1e-20)).sqrt().clamp(0.0, 1.0),
            rms_dbfs,
        });
    }
    bars
}

fn estimate_chord(chroma: &[f64; 12]) -> (String, f64, Vec<ChordCandidate>) {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let total = chroma.iter().sum::<f64>().max(1e-20);
    let mut candidates = Vec::with_capacity(24);
    for root in 0..12 {
        for (quality, third) in [("major", 4usize), ("minor", 3usize)] {
            let score =
                (chroma[root] + chroma[(root + third) % 12] + chroma[(root + 7) % 12]) / total;
            candidates.push((score, root, quality));
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    let best = candidates.first().copied().unwrap_or((0.0, 0, "major"));
    let runner_up = candidates
        .get(1)
        .map(|candidate| candidate.0)
        .unwrap_or(0.0);
    let confidence = ((best.0 - runner_up) / best.0.max(1e-20)).clamp(0.0, 1.0);
    let top_candidates = candidates
        .iter()
        .take(5)
        .map(|(score, root, quality)| ChordCandidate {
            chord: format!("{} {}", NAMES[*root], quality),
            template_score: *score,
        })
        .collect();
    (
        format!("{} {}", NAMES[best.1], best.2),
        confidence,
        top_candidates,
    )
}

fn merge_chords(bars: &[BarAnalysis]) -> Vec<ChordSegment> {
    let mut segments: Vec<ChordSegment> = Vec::new();
    for bar in bars {
        if let Some(last) = segments.last_mut() {
            if last.chord == bar.chord {
                let count = (last.end_bar - last.start_bar + 1) as f64;
                last.confidence = (last.confidence * count + bar.chord_confidence) / (count + 1.0);
                last.end_bar = bar.bar;
                continue;
            }
        }
        segments.push(ChordSegment {
            chord: bar.chord.clone(),
            start_bar: bar.bar,
            end_bar: bar.bar,
            confidence: bar.chord_confidence,
        });
    }
    segments
}

fn detect_transients(
    samples: &[f64],
    sample_rate: u32,
    beat_interval_seconds: f64,
) -> Vec<TransientEvent> {
    let hop = (sample_rate as usize / 100).max(1);
    let low_alpha = 1.0 - (-std::f64::consts::TAU * 180.0 / sample_rate as f64).exp();
    let upper_alpha = 1.0 - (-std::f64::consts::TAU * 4_000.0 / sample_rate as f64).exp();
    let mut low_state = 0.0;
    let mut upper_state = 0.0;
    let mut bands: Vec<[f64; 3]> = Vec::with_capacity(samples.len() / hop + 1);
    let mut sums = [0.0; 3];
    let mut count = 0usize;
    for sample in samples {
        low_state += low_alpha * (*sample - low_state);
        upper_state += upper_alpha * (*sample - upper_state);
        let values = [low_state, upper_state - low_state, *sample - upper_state];
        for band in 0..3 {
            sums[band] += values[band] * values[band];
        }
        count += 1;
        if count == hop {
            bands.push([
                (sums[0] / count as f64).sqrt(),
                (sums[1] / count as f64).sqrt(),
                (sums[2] / count as f64).sqrt(),
            ]);
            sums = [0.0; 3];
            count = 0;
        }
    }
    if count > 0 {
        bands.push([
            (sums[0] / count as f64).sqrt(),
            (sums[1] / count as f64).sqrt(),
            (sums[2] / count as f64).sqrt(),
        ]);
    }
    let mut novelty = vec![[0.0; 3]; bands.len()];
    for index in 1..bands.len() {
        for band in 0..3 {
            novelty[index][band] = (bands[index][band] - bands[index - 1][band]).max(0.0);
        }
    }
    let mut scales = [0.0; 3];
    for band in 0..3 {
        let mut values: Vec<f64> = novelty.iter().map(|frame| frame[band]).collect();
        values.sort_by(f64::total_cmp);
        scales[band] = values[(values.len() * 95 / 100).min(values.len() - 1)].max(1e-12);
    }
    let mut events = Vec::new();
    let mut last_event_frame = 0usize;
    for index in 2..novelty.len().saturating_sub(2) {
        let normalized = [
            novelty[index][0] / scales[0],
            novelty[index][1] / scales[1],
            novelty[index][2] / scales[2],
        ];
        let strength = normalized.iter().copied().fold(0.0, f64::max);
        let local_max = (index - 2..=index + 2).all(|neighbor| {
            novelty[index][0] + novelty[index][1] + novelty[index][2]
                >= novelty[neighbor][0] + novelty[neighbor][1] + novelty[neighbor][2]
        });
        if !local_max || strength < 0.45 || index.saturating_sub(last_event_frame) < 8 {
            continue;
        }
        let band = normalized
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .unwrap_or(1);
        let time = index as f64 * hop as f64 / sample_rate as f64;
        let beat_position = time / beat_interval_seconds;
        let nearest_beat = beat_position.round() as usize;
        let fraction = (beat_position - beat_position.floor()).rem_euclid(1.0);
        let subdivision = match (fraction * 4.0).round() as usize {
            0 => "beat",
            1 => "16th offset",
            2 => "eighth offbeat",
            _ => "16th offset",
        };
        let character = match band {
            0 => "low transient (kick-like)",
            1 => "mid transient (snare/percussion-like)",
            _ => "high transient (hat/cymbal-like)",
        };
        events.push(TransientEvent {
            time_seconds: time,
            nearest_beat,
            beat_subdivision: subdivision.into(),
            character: character.into(),
            strength: strength.min(1.0),
        });
        last_event_frame = index;
    }
    events
}

fn characterize_sound(bands: &[f64; 3], total: f64, sections: &[EnergySection]) -> SoundCharacter {
    let low = bands[0] / total;
    let high = bands[2] / total;
    let descriptor = if low > 0.65 {
        "Low-frequency energy dominates the measured spectrum."
    } else if high > 0.12 {
        "The measured spectrum has relatively strong high-frequency energy."
    } else {
        "The measured spectrum is weighted toward low and mid frequencies."
    };
    let dynamics = if sections
        .iter()
        .map(|section| section.relative_energy)
        .fold(0.0, f64::max)
        - sections
            .iter()
            .map(|section| section.relative_energy)
            .fold(1.0, f64::min)
        > 0.6
    {
        "Large energy changes across the arrangement."
    } else {
        "Relatively steady energy across the arrangement."
    };
    SoundCharacter {
        description: format!("{descriptor} {dynamics}"),
        notes: vec![
            "Spectral balance measures aggregate frequency energy; it is not an EQ curve or instrument detector.",
            "Transient labels are heuristic frequency bands, not verified drum identities.",
        ],
    }
}

fn fft(real: &mut [f64], imag: &mut [f64]) {
    let n = real.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = -std::f64::consts::TAU / len as f64;
        let (wlen_r, wlen_i) = (angle.cos(), angle.sin());
        for base in (0..n).step_by(len) {
            let (mut wr, mut wi) = (1.0, 0.0);
            for offset in 0..len / 2 {
                let even = base + offset;
                let odd = even + len / 2;
                let tr = real[odd] * wr - imag[odd] * wi;
                let ti = real[odd] * wi + imag[odd] * wr;
                real[odd] = real[even] - tr;
                imag[odd] = imag[even] - ti;
                real[even] += tr;
                imag[even] += ti;
                (wr, wi) = (wr * wlen_r - wi * wlen_i, wr * wlen_i + wi * wlen_r);
            }
        }
        len <<= 1;
    }
}

fn estimate_key(chroma: &[f64; 12]) -> (String, f64) {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    const MAJOR: [f64; 12] = [
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
    ];
    const MINOR: [f64; 12] = [
        6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
    ];
    let mean = chroma.iter().sum::<f64>() / 12.0;
    let centered: Vec<f64> = chroma.iter().map(|x| x - mean).collect();
    let norm = centered
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
        .max(1e-20);
    let mut candidates = Vec::new();
    for root in 0..12 {
        for (scale, profile) in [("major", &MAJOR), ("minor", &MINOR)] {
            let pmean = profile.iter().sum::<f64>() / 12.0;
            let pc: Vec<f64> = profile.iter().map(|x| x - pmean).collect();
            let pnorm = pc.iter().map(|x| x * x).sum::<f64>().sqrt();
            let score = (0..12)
                .map(|i| centered[(i + root) % 12] * pc[i])
                .sum::<f64>()
                / (norm * pnorm);
            candidates.push((score, format!("{} {scale}", NAMES[root])));
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    let best = candidates
        .first()
        .cloned()
        .unwrap_or((0.0, "unknown".into()));
    let second = candidates.get(1).map(|x| x.0).unwrap_or(0.0);
    (
        best.1,
        ((best.0 - second) / best.0.abs().max(1e-20)).clamp(0.0, 1.0),
    )
}
