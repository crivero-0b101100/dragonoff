// Copyright (C) 2026 OpenAI
// This program is free software under the GNU GPL v2 or (at your option) any later version.

//! Lightweight, dependency-free analysis of a finished stereo or mono WAV mix.
//!
//! These estimates describe audible properties of the mix. They do not recover
//! the original multitrack session, instruments, or effect settings.

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
    pub tempo_confidence: f64,
    pub key: String,
    pub key_confidence: f64,
    pub spectral_balance: SpectralBalance,
    pub sections: Vec<EnergySection>,
    pub limitations: &'static str,
}

#[derive(Debug, Serialize)]
pub struct SpectralBalance {
    pub low_below_180_hz: f64,
    pub mid_180_to_4000_hz: f64,
    pub high_above_4000_hz: f64,
}

#[derive(Debug, Serialize)]
pub struct EnergySection {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub rms_dbfs: f64,
    pub relative_energy: f64,
}

struct Wav {
    samples: Vec<f64>,
    channels: usize,
    sample_rate: u32,
}

pub fn analyze_wav(path: &Path) -> Result<TrackAnalysis, String> {
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

    let section_len = (wav.sample_rate as usize * 4).max(1);
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

    let (tempo_bpm, tempo_confidence) = estimate_tempo(&mono, wav.sample_rate);
    let (chroma, band_energy) = spectral_features(&mono, wav.sample_rate);
    let (key, key_confidence) = estimate_key(&chroma);
    let total = band_energy.iter().sum::<f64>().max(1e-20);

    Ok(TrackAnalysis {
        source: path.display().to_string(),
        duration_seconds: duration,
        sample_rate: wav.sample_rate,
        tempo_bpm,
        tempo_confidence,
        key,
        key_confidence,
        spectral_balance: SpectralBalance {
            low_below_180_hz: band_energy[0] / total,
            mid_180_to_4000_hz: band_energy[1] / total,
            high_above_4000_hz: band_energy[2] / total,
        },
        sections,
        limitations: "Tempo, key, and energy sections are signal-based estimates. This tool does not separate stems, transcribe notes/chords, identify instruments, or recover original synth/effect settings.",
    })
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
    if channels == 0 || channels > 8 || sample_rate < 8_000 || sample_rate > 384_000 {
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

fn spectral_features(samples: &[f64], sample_rate: u32) -> ([f64; 12], [f64; 3]) {
    const N: usize = 8192;
    let hop = 4096usize;
    let mut chroma = [0.0; 12];
    let mut bands = [0.0; 3];
    if samples.len() < N {
        return (chroma, bands);
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
        for pitch_class in 0..12 {
            chroma[pitch_class] += frame_chroma[pitch_class] / frame_total;
        }
    }
    (chroma, bands)
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
