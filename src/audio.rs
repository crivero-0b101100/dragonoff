// Copyright (C) 2026 OpenAI
// This program is free software under the GNU GPL v2 or (at your option) any later version.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioMetrics {
    pub peak: f64,
    pub estimated_true_peak_dbfs: f64,
    pub rms: f64,
    pub integrated_lufs: f64,
    pub clipped_fraction: f64,
    pub low_ratio: f64,
    pub high_ratio: f64,
    pub stereo_correlation: f64,
    pub stereo_side_mid_db: f64,
}

#[derive(Clone, Copy)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    fn high_shelf(sample_rate: f64, frequency: f64, gain_db: f64, q: f64) -> Self {
        let a = 10.0f64.powf(gain_db / 40.0);
        let w = std::f64::consts::TAU * frequency / sample_rate;
        let c = w.cos();
        let alpha = w.sin() * 0.5 * (((a + 1.0 / a) * (1.0 / q - 1.0) + 2.0).max(0.0)).sqrt();
        let root_a = a.sqrt();
        Self::normalized(
            a * ((a + 1.0) + (a - 1.0) * c + 2.0 * root_a * alpha),
            -2.0 * a * ((a - 1.0) + (a + 1.0) * c),
            a * ((a + 1.0) + (a - 1.0) * c - 2.0 * root_a * alpha),
            (a + 1.0) - (a - 1.0) * c + 2.0 * root_a * alpha,
            2.0 * ((a - 1.0) - (a + 1.0) * c),
            (a + 1.0) - (a - 1.0) * c - 2.0 * root_a * alpha,
        )
    }

    fn high_pass(sample_rate: f64, frequency: f64, q: f64) -> Self {
        let w = std::f64::consts::TAU * frequency / sample_rate;
        let c = w.cos();
        let alpha = w.sin() / (2.0 * q);
        Self::normalized(
            (1.0 + c) * 0.5,
            -(1.0 + c),
            (1.0 + c) * 0.5,
            1.0 + alpha,
            -2.0 * c,
            1.0 - alpha,
        )
    }

    fn normalized(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn process(&mut self, input: f64) -> f64 {
        let output = self.b0 * input + self.z1;
        self.z1 = self.b1 * input - self.a1 * output + self.z2;
        self.z2 = self.b2 * input - self.a2 * output;
        output
    }
}

struct TruePeakEstimate {
    history: [f64; 4],
    filled: usize,
    peak: f64,
}

impl TruePeakEstimate {
    fn new() -> Self {
        Self {
            history: [0.0; 4],
            filled: 0,
            peak: 0.0,
        }
    }

    fn push(&mut self, sample: f64) {
        self.peak = self.peak.max(sample.abs());
        if self.filled < 4 {
            self.history[self.filled] = sample;
            self.filled += 1;
            return;
        }
        self.history.rotate_left(1);
        self.history[3] = sample;
        let [y0, y1, y2, y3] = self.history;
        for t in [0.25f64, 0.5, 0.75] {
            let t2 = t * t;
            let t3 = t2 * t;
            let value = 0.5
                * ((2.0 * y1)
                    + (-y0 + y2) * t
                    + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t2
                    + (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t3);
            self.peak = self.peak.max(value.abs());
        }
    }
}
pub fn analyze_wav(path: &Path) -> Result<AudioMetrics, String> {
    let f = File::open(path).map_err(|e| format!("cannot read rendered WAV: {e}"))?;
    let mut r = BufReader::with_capacity(64 * 1024, f);
    let mut header = [0u8; 44];
    r.read_exact(&mut header)
        .map_err(|e| format!("WAV header read failed: {e}"))?;
    if &header[0..4] != b"RIFF"
        || &header[8..12] != b"WAVE"
        || !matches!(u16::from_le_bytes([header[20], header[21]]), 1 | 3)
        || u16::from_le_bytes([header[22], header[23]]) != 2
        || !matches!(u16::from_le_bytes([header[34], header[35]]), 16 | 24 | 32)
    {
        return Err("audio analysis supports stereo PCM16/24/32 or float32 WAV".into());
    }
    let bits = u16::from_le_bytes([header[34], header[35]]);
    let format = u16::from_le_bytes([header[20], header[21]]);
    let sample_bytes = bits as usize / 8;
    let frame_bytes = sample_bytes * 2;
    let mut remaining = u32::from_le_bytes(header[40..44].try_into().unwrap()) as usize;
    let mut buf = vec![0u8; frame_bytes * 4096];
    let decode = |bytes: &[u8]| -> f64 {
        match (format, bits) {
            (3, 32) => f32::from_le_bytes(bytes[..4].try_into().unwrap()) as f64,
            (_, 16) => i16::from_le_bytes(bytes[..2].try_into().unwrap()) as f64 / 32768.0,
            (_, 24) => {
                ((i32::from_le_bytes([bytes[0], bytes[1], bytes[2], 0]) << 8) >> 8) as f64
                    / 8388608.0
            }
            _ => i32::from_le_bytes(bytes[..4].try_into().unwrap()) as f64 / 2147483648.0,
        }
    };
    let mut peak = 0.0f64;
    let mut sq = 0.0f64;
    let mut low_sq = 0.0f64;
    let mut high_sq = 0.0f64;
    let mut count = 0u64;
    let mut clipped = 0u64;
    let sr = u32::from_le_bytes([header[24], header[25], header[26], header[27]]) as f64;
    let mut low_state = 0.0f64;
    let alpha = (1.0 - (-2.0 * std::f64::consts::PI * 180.0 / sr).exp()).clamp(0.0, 1.0);
    let mut k_left_shelf = Biquad::high_shelf(
        sr,
        1_681.974_450_955_533,
        3.999_843_853_973_347,
        0.707_175_236_955_419_6,
    );
    let mut k_left_highpass = Biquad::high_pass(sr, 38.135_470_876_024_44, 0.500_327_037_323_877_8);
    let mut k_right_shelf = k_left_shelf;
    let mut k_right_highpass = k_left_highpass;
    let hop_frames = (sr * 0.1).round().max(1.0) as u64;
    let mut hop_count = 0u64;
    let mut hop_energy = 0.0f64;
    let mut recent_hops = VecDeque::with_capacity(4);
    let mut rolling_energy = 0.0f64;
    let mut loudness_blocks = Vec::new();
    let mut true_peak_l = TruePeakEstimate::new();
    let mut true_peak_r = TruePeakEstimate::new();
    let mut sum_lr = 0.0f64;
    let mut sum_l2 = 0.0f64;
    let mut sum_r2 = 0.0f64;
    let mut mid_sq = 0.0f64;
    let mut side_high_sq = 0.0f64;
    let mut side_low_state = 0.0f64;
    let side_alpha = (1.0 - (-2.0 * std::f64::consts::PI * 120.0 / sr).exp()).clamp(0.0, 1.0);
    while remaining >= frame_bytes {
        let n = remaining.min(buf.len());
        let n = n - n % frame_bytes;
        r.read_exact(&mut buf[..n])
            .map_err(|e| format!("truncated WAV data: {e}"))?;
        remaining -= n;
        for ch in buf[..n].chunks_exact(frame_bytes) {
            let l = decode(ch);
            let rr = decode(&ch[sample_bytes..]);
            if !l.is_finite() || !rr.is_finite() {
                return Err("non-finite WAV samples".into());
            }
            let x = (l + rr) * 0.5;
            let side = (l - rr) * 0.5;
            side_low_state += side_alpha * (side - side_low_state);
            let side_high = side - side_low_state;
            true_peak_l.push(l);
            true_peak_r.push(rr);
            sum_lr += l * rr;
            sum_l2 += l * l;
            sum_r2 += rr * rr;
            mid_sq += x * x;
            side_high_sq += side_high * side_high;
            peak = peak.max(l.abs()).max(rr.abs());
            if l.abs() >= 0.999 || rr.abs() >= 0.999 {
                clipped += 1;
            }
            sq += x * x;
            low_state += alpha * (x - low_state);
            low_sq += low_state * low_state;
            high_sq += (x - low_state) * (x - low_state);
            let kw_l = k_left_highpass.process(k_left_shelf.process(l));
            let kw_r = k_right_highpass.process(k_right_shelf.process(rr));
            hop_energy += kw_l * kw_l + kw_r * kw_r;
            hop_count += 1;
            if hop_count >= hop_frames {
                recent_hops.push_back(hop_energy);
                rolling_energy += hop_energy;
                if recent_hops.len() > 4 {
                    rolling_energy -= recent_hops.pop_front().unwrap_or(0.0);
                }
                if recent_hops.len() == 4 {
                    loudness_blocks.push(rolling_energy / (hop_frames * 4) as f64);
                }
                hop_count = 0;
                hop_energy = 0.0;
            }
            count += 1;
        }
    }
    if count == 0 {
        return Err("WAV contains no samples".into());
    }
    let rms = (sq / count as f64).sqrt();
    let total_band = low_sq + high_sq;
    let absolute_gate = 10.0f64.powf((-70.0 + 0.691) / 10.0);
    let absolute_blocks: Vec<f64> = loudness_blocks
        .iter()
        .copied()
        .filter(|energy| *energy > absolute_gate)
        .collect();
    let absolute_mean = if absolute_blocks.is_empty() {
        0.0
    } else {
        absolute_blocks.iter().sum::<f64>() / absolute_blocks.len() as f64
    };
    let relative_gate = absolute_mean * 0.1;
    let gated_blocks: Vec<f64> = absolute_blocks
        .into_iter()
        .filter(|energy| *energy >= relative_gate)
        .collect();
    let integrated_lufs = if gated_blocks.is_empty() {
        -70.0
    } else {
        let mean = gated_blocks.iter().sum::<f64>() / gated_blocks.len() as f64;
        -0.691 + 10.0 * mean.max(1e-20).log10()
    };
    let estimated_true_peak = true_peak_l.peak.max(true_peak_r.peak).max(1e-20);
    let stereo_correlation = sum_lr / (sum_l2 * sum_r2).sqrt().max(1e-20);
    let stereo_side_mid_db = 10.0 * (side_high_sq / mid_sq.max(1e-20)).max(1e-20).log10();
    Ok(AudioMetrics {
        peak,
        estimated_true_peak_dbfs: 20.0 * estimated_true_peak.log10(),
        rms,
        integrated_lufs,
        clipped_fraction: clipped as f64 / (count * 2) as f64,
        low_ratio: if total_band <= 0.0 {
            0.0
        } else {
            low_sq / total_band
        },
        high_ratio: if total_band <= 0.0 {
            0.0
        } else {
            high_sq / total_band
        },
        stereo_correlation,
        stereo_side_mid_db,
    })
}
