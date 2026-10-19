// Copyright (C) 2026 OpenAI; GPL-2.0-or-later.
// RBJ filter equations: https://www.w3.org/TR/audio-eq-cookbook/
use crate::model::{Effects, EqBand};
use serde::Serialize;
use std::f64::consts::{PI, TAU};

pub fn gain(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}
pub fn db(x: f64) -> f64 {
    20.0 * x.max(1e-12).log10()
}

/// Double precision states keep low-frequency EQ well conditioned at high sample rates.
#[derive(Clone, Copy)]
pub struct Biquad {
    b: [f64; 3],
    a: [f64; 2],
    z: [f64; 2],
}
impl Biquad {
    pub fn identity() -> Self {
        Self {
            b: [1.0, 0.0, 0.0],
            a: [0.0; 2],
            z: [0.0; 2],
        }
    }
    pub fn design(kind: &str, sr: f32, hz: f32, q: f32, gain_db: f32) -> Self {
        let w = TAU * (hz as f64).clamp(5.0, sr as f64 * 0.45) / sr as f64;
        let (s, c) = w.sin_cos();
        let alpha = s / (2.0 * (q as f64).clamp(0.1, 16.0));
        let a = 10.0f64.powf(gain_db as f64 / 40.0);
        let beta = 2.0 * a.sqrt() * s / 2.0f64.sqrt();
        let (b0, b1, b2, a0, a1, a2) = match kind {
            "highpass" => (
                (1.0 + c) / 2.0,
                -(1.0 + c),
                (1.0 + c) / 2.0,
                1.0 + alpha,
                -2.0 * c,
                1.0 - alpha,
            ),
            "lowpass" => (
                (1.0 - c) / 2.0,
                1.0 - c,
                (1.0 - c) / 2.0,
                1.0 + alpha,
                -2.0 * c,
                1.0 - alpha,
            ),
            "bandpass" => (alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * c, 1.0 - alpha),
            "notch" => (1.0, -2.0 * c, 1.0, 1.0 + alpha, -2.0 * c, 1.0 - alpha),
            "low_shelf" => (
                a * ((a + 1.0) - (a - 1.0) * c + beta),
                2.0 * a * ((a - 1.0) - (a + 1.0) * c),
                a * ((a + 1.0) - (a - 1.0) * c - beta),
                (a + 1.0) + (a - 1.0) * c + beta,
                -2.0 * ((a - 1.0) + (a + 1.0) * c),
                (a + 1.0) + (a - 1.0) * c - beta,
            ),
            "high_shelf" => (
                a * ((a + 1.0) + (a - 1.0) * c + beta),
                -2.0 * a * ((a - 1.0) + (a + 1.0) * c),
                a * ((a + 1.0) + (a - 1.0) * c - beta),
                (a + 1.0) - (a - 1.0) * c + beta,
                2.0 * ((a - 1.0) - (a + 1.0) * c),
                (a + 1.0) - (a - 1.0) * c - beta,
            ),
            _ => (
                1.0 + alpha * a,
                -2.0 * c,
                1.0 - alpha * a,
                1.0 + alpha / a,
                -2.0 * c,
                1.0 - alpha / a,
            ),
        };
        Self {
            b: [b0 / a0, b1 / a0, b2 / a0],
            a: [a1 / a0, a2 / a0],
            z: [0.0; 2],
        }
    }
    pub fn highpass(sr: f32, hz: f32) -> Self {
        Self::design("highpass", sr, hz, 0.70710677, 0.0)
    }
    pub fn retune(&mut self, coefficients: Self) {
        self.b = coefficients.b;
        self.a = coefficients.a;
    }
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let x = x as f64;
        let y = self.b[0] * x + self.z[0];
        self.z[0] = self.b[1] * x - self.a[0] * y + self.z[1];
        self.z[1] = self.b[2] * x - self.a[1] * y;
        y as f32
    }
}

struct EqualizerBand {
    config: EqBand,
    filters: [Biquad; 2],
    detector: [Biquad; 2],
    rms: f64,
    reduction: f32,
    attack: f64,
    release: f64,
    frame: u64,
    sr: f32,
}
impl EqualizerBand {
    fn new(c: &EqBand, sr: f32) -> Self {
        Self {
            config: c.clone(),
            filters: [Biquad::design(&c.kind, sr, c.frequency_hz, c.q, c.gain_db); 2],
            detector: [Biquad::design("bandpass", sr, c.frequency_hz, c.q, 0.0); 2],
            rms: 0.0,
            reduction: 0.0,
            attack: (-1.0 / (sr as f64 * c.attack_ms.max(0.1) as f64 / 1000.0)).exp(),
            release: (-1.0 / (sr as f64 * c.release_ms.max(5.0) as f64 / 1000.0)).exp(),
            frame: 0,
            sr,
        }
    }
    fn process(&mut self, l: f32, r: f32) -> (f32, f32) {
        if let Some(threshold) = self.config.dynamic_threshold_db {
            let dl = self.detector[0].process(l);
            let dr = self.detector[1].process(r);
            let power = ((dl * dl + dr * dr) * 0.5) as f64;
            let coeff = if power > self.rms {
                self.attack
            } else {
                self.release
            };
            self.rms = power + coeff * (self.rms - power);
            let over = (10.0 * self.rms.max(1e-20).log10() - threshold as f64).max(0.0) as f32;
            let desired = (over * (1.0 - 1.0 / self.config.dynamic_ratio.max(1.0)))
                .min(self.config.dynamic_range_db.abs());
            self.reduction += (desired - self.reduction) * (1.0 - (-1.0 / (self.sr * 0.01)).exp());
            if self.frame.is_multiple_of(64) {
                let c = Biquad::design(
                    &self.config.kind,
                    self.sr,
                    self.config.frequency_hz,
                    self.config.q,
                    self.config.gain_db - self.reduction,
                );
                for f in &mut self.filters {
                    f.retune(c);
                }
            }
        }
        self.frame += 1;
        (self.filters[0].process(l), self.filters[1].process(r))
    }
}

/// Linked RMS compressor, soft knee, explicit makeup, optional detector highpass.
pub struct Compressor {
    power: f64,
    attack: f64,
    release: f64,
    hp: [Biquad; 2],
    config: Effects,
    pub max_reduction_db: f32,
    total_reduction: f64,
    frames: u64,
}
impl Compressor {
    pub fn new(c: &Effects, sr: u32) -> Self {
        let hp = if c.compressor_sidechain_highpass_hz >= 5.0 {
            Biquad::highpass(sr as f32, c.compressor_sidechain_highpass_hz)
        } else {
            Biquad::identity()
        };
        Self {
            power: 0.0,
            attack: (-1.0 / (sr as f64 * c.compressor_attack_ms.max(0.1) as f64 / 1000.0)).exp(),
            release: (-1.0 / (sr as f64 * c.compressor_release_ms.max(5.0) as f64 / 1000.0)).exp(),
            hp: [hp; 2],
            config: c.clone(),
            max_reduction_db: 0.0,
            total_reduction: 0.0,
            frames: 0,
        }
    }
    pub fn process(&mut self, l: f32, r: f32) -> (f32, f32) {
        let dl = self.hp[0].process(l);
        let dr = self.hp[1].process(r);
        let power = ((dl * dl + dr * dr) * 0.5) as f64;
        let coeff = if power > self.power {
            self.attack
        } else {
            self.release
        };
        self.power = power + coeff * (self.power - power);
        let over =
            (10.0 * self.power.max(1e-20).log10()) as f32 - self.config.compressor_threshold_db;
        let knee = self.config.compressor_knee_db.max(0.0);
        let slope = 1.0 - 1.0 / self.config.compressor_ratio.max(1.0);
        let gr = if knee > 0.0 && over.abs() < knee * 0.5 {
            slope * (over + knee * 0.5).powi(2) / (2.0 * knee)
        } else if over > 0.0 {
            slope * over
        } else {
            0.0
        };
        self.max_reduction_db = self.max_reduction_db.max(gr);
        self.total_reduction += gr as f64;
        self.frames += 1;
        let g = gain(self.config.compressor_makeup_db - gr);
        (l * g, r * g)
    }
    pub fn average_reduction_db(&self) -> f64 {
        self.total_reduction / self.frames.max(1) as f64
    }
}

/// First-order antiderivative anti-aliasing. Unlike the old stage, drive does not
/// secretly multiply the small-signal gain. Identical delays on L/R preserve image.
#[derive(Default)]
struct Saturator {
    previous: f64,
}
impl Saturator {
    fn log_cosh(x: f64) -> f64 {
        let a = x.abs();
        a + (-2.0 * a).exp().ln_1p() - std::f64::consts::LN_2
    }
    fn process(&mut self, x: f32, amount: f32, mode: &str) -> f32 {
        if amount <= 0.0 {
            self.previous = x as f64;
            return x;
        }
        let drive = 10.0f64.powf(amount as f64 * 18.0 / 20.0);
        let now = x as f64;
        let last = self.previous;
        self.previous = now;
        let shaped = if mode == "soft_clip" {
            if (now - last).abs() > 1e-5 {
                (Self::log_cosh(drive * now) - Self::log_cosh(drive * last))
                    / (drive * drive * (now - last))
            } else {
                (drive * (now + last) * 0.5).tanh() / drive
            }
        } else if mode == "hard_clip" {
            (now * drive).clamp(-1.0, 1.0) / drive
        } else {
            let v = (now * drive + 1.0).rem_euclid(4.0);
            (if v <= 2.0 { v - 1.0 } else { 3.0 - v }) / drive
        };
        (now * (1.0 - amount as f64) + shaped * amount as f64) as f32
    }
}

pub struct StereoStrip {
    pub compressor: Compressor,
    eq: Vec<EqualizerBand>,
    hp: [Biquad; 2],
    lp: [Biquad; 2],
    sat: [Saturator; 2],
    config: Effects,
    sc_env: f32,
    sc_attack: f32,
    sc_release: f32,
    sr: f32,
    lowpass: Option<f32>,
    frame: u64,
}

/// Simple headphone-oriented interaural time difference for a moving stereo source.
/// This is a compact binaural cue, not a full HRTF renderer.
pub struct BinauralDelay {
    left: Vec<f32>,
    right: Vec<f32>,
    write: usize,
    max_delay_frames: f32,
}

impl BinauralDelay {
    pub fn new(sr: u32, max_delay_ms: f32) -> Self {
        let max_delay_frames = (sr as f32 * max_delay_ms.clamp(0.0, 1.2) / 1000.0).max(0.0);
        let capacity = max_delay_frames.ceil() as usize + 2;
        Self {
            left: vec![0.0; capacity],
            right: vec![0.0; capacity],
            write: 0,
            max_delay_frames,
        }
    }

    fn read(buffer: &[f32], write: usize, delay: f32) -> f32 {
        if delay <= 0.001 {
            return buffer[write];
        }
        let pos = (write as f64 - delay as f64).rem_euclid(buffer.len() as f64);
        let base = pos.floor();
        let i0 = (base as usize).min(buffer.len() - 1);
        let i1 = (i0 + 1) % buffer.len();
        let frac = (pos - base).clamp(0.0, 1.0) as f32;
        buffer[i0] + (buffer[i1] - buffer[i0]) * frac
    }

    pub fn process(&mut self, left: f32, right: f32, pan: f32) -> (f32, f32) {
        self.left[self.write] = left;
        self.right[self.write] = right;
        let pan = pan.clamp(-1.0, 1.0);
        let left_delay = pan.max(0.0) * self.max_delay_frames;
        let right_delay = (-pan).max(0.0) * self.max_delay_frames;
        let out = (
            Self::read(&self.left, self.write, left_delay),
            Self::read(&self.right, self.write, right_delay),
        );
        self.write = (self.write + 1) % self.left.len();
        out
    }
}

impl StereoStrip {
    pub fn new(e: &Effects, sr: u32) -> Self {
        let hp = e
            .highpass_hz
            .map(|hz| Biquad::highpass(sr as f32, hz))
            .unwrap_or(Biquad::identity());
        let lp = e
            .lowpass_hz
            .map(|hz| Biquad::design("lowpass", sr as f32, hz, 0.70710677, 0.0))
            .unwrap_or(Biquad::identity());
        Self {
            compressor: Compressor::new(e, sr),
            eq: e
                .eq
                .iter()
                .map(|c| EqualizerBand::new(c, sr as f32))
                .collect(),
            hp: [hp; 2],
            lp: [lp; 2],
            sat: std::array::from_fn(|_| Saturator::default()),
            config: e.clone(),
            sc_env: 0.0,
            sc_attack: (-1.0 / (sr as f32 * 0.001)).exp(),
            sc_release: (-1.0 / (sr as f32 * e.sidechain_release_ms.max(5.0) / 1000.0)).exp(),
            sr: sr as f32,
            lowpass: e.lowpass_hz,
            frame: 0,
        }
    }
    pub fn set_lowpass(&mut self, hz: f32) {
        self.lowpass = Some(hz);
    }
    pub fn process(&mut self, mut l: f32, mut r: f32, sidechain: Option<f32>) -> (f32, f32) {
        if self.frame.is_multiple_of(32)
            && let Some(hz) = self.lowpass
        {
            let c = Biquad::design("lowpass", self.sr, hz, 0.70710677, 0.0);
            for f in &mut self.lp {
                f.retune(c);
            }
        }
        self.frame += 1;
        l = self.lp[0].process(self.hp[0].process(l));
        r = self.lp[1].process(self.hp[1].process(r));
        l = self.sat[0].process(l, self.config.saturation, &self.config.saturation_mode);
        r = self.sat[1].process(r, self.config.saturation, &self.config.saturation_mode);
        for b in &mut self.eq {
            (l, r) = b.process(l, r);
        }
        if self.config.sidechain {
            let detector = sidechain.unwrap_or(0.0).abs();
            let c = if detector > self.sc_env {
                self.sc_attack
            } else {
                self.sc_release
            };
            self.sc_env = detector + c * (self.sc_env - detector);
            let normalized =
                (self.sc_env / gain(self.config.sidechain_threshold_db)).clamp(0.0, 1.0);
            let duck = 1.0 - self.config.sidechain_amount.clamp(0.0, 0.95) * normalized;
            l *= duck;
            r *= duck;
        }
        self.compressor.process(l, r)
    }
}

/// 4x polyphase FIR interpolation -> gain-compensated tanh -> FIR decimation.
/// 65-tap filters, 16 base-rate frames of total latency, compensated by the renderer.
pub struct FourXClipper {
    kernel: [f64; 65],
    input: [f32; 17],
    input_pos: usize,
    history: [f32; 65],
    pos: usize,
    drive: f32,
    enabled: bool,
}
impl FourXClipper {
    pub const LATENCY: usize = 16;
    pub fn new(drive_db: f32) -> Self {
        let mut kernel = [0.0; 65];
        let cutoff = 0.1125;
        for (i, h) in kernel.iter_mut().enumerate() {
            let x = i as f64 - 32.0;
            let sinc = if x == 0.0 {
                2.0 * cutoff
            } else {
                (TAU * cutoff * x).sin() / (PI * x)
            };
            let window = 0.42 - 0.5 * (TAU * i as f64 / 64.0).cos()
                + 0.08 * (2.0 * TAU * i as f64 / 64.0).cos();
            *h = sinc * window;
        }
        let sum: f64 = kernel.iter().sum();
        for h in &mut kernel {
            *h /= sum;
        }
        Self {
            kernel,
            input: [0.0; 17],
            input_pos: 0,
            history: [0.0; 65],
            pos: 0,
            drive: gain(drive_db),
            enabled: drive_db > 0.0,
        }
    }
    // Single instance processes one channel; stereo wrapper uses two instances.
    pub fn process(&mut self, x: f32) -> f32 {
        self.input[self.input_pos] = x;
        let mut out = 0.0;
        for phase in 0..4 {
            let mut up = 0.0f64;
            for k in 0..17 {
                let index = phase + k * 4;
                if index >= 65 {
                    break;
                }
                up += self.input[(self.input_pos + 17 - k) % 17] as f64 * self.kernel[index] * 4.0;
            }
            let shaped = if self.enabled {
                (up * self.drive as f64).tanh() / self.drive as f64
            } else {
                up
            };
            self.history[self.pos] = shaped as f32;
            if phase == 0 {
                out = 0.0;
                for k in 0..65 {
                    out += self.history[(self.pos + 65 - k) % 65] * self.kernel[k] as f32;
                }
            }
            self.pos = (self.pos + 1) % 65;
        }
        self.input_pos = (self.input_pos + 1) % 17;
        out
    }
}

#[derive(Serialize)]
pub struct MeterReport {
    pub id: String,
    pub peak_dbfs: f64,
    pub rms_dbfs: f64,
    pub crest_db: f64,
    pub dc_offset: f64,
    pub stereo_correlation: f64,
    pub side_mid_db: f64,
    pub band_center_hz: [f32; 6],
    pub band_rms_dbfs: [f64; 6],
    pub internal_overload_fraction: f64,
    pub compressor_max_reduction_db: f32,
    pub compressor_average_reduction_db: f64,
}
pub struct Meter {
    frames: u64,
    peak: f64,
    power: f64,
    dc: f64,
    ll: f64,
    rr: f64,
    lr: f64,
    mid: f64,
    side: f64,
    bands: [Biquad; 6],
    band_power: [f64; 6],
    over: u64,
}
impl Meter {
    pub fn new(sr: u32) -> Self {
        Self {
            frames: 0,
            peak: 0.0,
            power: 0.0,
            dc: 0.0,
            ll: 0.0,
            rr: 0.0,
            lr: 0.0,
            mid: 0.0,
            side: 0.0,
            bands: [40.0, 85.0, 220.0, 900.0, 4200.0, 12000.0]
                .map(|f| Biquad::design("bandpass", sr as f32, f, 0.9, 0.0)),
            band_power: [0.0; 6],
            over: 0,
        }
    }
    pub fn push(&mut self, l: f32, r: f32) {
        let (l, r) = (l as f64, r as f64);
        self.frames += 1;
        self.peak = self.peak.max(l.abs()).max(r.abs());
        self.power += (l * l + r * r) * 0.5;
        self.dc += (l + r) * 0.5;
        self.ll += l * l;
        self.rr += r * r;
        self.lr += l * r;
        let m = (l + r) * 0.5;
        let s = (l - r) * 0.5;
        self.mid += m * m;
        self.side += s * s;
        if l.abs() > 1.0 || r.abs() > 1.0 {
            self.over += 1;
        }
        for (i, f) in self.bands.iter_mut().enumerate() {
            let b = f.process(m as f32) as f64;
            self.band_power[i] += b * b;
        }
    }
    pub fn report(&self, id: &str, comp: Option<&Compressor>) -> MeterReport {
        let n = self.frames.max(1) as f64;
        let rms = (self.power / n).sqrt();
        MeterReport {
            id: id.into(),
            peak_dbfs: db(self.peak),
            rms_dbfs: db(rms),
            crest_db: db(self.peak) - db(rms),
            dc_offset: self.dc / n,
            stereo_correlation: self.lr / (self.ll * self.rr).sqrt().max(1e-20),
            side_mid_db: 10.0 * (self.side / self.mid.max(1e-20)).max(1e-20).log10(),
            band_center_hz: [40.0, 85.0, 220.0, 900.0, 4200.0, 12000.0],
            band_rms_dbfs: self.band_power.map(|p| db((p / n).sqrt())),
            internal_overload_fraction: self.over as f64 / n,
            compressor_max_reduction_db: comp.map_or(0.0, |c| c.max_reduction_db),
            compressor_average_reduction_db: comp.map_or(0.0, |c| c.average_reduction_db()),
        }
    }
}

/// 4x FIR reconstructed peak detector. Interpolation latency is eight source frames.
/// This is an engineering estimate, not a certified BS.1770 true-peak instrument.
pub struct ReconstructedPeak {
    history: [[f32; 17]; 2],
    kernel: [f64; 65],
    pos: usize,
}
impl ReconstructedPeak {
    pub const LATENCY: usize = 8;
    pub fn new() -> Self {
        let mut kernel = [0.0; 65];
        let cutoff = 0.1125;
        for (i, h) in kernel.iter_mut().enumerate() {
            let x = i as f64 - 32.0;
            let sinc = if x == 0.0 {
                2.0 * cutoff
            } else {
                (TAU * cutoff * x).sin() / (PI * x)
            };
            let window = 0.42 - 0.5 * (TAU * i as f64 / 64.0).cos()
                + 0.08 * (2.0 * TAU * i as f64 / 64.0).cos();
            *h = sinc * window;
        }
        let sum: f64 = kernel.iter().sum();
        for h in &mut kernel {
            *h /= sum;
        }
        Self {
            history: [[0.0; 17]; 2],
            kernel,
            pos: 0,
        }
    }
    pub fn push(&mut self, l: f32, r: f32) -> f32 {
        self.history[0][self.pos] = l;
        self.history[1][self.pos] = r;
        let mut peak = 0.0f64;
        for channel in 0..2 {
            for phase in 0..4 {
                let mut y = 0.0;
                for k in 0..17 {
                    let index = phase + 4 * k;
                    if index >= 65 {
                        break;
                    }
                    y += self.history[channel][(self.pos + 17 - k) % 17] as f64
                        * self.kernel[index]
                        * 4.0;
                }
                peak = peak.max(y.abs());
            }
        }
        self.pos = (self.pos + 1) % 17;
        peak as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn response(f: &Biquad, hz: f64, sr: f64) -> f64 {
        let w = TAU * hz / sr;
        let nr = f.b[0] + f.b[1] * w.cos() + f.b[2] * (2.0 * w).cos();
        let ni = -f.b[1] * w.sin() - f.b[2] * (2.0 * w).sin();
        let dr = 1.0 + f.a[0] * w.cos() + f.a[1] * (2.0 * w).cos();
        let di = -f.a[0] * w.sin() - f.a[1] * (2.0 * w).sin();
        10.0 * ((nr * nr + ni * ni) / (dr * dr + di * di)).log10()
    }
    #[test]
    fn shelf_and_bell_gains_match_requested_decibels_at_all_rates() {
        for sr in [44100.0, 48000.0, 96000.0, 128000.0] {
            for g in [-12.0, -6.0, 0.0, 6.0, 12.0] {
                let hi = Biquad::design("high_shelf", sr, 4000.0, 0.70710677, g);
                assert!(
                    response(&hi, 0.0, sr as f64).abs() < 1e-6,
                    "high shelf changed DC gain"
                );
                assert!((response(&hi, sr as f64 / 2.0, sr as f64) - g as f64).abs() < 1e-6);
                let lo = Biquad::design("low_shelf", sr, 120.0, 0.70710677, g);
                assert!((response(&lo, 0.0, sr as f64) - g as f64).abs() < 1e-6);
                assert!(response(&lo, sr as f64 / 2.0, sr as f64).abs() < 1e-6);
                let bell = Biquad::design("bell", sr, 2400.0, 2.0, g);
                assert!((response(&bell, 2400.0, sr as f64) - g as f64).abs() < 1e-5);
            }
        }
    }
    #[test]
    fn boost_and_inverse_cut_cancel() {
        let a = Biquad::design("bell", 48000.0, 850.0, 1.7, 9.0);
        let b = Biquad::design("bell", 48000.0, 850.0, 1.7, -9.0);
        for f in [20.0, 100.0, 440.0, 850.0, 4000.0, 18000.0] {
            assert!((response(&a, f, 48000.0) + response(&b, f, 48000.0)).abs() < 1e-8);
        }
    }
    fn tone(samples: &[f32], hz: f64, sr: f64) -> f64 {
        let mut a = 0.0;
        let mut b = 0.0;
        for (i, &x) in samples.iter().enumerate() {
            let p = TAU * hz * i as f64 / sr;
            a += x as f64 * p.cos();
            b += x as f64 * p.sin();
        }
        2.0 * (a * a + b * b).sqrt() / samples.len() as f64
    }
    #[test]
    fn dynamic_eq_reduces_the_target_band_without_turning_down_the_whole_mix() {
        let e:Effects=serde_json::from_value(serde_json::json!({"eq":[{"kind":"bell","frequency_hz":2000,"q":4,"gain_db":0,"dynamic_threshold_db":-24,"dynamic_ratio":8,"dynamic_range_db":6,"attack_ms":2,"release_ms":80}]})).unwrap();
        let mut strip = StereoStrip::new(&e, 48000);
        let mut y = Vec::new();
        for i in 0..96000 {
            let t = i as f64 / 48000.0;
            let x = (0.35 * (TAU * 2000.0 * t).sin() + 0.05 * (TAU * 8000.0 * t).sin()) as f32;
            let (l, _) = strip.process(x, x, None);
            if i >= 48000 {
                y.push(l);
            }
        }
        let target = db(tone(&y, 2000.0, 48000.0) / 0.35);
        let other = db(tone(&y, 8000.0, 48000.0) / 0.05);
        assert!(target < -5.0, "target reduction {target}");
        assert!(other.abs() < 0.6, "unrelated band changed {other}");
    }
    #[test]
    fn sidechain_only_ducks_in_response_to_audio() {
        let e = Effects {
            sidechain: true,
            sidechain_amount: 0.8,
            sidechain_release_ms: 80.0,
            ..Default::default()
        };
        let mut s = StereoStrip::new(&e, 48000);
        for _ in 0..4800 {
            assert!((s.process(0.2, 0.2, None).0 - 0.2).abs() < 1e-6);
        }
        let mut ducked = 0.2;
        for _ in 0..480 {
            ducked = s.process(0.2, 0.2, Some(0.8)).0;
        }
        assert!(ducked < 0.05);
        let mut recovered = 0.0;
        for _ in 0..48000 {
            recovered = s.process(0.2, 0.2, Some(0.0)).0;
        }
        assert!((recovered - 0.2).abs() < 1e-4);
    }
    #[test]
    fn oversampling_is_unity_at_small_signal_and_has_known_latency() {
        let mut clip = FourXClipper::new(0.0);
        let mut impulse = Vec::new();
        for i in 0..100 {
            impulse.push(clip.process(if i == 0 { 1.0 } else { 0.0 }));
        }
        let peak = impulse
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak, FourXClipper::LATENCY);
        let mut clip = FourXClipper::new(3.0);
        let mut y = Vec::new();
        for i in 0..50000 {
            let x = 0.001 * (TAU * 1000.0 * i as f64 / 48000.0).sin();
            let a = clip.process(x as f32);
            if i >= 2000 {
                y.push(a);
            }
        }
        assert!(db(tone(&y, 1000.0, 48000.0) / 0.001).abs() < 0.02);
    }
    #[test]
    fn saturation_has_no_hidden_low_level_gain() {
        let mut s = Saturator::default();
        let mut y = 0.0;
        for _ in 0..1000 {
            y = s.process(0.001, 0.8, "soft_clip");
        }
        assert!((y / 0.001 - 1.0).abs() < 0.001);
    }
}
