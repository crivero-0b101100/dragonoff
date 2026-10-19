// Copyright (C) 2026 OpenAI
// This program is free software under the GNU GPL v2 or (at your option) any later version.

use crate::model::{NoteSynthOverrides, Song, SoundSettings, SynthLayer, Track};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

const TABLE_BITS: usize = 12;
const TABLE_SIZE: usize = 1 << TABLE_BITS;

#[derive(serde::Serialize)]
pub struct RenderStats {
    pub frames: u64,
    pub sample_rate: u32,
    pub peak: f32,
    pub clipped_samples: u64,
    pub rms: f32,
    pub track_reports: Vec<crate::dsp::MeterReport>,
    pub bus_reports: Vec<crate::dsp::MeterReport>,
    pub premaster: crate::dsp::MeterReport,
    pub master: crate::dsp::MeterReport,
    pub limiter_max_reduction_db: f32,
    pub limiter_average_reduction_db: f64,
    pub reverb_return: crate::dsp::MeterReport,
}

struct DelayLine {
    left: Vec<f32>,
    right: Vec<f32>,
    pos: usize,
    feedback: f32,
}

struct Reverb {
    combs: Vec<(DampedComb, DampedComb)>,
    diffusion: Vec<Allpass>,
    highpass: [crate::dsp::Biquad; 2],
    lowpass: [crate::dsp::Biquad; 2],
    predelay: DelayLine,
}
struct DampedComb {
    buffer: Vec<f32>,
    pos: usize,
    feedback: f32,
    low: f32,
    alpha: f32,
}
impl DampedComb {
    fn new(sr: u32, seconds: f32, decay: f32, damping: f32) -> Self {
        Self {
            buffer: vec![0.0; (seconds * sr as f32).round() as usize],
            pos: 0,
            feedback: 10.0f32.powf(-3.0 * seconds / decay),
            low: 0.0,
            alpha: 1.0
                - (-std::f32::consts::TAU * damping.clamp(500.0, sr as f32 * 0.4) / sr as f32)
                    .exp(),
        }
    }
    fn process(&mut self, x: f32) -> f32 {
        let out = self.buffer[self.pos];
        self.low += self.alpha * (out - self.low);
        self.buffer[self.pos] = x + self.low * self.feedback;
        self.pos = (self.pos + 1) % self.buffer.len();
        out
    }
}

struct Allpass {
    left: Vec<f32>,
    right: Vec<f32>,
    pos: usize,
    gain: f32,
}

struct AudioSample {
    mono: Vec<f32>,
    side: Vec<f32>,
    sample_rate: u32,
}

struct LookaheadLimiter {
    lookahead_frames: usize,
    ceiling: f32,
    release_coeff: f32,
    attack_coeff: f32,
    gain: f32,
    max_reduction_db: f32,
    total_reduction_db: f64,
    reduction_frames: u64,
    detector: crate::dsp::ReconstructedPeak,
    index: u64,
    samples: VecDeque<(u64, f32, f32)>,
    peaks: VecDeque<(u64, f32)>,
}

impl LookaheadLimiter {
    fn new(sample_rate: u32, lookahead_ms: f32, ceiling_db: f32, release_ms: f32) -> Self {
        let lookahead_frames = (lookahead_ms.clamp(0.0, 20.0) * sample_rate as f32 / 1000.0).round()
            as usize
            + crate::dsp::ReconstructedPeak::LATENCY;
        Self {
            lookahead_frames,
            ceiling: 10.0f32.powf((ceiling_db.clamp(-12.0, 0.0) - 0.2) / 20.0),
            release_coeff: (-1.0 / (release_ms.clamp(10.0, 1000.0) * sample_rate as f32 / 1000.0))
                .exp(),
            // Six time constants fit inside the available lookahead. The
            // last sample guard remains available for pathological transients.
            attack_coeff: (-6.0
                / lookahead_frames
                    .saturating_sub(crate::dsp::ReconstructedPeak::LATENCY)
                    .max(1) as f32)
                .exp(),
            gain: 1.0,
            max_reduction_db: 0.0,
            total_reduction_db: 0.0,
            reduction_frames: 0,
            detector: crate::dsp::ReconstructedPeak::new(),
            index: 0,
            samples: VecDeque::with_capacity(lookahead_frames + 2),
            peaks: VecDeque::with_capacity(lookahead_frames + 2),
        }
    }

    fn process(&mut self, left: f32, right: f32) -> Option<(f32, f32)> {
        let index = self.index;
        self.index += 1;
        self.samples.push_back((index, left, right));
        let peak = self.detector.push(left, right);
        let peak_index = index.saturating_sub(crate::dsp::ReconstructedPeak::LATENCY as u64);
        while self.peaks.back().is_some_and(|(_, value)| *value <= peak) {
            self.peaks.pop_back();
        }
        self.peaks.push_back((peak_index, peak));
        let first_index = index.saturating_sub(self.lookahead_frames as u64);
        while self
            .peaks
            .front()
            .is_some_and(|(peak_index, _)| *peak_index < first_index)
        {
            self.peaks.pop_front();
        }
        let window_peak = self.peaks.front().map(|(_, p)| *p).unwrap_or(0.0);
        let target_gain = if window_peak > self.ceiling {
            self.ceiling / window_peak
        } else {
            1.0
        };
        if index < self.lookahead_frames as u64 {
            // No audio is being emitted yet, so pre-load protection without
            // creating a gain step on the first audible transient.
            self.gain = self.gain.min(target_gain);
            return None;
        }
        let (_, delayed_left, delayed_right) = self.samples.pop_front()?;
        if target_gain < self.gain {
            self.gain = target_gain + self.attack_coeff * (self.gain - target_gain);
        } else {
            let releasing = 1.0 + self.release_coeff * (self.gain - 1.0);
            self.gain = releasing.min(target_gain);
        }
        let sample_peak = delayed_left.abs().max(delayed_right.abs());
        if sample_peak * self.gain > self.ceiling {
            self.gain = self.ceiling / sample_peak;
        }
        let reduction = -20.0 * self.gain.max(1e-8).log10();
        self.max_reduction_db = self.max_reduction_db.max(reduction);
        self.total_reduction_db += reduction as f64;
        self.reduction_frames += 1;
        Some((delayed_left * self.gain, delayed_right * self.gain))
    }

    fn flush_frames(&self) -> usize {
        self.lookahead_frames
    }
}

impl Allpass {
    fn new(frames: usize, gain: f32) -> Self {
        let size = frames.max(1);
        Self {
            left: vec![0.0; size],
            right: vec![0.0; size],
            pos: 0,
            gain,
        }
    }

    fn process(&mut self, left: f32, right: f32) -> (f32, f32) {
        let dl = self.left[self.pos];
        let dr = self.right[self.pos];
        let yl = dl - self.gain * left;
        let yr = dr - self.gain * right;
        self.left[self.pos] = left + self.gain * yl;
        self.right[self.pos] = right + self.gain * yr;
        self.pos = (self.pos + 1) % self.left.len();
        (yl, yr)
    }
}

impl Reverb {
    fn new(sr: u32, settings: &crate::model::ReverbSettings) -> Self {
        let left = [
            0.0297, 0.0371, 0.0411, 0.0437, 0.0511, 0.0577, 0.0617, 0.0671,
        ];
        let right = [
            0.0311, 0.0359, 0.0427, 0.0479, 0.0527, 0.0593, 0.0637, 0.0713,
        ];
        Self {
            combs: left
                .into_iter()
                .zip(right)
                .map(|(l, r)| {
                    (
                        DampedComb::new(
                            sr,
                            l,
                            settings.decay_seconds.clamp(0.1, 12.0),
                            settings.damping_hz,
                        ),
                        DampedComb::new(
                            sr,
                            r,
                            settings.decay_seconds.clamp(0.1, 12.0),
                            settings.damping_hz,
                        ),
                    )
                })
                .collect(),
            diffusion: [0.0053, 0.0037, 0.0017, 0.0009]
                .into_iter()
                .map(|t| Allpass::new((sr as f32 * t).round() as usize, 0.5))
                .collect(),
            highpass: [crate::dsp::Biquad::highpass(sr as f32, settings.highpass_hz); 2],
            lowpass: [crate::dsp::Biquad::design(
                "lowpass",
                sr as f32,
                settings.lowpass_hz,
                0.70710677,
                0.0,
            ); 2],
            predelay: DelayLine::new(
                (sr as f32 * settings.predelay_ms.clamp(0.0, 250.0) / 1000.0).round() as usize,
                0.0,
            ),
        }
    }
    fn process(&mut self, l: f32, r: f32) -> (f32, f32) {
        let (l, r) = self
            .predelay
            .process(self.highpass[0].process(l), self.highpass[1].process(r));
        let (mut left, mut right) = (0.0, 0.0);
        for (cl, cr) in &mut self.combs {
            left += cl.process(l);
            right += cr.process(r);
        }
        left *= 0.125;
        right *= 0.125;
        for ap in &mut self.diffusion {
            (left, right) = ap.process(left, right);
        }
        (
            self.lowpass[0].process(left),
            self.lowpass[1].process(right),
        )
    }
}

struct Chorus {
    buffer: Vec<f32>,
    write: usize,
    phase: f32,
    sample_rate: f32,
    rate_hz: f32,
}

impl Chorus {
    fn new(sample_rate: u32, rate_hz: f32) -> Self {
        Self {
            buffer: vec![0.0; (sample_rate as usize / 20).max(2)],
            write: 0,
            phase: 0.0,
            sample_rate: sample_rate as f32,
            rate_hz: rate_hz.clamp(0.02, 8.0),
        }
    }

    fn process(&mut self, input: f32, depth: f32) -> (f32, f32) {
        let len = self.buffer.len();
        // A longer, wider pair of modulated taps gives mono synth sources a
        // real stereo difference signal, rather than the subtle double-mono
        // smear produced by two nearly identical 18 ms taps.
        let base = self.sample_rate * 0.023;
        let spread = self.sample_rate * 0.012 * depth.clamp(0.0, 1.0);
        let wave = self.phase.sin();
        let right_wave = (self.phase + std::f32::consts::FRAC_PI_2).sin();
        let left_delay = (base + spread * wave).clamp(1.0, (len - 2) as f32);
        let right_delay = (base + spread * right_wave).clamp(1.0, (len - 2) as f32);
        let read = |delay: f32| {
            let d = delay.floor() as usize;
            let f = delay - d as f32;
            let a = self.buffer[(self.write + len - d) % len];
            let b = self.buffer[(self.write + len - d - 1) % len];
            a + (b - a) * f
        };
        let left = read(left_delay);
        let right = read(right_delay);
        self.buffer[self.write] = input;
        self.write = (self.write + 1) % len;
        self.phase += std::f32::consts::TAU * self.rate_hz / self.sample_rate;
        if self.phase >= std::f32::consts::TAU {
            self.phase -= std::f32::consts::TAU;
        }
        (left, right)
    }
}

impl DelayLine {
    fn new(frames: usize, feedback: f32) -> Self {
        let size = frames.max(1);
        Self {
            left: vec![0.0; size],
            right: vec![0.0; size],
            pos: 0,
            feedback: feedback.clamp(0.0, 0.85),
        }
    }
    fn process(&mut self, l: f32, r: f32) -> (f32, f32) {
        let dl = self.left[self.pos];
        let dr = self.right[self.pos];
        self.left[self.pos] = l + dl * self.feedback;
        self.right[self.pos] = r + dr * self.feedback;
        self.pos = (self.pos + 1) % self.left.len();
        (dl, dr)
    }
}

pub fn render_wav(song: &Song, path: &Path) -> Result<RenderStats, String> {
    render_wav_options(song, path, None, false)
}

pub fn render_wav_options(
    song: &Song,
    path: &Path,
    stems_directory: Option<&Path>,
    master_bypass: bool,
) -> Result<RenderStats, String> {
    use crate::dsp::{BinauralDelay, Biquad, FourXClipper, Meter, StereoStrip, gain};
    if !song.bpm.is_finite() || !(60.0..=180.0).contains(&song.bpm) {
        return Err("bpm must be in 60..180".into());
    }
    if !(22050..=192000).contains(&song.sample_rate) || song.numerator == 0 || song.denominator == 0
    {
        return Err("invalid sample rate or meter".into());
    }
    let valid_id = |id: &str| {
        !id.is_empty()
            && id.len() <= 48
            && id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    };
    crate::model::validate_eq(&song.master.eq)?;
    for track in &song.tracks {
        if !valid_id(&track.id)
            || !["auto", "synth", "one_shot"].contains(&track.source_type.as_str())
        {
            return Err(format!("invalid track id/source type: {}", track.id));
        }
        crate::model::validate_sound(&track.sound)?;
        crate::model::validate_drum(&track.drum)?;
        crate::model::validate_effects(&track.effects)?;
        if track.source_type == "one_shot"
            && track.sample_id.is_none()
            && track.sample_alt_id.is_none()
            && track.sample_path.is_none()
            && track.sample_alt_path.is_none()
        {
            return Err(format!("one_shot track {} requires a sample", track.id));
        }
    }
    for bus in &song.buses {
        if !valid_id(&bus.id) {
            return Err(format!("invalid bus id: {}", bus.id));
        }
        crate::model::validate_effects(&bus.effects)?;
        if bus.effects.delay_ms > 0.0
            || bus.effects.delay_beats.is_some()
            || bus.effects.delay_wet > 0.0
            || bus.effects.reverb_send > 0.0
            || bus.effects.chorus_depth > 0.0
        {
            return Err(format!(
                "unsupported bus time effect on {}; use an independent track",
                bus.id
            ));
        }
    }
    let mut bus_indices = HashMap::new();
    for (index, bus) in song.buses.iter().enumerate() {
        if bus_indices.insert(bus.id.clone(), index).is_some() {
            return Err(format!("duplicate bus {}", bus.id));
        }
    }
    let mut ids = std::collections::HashSet::new();
    for track in &song.tracks {
        if !ids.insert(track.id.clone()) {
            return Err(format!("duplicate track {}", track.id));
        }
        if let Some(bus) = &track.bus_id
            && !bus_indices.contains_key(bus)
        {
            return Err(format!("track {} routes to missing bus {bus}", track.id));
        }
    }
    let samples = load_samples(song)?;
    let table = make_tables();
    let sr = song.sample_rate;
    let spq = 60.0 / song.bpm;
    let bar_seconds = spq * song.numerator as f64 * 4.0 / song.denominator as f64;
    let frames_per_bar = (bar_seconds * sr as f64).round() as usize;
    let bars = song.total_bars().max(1);
    let tail_frames: Vec<usize> = song
        .tracks
        .iter()
        .map(|track| {
            let gate = track
                .patterns
                .iter()
                .flat_map(|p| &p.notes)
                .map(|n| n.length_beats * 4.0 / song.denominator as f64 * spq)
                .fold(0.0f64, f64::max);
            let sample_seconds = [track.sample_id.as_ref(), track.sample_alt_id.as_ref()]
                .into_iter()
                .flatten()
                .filter_map(|id| samples.get(id))
                .map(|s| {
                    s.mono.len() as f64 / s.sample_rate as f64
                        * 2.0f64.powf(-track.sample_tune_semitones as f64 / 12.0)
                })
                .fold(0.0f64, f64::max);
            let max_sample = if track.drum.decay_ms > 0.0 {
                sample_seconds.min(track.drum.decay_ms as f64 / 1000.0)
            } else {
                sample_seconds
            };
            let seconds = (gate
                + (track.sound.release_ms as f64 / 1000.0).max(
                    if track.source_type != "synth" && track.role.starts_with("sf2_") {
                        0.22
                    } else {
                        0.0
                    },
                ))
            .max(max_sample)
            .max(
                if matches!(track.role.as_str(), "piano" | "guitar" | "steel_guitar") {
                    2.3
                } else {
                    0.10
                },
            );
            (seconds * sr as f64).ceil() as usize
        })
        .collect();
    let mut tails: Vec<Vec<f32>> = tail_frames.iter().map(|&n| vec![0.0; n]).collect();
    let mut tails_side = tails.clone();
    let mut strips: Vec<StereoStrip> = song
        .tracks
        .iter()
        .map(|t| StereoStrip::new(&t.effects, sr))
        .collect();
    let mut binaural_delays: Vec<Option<BinauralDelay>> = song
        .tracks
        .iter()
        .map(|t| {
            (t.sound.binaural_itd_ms > 0.0).then(|| BinauralDelay::new(sr, t.sound.binaural_itd_ms))
        })
        .collect();
    let mut spatial_phases: Vec<f64> = song
        .tracks
        .iter()
        .map(|t| t.sound.stereo_motion_phase_radians as f64)
        .collect();
    let mut bus_strips: Vec<StereoStrip> = song
        .buses
        .iter()
        .map(|b| StereoStrip::new(&b.effects, sr))
        .collect();
    let mut choruses: Vec<Option<Chorus>> = song
        .tracks
        .iter()
        .map(|t| {
            if t.effects.chorus_depth > 0.0
                || t.automation.iter().any(|p| p.parameter == "chorus_depth")
            {
                Some(Chorus::new(sr, t.effects.chorus_rate_hz))
            } else {
                None
            }
        })
        .collect();
    let mut delays: Vec<Option<DelayLine>> = song
        .tracks
        .iter()
        .map(|t| {
            let ms = t
                .effects
                .delay_beats
                .map(|b| b * 60000.0 / song.bpm as f32)
                .unwrap_or(t.effects.delay_ms);
            if ms > 0.0 {
                Some(DelayLine::new(
                    (ms * sr as f32 / 1000.0).round() as usize,
                    t.effects.delay_feedback,
                ))
            } else {
                None
            }
        })
        .collect();
    let mut reverb = Reverb::new(sr, &song.reverb);
    let mut return_meter = Meter::new(sr);
    let mut meters: Vec<Meter> = song.tracks.iter().map(|_| Meter::new(sr)).collect();
    let mut bus_meters: Vec<Meter> = song.buses.iter().map(|_| Meter::new(sr)).collect();
    let mut pre_meter = Meter::new(sr);
    let mut master_meter = Meter::new(sr);
    let mut final_wav = WavWriter::new(path, sr, master_bypass)?;
    let mut track_writers = Vec::new();
    let mut bus_writers = Vec::new();
    let mut pre_writer = None;
    let mut return_writer = None;
    if let Some(dir) = stems_directory {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        for t in &song.tracks {
            track_writers.push(WavWriter::new(
                &dir.join(format!("track-{}.wav", safe_id(&t.id))),
                sr,
                true,
            )?);
        }
        for b in &song.buses {
            bus_writers.push(WavWriter::new(
                &dir.join(format!("bus-{}.wav", safe_id(&b.id))),
                sr,
                true,
            )?);
        }
        pre_writer = Some(WavWriter::new(&dir.join("premaster.wav"), sr, true)?);
        return_writer = Some(WavWriter::new(&dir.join("return-room.wav"), sr, true)?);
    }
    let default_kick = song
        .tracks
        .iter()
        .find(|t| t.role == "kick")
        .map(|t| t.id.clone());
    let sidechain_ids: Vec<Option<String>> = song
        .tracks
        .iter()
        .map(|t| {
            if t.effects.sidechain {
                t.effects.sidechain_source.clone().or(default_kick.clone())
            } else {
                None
            }
        })
        .collect();
    let bus_sc: Vec<Option<String>> = song
        .buses
        .iter()
        .map(|b| {
            if b.effects.sidechain {
                b.effects.sidechain_source.clone().or(default_kick.clone())
            } else {
                None
            }
        })
        .collect();
    let donor_ids: std::collections::HashSet<String> = sidechain_ids
        .iter()
        .chain(&bus_sc)
        .flatten()
        .cloned()
        .collect();
    for id in &donor_ids {
        if !ids.contains(id) {
            return Err(format!("missing sidechain source {id}"));
        }
    }
    let m = &song.master;
    let mut master_config = crate::model::Effects {
        compressor_threshold_db: m.compressor_threshold_db,
        compressor_ratio: m.compressor_ratio,
        compressor_attack_ms: m.compressor_attack_ms,
        compressor_release_ms: m.compressor_release_ms,
        compressor_knee_db: m.compressor_knee_db,
        compressor_sidechain_highpass_hz: m.compressor_sidechain_highpass_hz,
        ..Default::default()
    };
    let master_bands = [
        ("highpass", m.eq_highpass_hz, 0.70710677, 0.0),
        ("low_shelf", 110.0, 0.70710677, m.eq_low_shelf_db),
        ("bell", m.eq_low_mid_hz, m.eq_low_mid_q, m.eq_low_mid_db),
        ("bell", m.eq_presence_hz, m.eq_presence_q, m.eq_presence_db),
        ("high_shelf", 11000.0, 0.70710677, m.eq_high_shelf_db),
    ];
    for (kind, hz, q, g) in master_bands {
        master_config.eq.push(crate::model::EqBand {
            kind: kind.into(),
            frequency_hz: hz,
            q,
            gain_db: g,
            dynamic_threshold_db: None,
            dynamic_ratio: 2.0,
            dynamic_range_db: 0.0,
            attack_ms: 10.0,
            release_ms: 120.0,
        });
    }
    master_config.eq.extend(m.eq.clone());
    let mut master_strip = StereoStrip::new(&master_config, sr);
    let mut side_low = Biquad::design("lowpass", sr as f32, 120.0, 0.70710677, 0.0);
    let mut shapers = [
        FourXClipper::new(m.clipper_drive_db),
        FourXClipper::new(m.clipper_drive_db),
    ];
    let mut limiter = LookaheadLimiter::new(
        sr,
        m.limiter_lookahead_ms,
        m.limiter_ceiling_db,
        m.limiter_release_ms,
    );
    let mut shaper_skip = if master_bypass {
        0
    } else {
        FourXClipper::LATENCY
    };
    for bar0 in 0..bars {
        let mut direct_l = vec![0.0; frames_per_bar];
        let mut direct_r = vec![0.0; frames_per_bar];
        let mut bus_audio: Vec<(Vec<f32>, Vec<f32>)> = song
            .buses
            .iter()
            .map(|_| (vec![0.0; frames_per_bar], vec![0.0; frames_per_bar]))
            .collect();
        let mut sends_l = vec![0.0; frames_per_bar];
        let mut sends_r = vec![0.0; frames_per_bar];
        // Only sidechain sources are cached. Track processing order never changes ducking.
        let mut donors = HashMap::new();
        for (ti, t) in song
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| donor_ids.contains(&t.id))
        {
            let source = render_source(
                t,
                bar0,
                song,
                frames_per_bar,
                &table,
                &samples,
                &mut tails[ti],
                &mut tails_side[ti],
            );
            donors.insert(t.id.clone(), source);
        }
        for (ti, t) in song.tracks.iter().enumerate() {
            let owned;
            let (mid, side) = if let Some(source) = donors.get(&t.id) {
                (&source.0, &source.1)
            } else {
                owned = render_source(
                    t,
                    bar0,
                    song,
                    frames_per_bar,
                    &table,
                    &samples,
                    &mut tails[ti],
                    &mut tails_side[ti],
                );
                (&owned.0, &owned.1)
            };
            let donor = sidechain_ids[ti].as_ref().and_then(|id| donors.get(id));
            let start_beat = bar0 as f64 * song.numerator as f64;
            let value = |name: &str, fallback: f32, beat: f64| {
                automation_at(t, name, beat, fallback as f64, song.numerator) as f32
            };
            let g0 = value("gain_db", t.gain_db, start_beat);
            let g1 = value("gain_db", t.gain_db, start_beat + song.numerator as f64);
            let lp0 = value(
                "lowpass_hz",
                t.effects.lowpass_hz.unwrap_or(20000.0),
                start_beat,
            );
            let lp1 = value(
                "lowpass_hz",
                t.effects.lowpass_hz.unwrap_or(20000.0),
                start_beat + song.numerator as f64,
            );
            let rs0 = value("reverb_send", t.effects.reverb_send, start_beat);
            let rs1 = value(
                "reverb_send",
                t.effects.reverb_send,
                start_beat + song.numerator as f64,
            );
            let dw0 = value("delay_wet", t.effects.delay_wet, start_beat);
            let dw1 = value(
                "delay_wet",
                t.effects.delay_wet,
                start_beat + song.numerator as f64,
            );
            let df0 = value("delay_feedback", t.effects.delay_feedback, start_beat);
            let df1 = value(
                "delay_feedback",
                t.effects.delay_feedback,
                start_beat + song.numerator as f64,
            );
            let ch0 = value("chorus_depth", t.effects.chorus_depth, start_beat);
            let ch1 = value(
                "chorus_depth",
                t.effects.chorus_depth,
                start_beat + song.numerator as f64,
            );
            let motion_depth0 = value(
                "stereo_motion_depth",
                t.sound.stereo_motion_depth,
                start_beat,
            );
            let motion_depth1 = value(
                "stereo_motion_depth",
                t.sound.stereo_motion_depth,
                start_beat + song.numerator as f64,
            );
            let motion_rate0 = value(
                "stereo_motion_rate_hz",
                t.sound.stereo_motion_rate_hz,
                start_beat,
            );
            let motion_rate1 = value(
                "stereo_motion_rate_hz",
                t.sound.stereo_motion_rate_hz,
                start_beat + song.numerator as f64,
            );
            let bus = t
                .bus_id
                .as_ref()
                .and_then(|id| bus_indices.get(id))
                .copied();
            for i in 0..frames_per_bar {
                let u = i as f32 / frames_per_bar as f32;
                let lerp = |a: f32, b: f32| a + (b - a) * u;
                if t.effects.lowpass_hz.is_some()
                    || t.automation.iter().any(|p| p.parameter == "lowpass_hz")
                {
                    strips[ti].set_lowpass(lerp(lp0, lp1));
                }
                let detector = donor.map(|(m, s)| (m[i] + s[i]).abs().max((m[i] - s[i]).abs()));
                let (mut l, mut r) =
                    strips[ti].process(mid[i] + side[i], mid[i] - side[i], detector);
                let g = if t.muted {
                    0.0
                } else {
                    gain(lerp(g0, g1).clamp(-60.0, 12.0))
                };
                let motion_depth = lerp(motion_depth0, motion_depth1).clamp(0.0, 1.0);
                let motion_rate = lerp(motion_rate0, motion_rate1).clamp(0.0, 8.0);
                let pan = (t.pan + spatial_phases[ti].sin() as f32 * motion_depth).clamp(-1.0, 1.0);
                let pl = (1.0 - pan.max(0.0)).sqrt();
                let pr = (1.0 + pan.min(0.0)).sqrt();
                spatial_phases[ti] = (spatial_phases[ti]
                    + std::f64::consts::TAU * motion_rate as f64 / sr as f64)
                    .rem_euclid(std::f64::consts::TAU);
                l *= g * pl;
                r *= g * pr;
                if let Some(spatializer) = binaural_delays[ti].as_mut() {
                    (l, r) = spatializer.process(l, r, pan);
                }
                if let Some(delay) = delays[ti].as_mut() {
                    delay.feedback = lerp(df0, df1).clamp(0.0, 0.85);
                    let (dl, dr) = delay.process(l, r);
                    let wet = lerp(dw0, dw1).clamp(0.0, 0.8);
                    l += dl * wet;
                    r += dr * wet;
                }
                if let Some(chorus) = choruses[ti].as_mut() {
                    let depth = lerp(ch0, ch1).clamp(0.0, 1.0);
                    let (cl, cr) = chorus.process((l + r) * 0.5, depth);
                    let wet = depth * 0.4;
                    l = l * (1.0 - wet) + cl * wet;
                    r = r * (1.0 - wet) + cr * wet;
                }
                meters[ti].push(l, r);
                if let Some(w) = track_writers.get_mut(ti) {
                    w.push(l, r)?;
                }
                let send = lerp(rs0, rs1).clamp(0.0, 0.8);
                sends_l[i] += l * send;
                sends_r[i] += r * send;
                if let Some(index) = bus {
                    bus_audio[index].0[i] += l;
                    bus_audio[index].1[i] += r;
                } else {
                    direct_l[i] += l;
                    direct_r[i] += r;
                }
            }
        }
        for (bi, bus) in song.buses.iter().enumerate() {
            let detector = bus_sc[bi].as_ref().and_then(|id| donors.get(id));
            for i in 0..frames_per_bar {
                let (l, r) = bus_strips[bi].process(
                    bus_audio[bi].0[i],
                    bus_audio[bi].1[i],
                    detector.map(|(m, s)| (m[i] + s[i]).abs().max((m[i] - s[i]).abs())),
                );
                let mid = (l + r) * 0.5;
                let side = (l - r) * 0.5 * bus.stereo_width.clamp(0.0, 2.0);
                let g = gain(bus.gain_db.clamp(-60.0, 12.0));
                let (l, r) = ((mid + side) * g, (mid - side) * g);
                direct_l[i] += l;
                direct_r[i] += r;
                bus_meters[bi].push(l, r);
                if let Some(w) = bus_writers.get_mut(bi) {
                    w.push(l, r)?;
                }
            }
        }
        for i in 0..frames_per_bar {
            let (rl, rr) = reverb.process(sends_l[i], sends_r[i]);
            let rg = gain(song.reverb.return_gain_db);
            let (rl, rr) = (rl * rg, rr * rg);
            return_meter.push(rl, rr);
            if let Some(w) = return_writer.as_mut() {
                w.push(rl, rr)?;
            }
            let (mut l, mut r) = (direct_l[i] + rl, direct_r[i] + rr);
            pre_meter.push(l, r);
            if let Some(w) = pre_writer.as_mut() {
                w.push(l, r)?;
            }
            if master_bypass {
                final_wav.push(l, r)?;
                master_meter.push(l, r);
                continue;
            }
            let mid = (l + r) * 0.5;
            let side = (l - r) * 0.5;
            let low = side_low.process(side);
            let wide = low * m.low_band_width.clamp(0.0, 1.0)
                + (side - low) * m.stereo_width.clamp(0.0, 2.0);
            (l, r) = master_strip.process(mid + wide, mid - wide, None);
            let g = gain(m.output_gain_db.clamp(-24.0, 18.0));
            if let Some((l, r)) =
                limiter.process(shapers[0].process(l) * g, shapers[1].process(r) * g)
            {
                if shaper_skip > 0 {
                    shaper_skip -= 1;
                } else {
                    final_wav.push(l, r)?;
                    master_meter.push(l, r);
                }
            }
        }
    }
    if !master_bypass {
        for _ in 0..FourXClipper::LATENCY + limiter.flush_frames() {
            if let Some((l, r)) = limiter.process(
                shapers[0].process(0.0) * gain(m.output_gain_db),
                shapers[1].process(0.0) * gain(m.output_gain_db),
            ) {
                if shaper_skip > 0 {
                    shaper_skip -= 1;
                } else {
                    final_wav.push(l, r)?;
                    master_meter.push(l, r);
                }
            }
        }
    }
    let frames = final_wav.frames;
    final_wav.finish()?;
    for w in track_writers
        .into_iter()
        .chain(bus_writers)
        .chain(pre_writer)
        .chain(return_writer)
    {
        w.finish()?;
    }
    let track_reports = meters
        .iter()
        .enumerate()
        .map(|(i, m)| m.report(&song.tracks[i].id, Some(&strips[i].compressor)))
        .collect();
    let bus_reports = bus_meters
        .iter()
        .enumerate()
        .map(|(i, m)| m.report(&song.buses[i].id, Some(&bus_strips[i].compressor)))
        .collect();
    let master_report = master_meter.report("master", Some(&master_strip.compressor));
    Ok(RenderStats {
        frames,
        sample_rate: sr,
        peak: gain(master_report.peak_dbfs as f32),
        rms: gain(master_report.rms_dbfs as f32),
        clipped_samples: (master_report.internal_overload_fraction * frames as f64).round() as u64,
        track_reports,
        bus_reports,
        premaster: pre_meter.report("premaster", None),
        master: master_report,
        limiter_max_reduction_db: limiter.max_reduction_db,
        limiter_average_reduction_db: limiter.total_reduction_db
            / limiter.reduction_frames.max(1) as f64,
        reverb_return: return_meter.report("room-return", None),
    })
}

fn safe_id(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[derive(Clone)]
struct SynthNoteEvent {
    start: usize,
    gate: usize,
    pitch: u8,
    velocity: u8,
    synth_overrides: Option<NoteSynthOverrides>,
}

#[allow(clippy::too_many_arguments)] // Explicit DSP buffers and musical-time context.
fn render_source(
    track: &Track,
    bar0: u32,
    song: &Song,
    frames_per_bar: usize,
    table: &WavetableBank,
    samples: &HashMap<String, AudioSample>,
    tail: &mut [f32],
    tail_side: &mut [f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut mid = vec![0.0; frames_per_bar + tail.len()];
    let mut side = vec![0.0; mid.len()];
    mid[..tail.len()].copy_from_slice(tail);
    side[..tail_side.len()].copy_from_slice(tail_side);
    let mut rng = track
        .id
        .bytes()
        .fold(
            song.seed.max(1) ^ (bar0 as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15),
            |h, b| (h ^ b as u64).wrapping_mul(1099511628211),
        )
        .max(1);
    let mut synth_events = Vec::new();
    if !track.muted {
        if track.source_type == "synth"
            || track.source_type == "one_shot"
            || !is_drum_role(&track.role)
        {
            for clip in &track.clips {
                if bar0 + 1 < clip.start_bar || bar0 + 1 >= clip.start_bar.saturating_add(clip.bars)
                {
                    continue;
                }
                if let Some(pattern) = track.patterns.iter().find(|p| p.id == clip.pattern_id) {
                    let pb = (bar0 + 1 - clip.start_bar) % pattern.bars.max(1) + 1;
                    for (event_index, note) in
                        pattern.notes.iter().filter(|n| n.bar == pb).enumerate()
                    {
                        let units = 4.0 / song.denominator as f64;
                        let start = (((note.beat - 1.0) * units + note.tick as f64 / 480.0) * 60.0
                            / song.bpm
                            * song.sample_rate as f64)
                            .round() as usize;
                        if start >= frames_per_bar {
                            continue;
                        }
                        let duration = (note.length_beats.max(0.001) * units * 60.0 / song.bpm
                            * song.sample_rate as f64)
                            .round() as usize;
                        let pitch = (note.pitch as i16 + clip.transpose).clamp(0, 127) as u8;
                        if (track.source_type == "one_shot"
                            || (track.source_type == "auto" && is_drum_role(&track.role)))
                            && fire_sample(
                                track,
                                samples,
                                start,
                                song.sample_rate,
                                (note.velocity as f32 / 100.0).clamp(0.0, 1.27),
                                bar0 as usize * 4096 + event_index,
                                &mut mid,
                                &mut rng,
                                &mut side,
                            )
                        {
                            continue;
                        }
                        if track.source_type != "synth"
                            && matches!(
                                track.role.as_str(),
                                "piano" | "flute" | "guitar" | "steel_guitar"
                            )
                            && fire_instrument_sample(
                                &track.role,
                                pitch,
                                note.velocity,
                                start,
                                duration,
                                track.sample_tune_semitones,
                                song.sample_rate,
                                samples,
                                &mut mid,
                                &mut side,
                            )
                        {
                            continue;
                        }
                        if track.source_type != "synth"
                            && track.role.starts_with("sf2_")
                            && fire_sf2_sample(
                                track,
                                pitch,
                                note.velocity,
                                start,
                                duration,
                                track.sample_tune_semitones,
                                song.sample_rate,
                                samples,
                                &mut mid,
                                &mut side,
                            )
                        {
                            continue;
                        }
                        synth_events.push(SynthNoteEvent {
                            start,
                            gate: duration,
                            pitch,
                            velocity: note.velocity,
                            synth_overrides: note.synth_overrides.clone(),
                        });
                    }
                }
            }
        }
        render_synth_events(
            track,
            &track.sound,
            &mut synth_events,
            bar0,
            song.numerator,
            frames_per_bar,
            song.sample_rate,
            table,
            &mut mid,
            &mut rng,
            &mut side,
        );
    }
    let len = tail.len();
    tail.copy_from_slice(&mid[frames_per_bar..frames_per_bar + len]);
    tail_side.copy_from_slice(&side[frames_per_bar..frames_per_bar + len]);
    (mid, side)
}

#[allow(clippy::too_many_arguments)] // Explicit DSP buffers and musical-time context.
fn render_synth_events(
    track: &Track,
    sound: &SoundSettings,
    events: &mut Vec<SynthNoteEvent>,
    bar0: u32,
    numerator: u8,
    frames_per_bar: usize,
    sr: u32,
    table: &WavetableBank,
    out: &mut [f32],
    rng: &mut u64,
    out_side: &mut [f32],
) {
    if events.is_empty() {
        return;
    }
    events.sort_by_key(|event| event.start);
    let mut merged = Vec::<SynthNoteEvent>::with_capacity(events.len());
    for event in events.drain(..) {
        if sound.legato
            && let Some(previous) = merged.last_mut()
            && previous.pitch == event.pitch
            && previous.synth_overrides == event.synth_overrides
            && event.start <= previous.start.saturating_add(previous.gate)
        {
            let end = event.start.saturating_add(event.gate);
            previous.gate = previous
                .start
                .saturating_add(previous.gate)
                .max(end)
                .saturating_sub(previous.start);
            previous.velocity = ((previous.velocity as u16 + event.velocity as u16) / 2) as u8;
        } else {
            merged.push(event);
        }
    }
    for event in merged {
        let event_beat = bar0 as f64 * numerator as f64
            + event.start as f64 / frames_per_bar.max(1) as f64 * numerator as f64;
        let mut note_sound = sound.clone();
        note_sound.pitch_envelope_semitones = automation_at(
            track,
            "pitch_envelope_semitones",
            event_beat,
            sound.pitch_envelope_semitones as f64,
            numerator,
        )
        .clamp(-24.0, 48.0) as f32;
        note_sound.lfo_depth = automation_at(
            track,
            "lfo_depth",
            event_beat,
            sound.lfo_depth as f64,
            numerator,
        )
        .clamp(0.0, 4.0) as f32;
        note_sound.lfo_bias = automation_at(
            track,
            "lfo_bias",
            event_beat,
            sound.lfo_bias as f64,
            numerator,
        )
        .clamp(-1.0, 1.0) as f32;
        note_sound.lfo_rate_hz = automation_at(
            track,
            "lfo_rate_hz",
            event_beat,
            sound.lfo_rate_hz as f64,
            numerator,
        )
        .clamp(0.0, 30.0) as f32;
        if let Some(overrides) = event.synth_overrides {
            if let Some(waveform) = overrides.waveform {
                note_sound.waveform = waveform.clone();
                for layer in &mut note_sound.layers {
                    layer.waveform = waveform.clone();
                }
            }
            if let Some(value) = overrides.pitch_envelope_semitones {
                note_sound.pitch_envelope_semitones = value;
            }
            if let Some(value) = overrides.pitch_decay_ms {
                note_sound.pitch_decay_ms = value;
            }
            if let Some(value) = overrides.attack_ms {
                note_sound.attack_ms = value;
            }
            if let Some(value) = overrides.decay_ms {
                note_sound.decay_ms = value;
            }
            if let Some(value) = overrides.sustain {
                note_sound.sustain = value;
            }
            if let Some(value) = overrides.release_ms {
                note_sound.release_ms = value;
            }
            if let Some(value) = overrides.cutoff_hz {
                note_sound.cutoff_hz = value;
                for layer in &mut note_sound.layers {
                    layer.cutoff_hz = Some(value);
                }
            }
            if let Some(value) = overrides.filter_env_amount {
                note_sound.filter_env_amount = value;
            }
            if let Some(value) = overrides.noise_amount {
                if note_sound.layers.is_empty() {
                    let mut layer = SynthLayer {
                        waveform: note_sound.waveform.clone(),
                        unison: note_sound.unison,
                        detune_cents: note_sound.detune_cents,
                        ..SynthLayer::default()
                    };
                    layer.noise_amount = value;
                    if overrides.cutoff_hz.is_some() {
                        layer.cutoff_hz = overrides.cutoff_hz;
                    }
                    note_sound.layers.push(layer);
                } else if let Some(layer) = note_sound
                    .layers
                    .iter_mut()
                    .rev()
                    .find(|layer| layer.waveform == "noise")
                {
                    layer.noise_amount = value;
                } else if let Some(layer) = note_sound.layers.last_mut() {
                    layer.noise_amount = value;
                }
            }
        }
        synth_note(
            track,
            &note_sound,
            event.pitch,
            event.velocity,
            event.start,
            event.gate,
            bar0,
            frames_per_bar,
            sr,
            table,
            out,
            rng,
            out_side,
        );
    }
}

struct WavWriter {
    final_path: std::path::PathBuf,
    temporary_path: std::path::PathBuf,
    committed: bool,
    out: BufWriter<File>,
    frames: u64,
    sr: u32,
    float: bool,
    rng: u64,
}
impl WavWriter {
    fn new(path: &Path, sr: u32, float: bool) -> Result<Self, String> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
        }
        let temporary_path = path.with_file_name(format!(
            ".{}.{}.partial",
            path.file_name()
                .ok_or("output filename missing")?
                .to_string_lossy(),
            std::process::id()
        ));
        let mut writer = Self {
            final_path: path.to_path_buf(),
            temporary_path: temporary_path.clone(),
            committed: false,
            out: BufWriter::with_capacity(
                128 * 1024,
                File::create(&temporary_path).map_err(|e| e.to_string())?,
            ),
            frames: 0,
            sr,
            float,
            rng: 0x5126feaabb,
        };
        writer.header(0)?;
        Ok(writer)
    }
    fn header(&mut self, bytes: u32) -> Result<(), String> {
        let bits = if self.float { 32u16 } else { 24 };
        let block = bits / 8 * 2;
        for b in [
            b"RIFF".as_slice(),
            &(36u32.saturating_add(bytes)).to_le_bytes(),
            b"WAVEfmt ",
            &16u32.to_le_bytes(),
            &(if self.float { 3u16 } else { 1u16 }).to_le_bytes(),
            &2u16.to_le_bytes(),
            &self.sr.to_le_bytes(),
            &(self.sr * block as u32).to_le_bytes(),
            &block.to_le_bytes(),
            &bits.to_le_bytes(),
            b"data",
            &bytes.to_le_bytes(),
        ] {
            self.out.write_all(b).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    fn push(&mut self, l: f32, r: f32) -> Result<(), String> {
        if !l.is_finite() || !r.is_finite() {
            return Err(format!("non-finite audio at frame {}", self.frames));
        }
        for x in [l, r] {
            if self.float {
                self.out
                    .write_all(&x.to_le_bytes())
                    .map_err(|e| e.to_string())?;
            } else {
                let tpdf = (next_noise(&mut self.rng) - next_noise(&mut self.rng)) * 0.5;
                let q = (x.clamp(-1.0, 1.0) * 8388608.0 + tpdf)
                    .round()
                    .clamp(-8388608.0, 8388607.0) as i32;
                self.out
                    .write_all(&q.to_le_bytes()[..3])
                    .map_err(|e| e.to_string())?;
            }
        }
        self.frames += 1;
        Ok(())
    }
    fn finish(mut self) -> Result<(), String> {
        let bytes = self.frames * if self.float { 8 } else { 6 };
        if bytes > u32::MAX as u64 - 36 {
            return Err("WAV exceeds RIFF size limit".into());
        }
        self.out.flush().map_err(|e| e.to_string())?;
        self.out
            .seek(SeekFrom::Start(0))
            .map_err(|e| e.to_string())?;
        self.header(bytes as u32)?;
        self.out.flush().map_err(|e| e.to_string())?;
        std::fs::rename(&self.temporary_path, &self.final_path).map_err(|e| e.to_string())?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for WavWriter {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_file(&self.temporary_path);
        }
    }
}

const FORMANT_MIPS: [usize; 8] = [1, 2, 3, 5, 8, 12, 18, 24];
type WavetableBank = [[Vec<f32>; 8]; 6];

#[allow(clippy::needless_range_loop)] // Parallel, fixed-size wavetable banks use shared indices.
fn make_tables() -> WavetableBank {
    let formants = [1.0f32, 0.82, 0.66, 0.48, 0.32, 0.22, 0.15, 0.10, 0.07, 0.05];
    let brass = [1.0f32, 0.72, 0.54, 0.39, 0.32, 0.23, 0.18, 0.12];
    let mut bank: WavetableBank =
        std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0; TABLE_SIZE]));
    for level in 0..8 {
        let partial_count = FORMANT_MIPS[level];
        let mut choir = vec![0.0f32; TABLE_SIZE];
        let mut brass_table = vec![0.0f32; TABLE_SIZE];
        for i in 0..TABLE_SIZE {
            let phase = i as f32 / TABLE_SIZE as f32;
            let mut sum = 0.0;
            for (h, weight) in formants
                .iter()
                .enumerate()
                .take(partial_count.min(formants.len()))
            {
                sum += (phase * std::f32::consts::TAU * (h + 1) as f32).sin() * weight;
            }
            choir[i] = sum;
            let mut sum = 0.0;
            for (h, weight) in brass
                .iter()
                .enumerate()
                .take(partial_count.min(brass.len()))
            {
                sum += (phase * std::f32::consts::TAU * (h + 1) as f32).sin() * weight;
            }
            brass_table[i] = sum;
        }
        bank[3][level] = choir;
        bank[4][level] = brass_table;
    }
    let choir_peak = bank[3][7].iter().fold(0.0f32, |p, x| p.max(x.abs()));
    let brass_peak = bank[4][7].iter().fold(0.0f32, |p, x| p.max(x.abs()));
    for level in 0..8 {
        for x in &mut bank[3][level] {
            *x *= 0.85 / choir_peak.max(1e-6);
        }
        for x in &mut bank[4][level] {
            *x *= 0.85 / brass_peak.max(1e-6);
        }
    }
    for level in 0..8 {
        for i in 0..TABLE_SIZE {
            let phase = i as f32 / TABLE_SIZE as f32;
            let mut v = 0.0;
            for h in (1..=FORMANT_MIPS[level]).step_by(2) {
                let sign = if (h - 1) / 2 % 2 == 0 { 1.0 } else { -1.0 };
                v += (std::f32::consts::TAU * phase * h as f32).sin() * sign / (h * h) as f32;
            }
            bank[5][level][i] = v * 8.0 / std::f32::consts::PI.powi(2);
        }
    }
    for i in 0..TABLE_SIZE {
        let phase = i as f32 / TABLE_SIZE as f32;
        let sine = (phase * std::f32::consts::TAU).sin();
        let mut saw = 0.0;
        let mut square = 0.0;
        for h in 1..=24 {
            let partial = (phase * std::f32::consts::TAU * h as f32).sin() / h as f32;
            saw += partial;
            if h % 2 == 1 {
                square += partial;
            }
        }
        for level in 0..8 {
            bank[0][level][i] = sine;
            bank[1][level][i] = (saw * 0.62).clamp(-1.0, 1.0);
            bank[2][level][i] = (square * 1.2).clamp(-1.0, 1.0);
        }
    }
    bank
}

fn table_sample(table: &[f32], phase: f32) -> f32 {
    let p = phase.rem_euclid(1.0) * TABLE_SIZE as f32;
    let i = p as usize;
    let frac = p - i as f32;
    let a = table[i % TABLE_SIZE];
    let b = table[(i + 1) % TABLE_SIZE];
    a + (b - a) * frac
}

#[inline]
fn formant_sample(mips: &[Vec<f32>; 8], phase: f32, frequency: f32, sr: f32) -> f32 {
    let max_harmonic = ((sr * 0.45 / frequency.max(1.0)).floor() as usize).max(1);
    let level = FORMANT_MIPS
        .iter()
        .rposition(|h| *h <= max_harmonic)
        .unwrap_or(0);
    table_sample(&mips[level], phase)
}

fn is_drum_role(role: &str) -> bool {
    matches!(
        role,
        "kick" | "hat" | "hat_16" | "open_hat" | "clap" | "snare" | "perc" | "ride"
    )
}

#[allow(clippy::too_many_arguments)] // Explicit DSP buffers and musical-time context.
fn fire_sample(
    track: &Track,
    samples: &HashMap<String, AudioSample>,
    start: usize,
    output_rate: u32,
    gain: f32,
    event_index: usize,
    out: &mut [f32],
    rng: &mut u64,
    out_side: &mut [f32],
) -> bool {
    let id = if event_index % 2 == 1 {
        track
            .sample_alt_id
            .as_deref()
            .or(track.sample_id.as_deref())
    } else {
        track
            .sample_id
            .as_deref()
            .or(track.sample_alt_id.as_deref())
    };
    let Some(sample) = id.and_then(|key| samples.get(key)) else {
        return false;
    };
    let ratio = sample.sample_rate as f64 / output_rate as f64
        * 2.0f64.powf(track.sample_tune_semitones.clamp(-24.0, 24.0) as f64 / 12.0);
    let source_start =
        (track.drum.sample_start_ms.max(0.0) as f64 * sample.sample_rate as f64 / 1000.0) as usize;
    let available = out.len().saturating_sub(start);
    let source_available = sample.mono.len().saturating_sub(source_start);
    let natural_frames = (source_available as f64 / ratio.max(1e-9)).ceil() as usize;
    let requested_frames = if track.drum.decay_ms > 0.0 {
        (track.drum.decay_ms * output_rate as f32 / 1000.0).round() as usize
    } else {
        natural_frames
    };
    let frames = requested_frames.min(natural_frames).min(available);
    let fade_frames = (output_rate as usize / 250).min(frames / 2).max(1);
    for i in 0..frames {
        let position = source_start as f64 + i as f64 * ratio;
        let index = position.floor() as usize;
        if index >= sample.mono.len() {
            break;
        }
        let time = i as f32 / output_rate as f32;
        let transient = 1.0 + track.drum.transient * (-time / 0.008).exp();
        let fade = if i >= frames.saturating_sub(fade_frames) {
            (frames - i) as f32 / fade_frames as f32
        } else {
            1.0
        };
        let noise = if track.drum.noise_mix > 0.0 {
            next_noise(rng) * track.drum.noise_mix * 0.12 * (-time / 0.009).exp()
        } else {
            0.0
        };
        out[start + i] +=
            (sample_value(&sample.mono, position, ratio) * transient + noise) * gain * fade;
        out_side[start + i] +=
            sample_value(&sample.side, position, ratio) * transient * gain * fade;
    }
    true
}

fn load_samples(song: &Song) -> Result<HashMap<String, AudioSample>, String> {
    let mut assets: Vec<(String, String)> = vec![
        (
            "house.kick.punch-02".into(),
            "samples/kicks/punch-02.wav".into(),
        ),
        (
            "house.kick.punch-04".into(),
            "samples/kicks/punch-04.wav".into(),
        ),
        (
            "house.kick.deep-02".into(),
            "samples/kicks/deep-02.wav".into(),
        ),
        (
            "house.hat.closed-08".into(),
            "samples/hats/closed-08.wav".into(),
        ),
        (
            "house.hat.closed-05".into(),
            "samples/hats/closed-05.wav".into(),
        ),
        (
            "house.hat.closed-06".into(),
            "samples/hats/closed-06.wav".into(),
        ),
        (
            "house.hat.closed-04".into(),
            "samples/hats/closed-04.wav".into(),
        ),
        (
            "house.hat.open-03".into(),
            "samples/hats/open-03.wav".into(),
        ),
        (
            "house.hat.open-04".into(),
            "samples/hats/open-04.wav".into(),
        ),
        (
            "house.clap.classic-02".into(),
            "samples/claps/classic-02.wav".into(),
        ),
        (
            "house.snare.snap-01".into(),
            "samples/claps/snap-01.wav".into(),
        ),
        (
            "house.perc.shaker-02".into(),
            "samples/percs/shaker-02.wav".into(),
        ),
        (
            "house.perc.tick-02".into(),
            "samples/percs/tick-02.wav".into(),
        ),
        (
            "house.perc.tom-01".into(),
            "samples/percs/tom-01.wav".into(),
        ),
        (
            "house.perc.tom-02".into(),
            "samples/percs/tom-02.wav".into(),
        ),
        (
            "house.ride.ping-01".into(),
            "samples/cymbals/ride-01.wav".into(),
        ),
    ];
    for pitch in [48, 55, 60, 65, 69, 74, 79, 84] {
        for velocity in [2, 4] {
            assets.push((
                format!("instrument.piano.v{velocity}.m{pitch}"),
                format!("samples/instruments/piano/piano-v{velocity}-m{pitch}.wav"),
            ));
        }
    }
    for pitch in [48, 52, 57, 60, 64, 69, 72, 76, 81, 84] {
        assets.push((
            format!("instrument.flute.m{pitch}"),
            format!("samples/instruments/flute/flute-m{pitch}.wav"),
        ));
    }
    for pitch in [40, 45, 50, 55, 60, 65, 69, 74, 79, 83] {
        for layer in ["f", "p", "p2"] {
            assets.push((
                format!("instrument.guitar.{layer}.m{pitch}"),
                format!("samples/instruments/guitar/guitar-{layer}-m{pitch}.wav"),
            ));
        }
    }
    for pitch in [45, 50, 55, 60, 64, 69, 74, 79, 84] {
        for layer in ["v2", "v3"] {
            assets.push((
                format!("instrument.steel_guitar.{layer}.m{pitch}"),
                format!("samples/instruments/steel-guitar/steel-guitar-{layer}-m{pitch}.wav"),
            ));
        }
    }
    for (instrument, roots) in [
        ("ghost-mult", [48, 60, 72, 84]),
        ("saw-morph", [48, 60, 72, 84]),
        ("supra-saw", [48, 60, 72, 84]),
        ("fuzz-bass", [36, 48, 60, 72]),
    ] {
        for pitch in roots {
            let id = format!("instrument.sf2.{instrument}.m{pitch}");
            assets.push((
                id,
                format!("samples/instruments/sf2/{instrument}/m{pitch}.wav"),
            ));
        }
    }
    for track in song.tracks.iter().filter(|t| !t.muted) {
        if let (Some(id), Some(path)) = (&track.sample_id, &track.sample_path) {
            assets.push((id.clone(), path.clone()));
        }
        if let (Some(id), Some(path)) = (&track.sample_alt_id, &track.sample_alt_path) {
            assets.push((id.clone(), path.clone()));
        }
    }
    let mut used = Vec::<String>::new();
    for track in song
        .tracks
        .iter()
        .filter(|t| !t.muted && !t.clips.is_empty() && t.source_type != "synth")
    {
        if !matches!(
            track.role.as_str(),
            "piano" | "flute" | "guitar" | "steel_guitar"
        ) && !track.role.starts_with("sf2_")
        {
            used.extend(
                [track.sample_id.clone(), track.sample_alt_id.clone()]
                    .into_iter()
                    .flatten(),
            );
            continue;
        }
        for clip in &track.clips {
            if let Some(pattern) = track.patterns.iter().find(|p| p.id == clip.pattern_id) {
                for note in pattern
                    .notes
                    .iter()
                    .filter(|n| n.bar <= clip.bars.min(pattern.bars))
                {
                    let pitch = (note.pitch as i16 + clip.transpose).clamp(0, 127);
                    if track.role.starts_with("sf2_") {
                        let program = track
                            .sample_id
                            .as_ref()
                            .ok_or("sampled synthesis requires sample_id")?;
                        let roots = if program == "instrument.sf2.fuzz-bass" {
                            [36i16, 48, 60, 72]
                        } else {
                            [48i16, 60, 72, 84]
                        };
                        let root = roots.into_iter().min_by_key(|r| (pitch - r).abs()).unwrap();
                        used.push(format!("{program}.m{root}"));
                    } else if let Some(id) = instrument_zone(&track.role, pitch, note.velocity) {
                        used.push(id);
                    }
                }
            }
        }
    }
    let mut loaded = HashMap::new();
    for id in used {
        if loaded.contains_key(&id) {
            continue;
        }
        let Some((_, relative)) = assets.iter().find(|(known, _)| known == &id) else {
            return Err(format!(
                "unknown sample_id `{id}`; use an ID from the installed CC0 sample library"
            ));
        };
        let relative_path = Path::new(relative);
        let path = if relative_path.is_absolute() {
            relative_path.to_path_buf()
        } else {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("assets")
                .join(relative_path)
        };
        loaded.insert(id.to_string(), read_pcm16_mono(&path)?);
    }
    Ok(loaded)
}

fn instrument_zone(role: &str, pitch: i16, velocity: u8) -> Option<String> {
    let (prefix, roots): (String, &[i16]) = match role {
        "piano" => (
            format!("instrument.piano.v{}.m", if velocity >= 76 { 4 } else { 2 }),
            &[48, 55, 60, 65, 69, 74, 79, 84],
        ),
        "flute" => (
            "instrument.flute.m".into(),
            &[48, 52, 57, 60, 64, 69, 72, 76, 81, 84],
        ),
        "guitar" => (
            format!(
                "instrument.guitar.{}.m",
                if velocity < 55 {
                    "f"
                } else if velocity >= 100 {
                    "p2"
                } else {
                    "p"
                }
            ),
            &[40, 45, 50, 55, 60, 65, 69, 74, 79, 83],
        ),
        "steel_guitar" => (
            format!(
                "instrument.steel_guitar.{}.m",
                if velocity >= 97 { "v3" } else { "v2" }
            ),
            &[45, 50, 55, 60, 64, 69, 74, 79, 84],
        ),
        _ => return None,
    };
    let root = roots.iter().min_by_key(|r| (pitch - **r).abs())?;
    Some(format!("{prefix}{root}"))
}

#[allow(clippy::too_many_arguments)] // Explicit DSP buffers and musical-time context.
fn fire_sf2_sample(
    track: &Track,
    pitch: u8,
    velocity: u8,
    start: usize,
    gate_frames: usize,
    tune_semitones: f32,
    output_rate: u32,
    samples: &HashMap<String, AudioSample>,
    out: &mut [f32],
    out_side: &mut [f32],
) -> bool {
    let Some(program) = track.sample_id.as_deref() else {
        return false;
    };
    let roots: &[i16] = if program == "instrument.sf2.fuzz-bass" {
        &[36, 48, 60, 72]
    } else {
        &[48, 60, 72, 84]
    };
    let root = *roots
        .iter()
        .min_by_key(|root| (i16::from(pitch) - **root).abs())
        .unwrap_or(&60);
    let id = format!("{program}.m{root}");
    let Some(sample) = samples.get(&id) else {
        return false;
    };
    let transpose = i16::from(pitch) - root;
    let rate_ratio = sample.sample_rate as f64 / output_rate as f64
        * 2.0f64.powf((transpose as f64 + tune_semitones.clamp(-24.0, 24.0) as f64) / 12.0);
    let available = out.len().saturating_sub(start);
    let release_seconds = match track.role.as_str() {
        "sf2_bass" => 0.105,
        "sf2_chords" => 0.16,
        _ => 0.22,
    };
    let release_frames = (output_rate as f32 * release_seconds) as usize;
    let frames = gate_frames.saturating_add(release_frames).min(available);
    let amp = (velocity as f32 / 100.0).clamp(0.3, 1.15).powf(1.12) * 0.54;
    for i in 0..frames {
        let position = i as f64 * rate_ratio;
        let index = position.floor() as usize;
        if index >= sample.mono.len() {
            break;
        }
        let attack = (i as f32 / (output_rate as f32 * 0.006).max(1.0)).clamp(0.0, 1.0);
        let release = if i > gate_frames {
            (-((i - gate_frames) as f32) / (release_frames as f32 * 0.34).max(1.0)).exp()
        } else {
            1.0
        };
        out[start + i] += sample_value(&sample.mono, position, rate_ratio) * amp * attack * release;
        out_side[start + i] +=
            sample_value(&sample.side, position, rate_ratio) * amp * attack * release;
    }
    true
}

#[allow(clippy::too_many_arguments)] // Explicit DSP buffers and musical-time context.
fn fire_instrument_sample(
    role: &str,
    pitch: u8,
    velocity: u8,
    start: usize,
    gate_frames: usize,
    tune_semitones: f32,
    output_rate: u32,
    samples: &HashMap<String, AudioSample>,
    out: &mut [f32],
    out_side: &mut [f32],
) -> bool {
    let (prefix, roots, layer) = if role == "piano" {
        let velocity_layer = if velocity >= 76 { "v4" } else { "v2" };
        (
            format!("instrument.piano.{velocity_layer}.m"),
            [48, 55, 60, 65, 69, 74, 79, 84].as_slice(),
            true,
        )
    } else if role == "flute" {
        (
            "instrument.flute.m".to_string(),
            [48, 52, 57, 60, 64, 69, 72, 76, 81, 84].as_slice(),
            false,
        )
    } else if role == "guitar" {
        let velocity_layer = if velocity < 55 {
            "f"
        } else if velocity >= 100 {
            "p2"
        } else {
            "p"
        };
        (
            format!("instrument.guitar.{velocity_layer}.m"),
            [40, 45, 50, 55, 60, 65, 69, 74, 79, 83].as_slice(),
            true,
        )
    } else if role == "steel_guitar" {
        let velocity_layer = if velocity >= 97 { "v3" } else { "v2" };
        (
            format!("instrument.steel_guitar.{velocity_layer}.m"),
            [45, 50, 55, 60, 64, 69, 74, 79, 84].as_slice(),
            true,
        )
    } else {
        return false;
    };
    let root = *roots
        .iter()
        .min_by_key(|root| (i16::from(pitch) - **root).abs())
        .unwrap_or(&60);
    let id = format!("{prefix}{root}");
    let Some(sample) = samples.get(&id) else {
        return false;
    };
    let transpose = i16::from(pitch) - root;
    let rate_ratio = sample.sample_rate as f64 / output_rate as f64
        * 2.0f64.powf((transpose as f64 + tune_semitones.clamp(-24.0, 24.0) as f64) / 12.0);
    let available = out.len().saturating_sub(start);
    let frames = if layer {
        ((output_rate as f64 * 2.2).round() as usize).min(available)
    } else {
        gate_frames
            .saturating_add((output_rate as f32 * 0.11) as usize)
            .min(available)
    };
    let amp = if layer {
        (velocity as f32 / 100.0).clamp(0.35, 1.1).powf(1.15) * 0.42
    } else {
        (velocity as f32 / 127.0).powf(1.2) * 0.32
    };
    for i in 0..frames {
        let position = i as f64 * rate_ratio;
        let index = position.floor() as usize;
        if index >= sample.mono.len() {
            break;
        }
        let mut env = 1.0;
        if !layer {
            let attack = (i as f32 / (output_rate as f32 * 0.012).max(1.0)).clamp(0.0, 1.0);
            let release_start = frames.saturating_sub((output_rate as f32 * 0.09) as usize);
            let release = if i >= release_start {
                (frames - i) as f32 / (frames - release_start).max(1) as f32
            } else {
                1.0
            };
            env = (attack * release).clamp(0.0, 1.0);
        }
        let edge_fade = if role == "guitar"
            && i > frames.saturating_sub((output_rate as f64 * 0.08) as usize)
        {
            (frames - i) as f32
                / (frames - frames.saturating_sub((output_rate as f64 * 0.08) as usize)).max(1)
                    as f32
        } else {
            1.0
        };
        out[start + i] += sample_value(&sample.mono, position, rate_ratio) * amp * env * edge_fade;
        out_side[start + i] +=
            sample_value(&sample.side, position, rate_ratio) * amp * env * edge_fade;
    }
    true
}

fn sample_value(samples: &[f32], position: f64, ratio: f64) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let index = position.floor() as isize;
    let frac = position - index as f64;
    let at = |i: isize| samples.get(i.max(0) as usize).copied().unwrap_or(0.0) as f64;
    if ratio <= 1.001 {
        let (a, b, c, d) = (at(index - 1), at(index), at(index + 1), at(index + 2));
        return (b + 0.5
            * frac
            * (c - a + frac * (2.0 * a - 5.0 * b + 4.0 * c - d + frac * (3.0 * (b - c) + d - a))))
            as f32;
    }
    let cutoff = 0.94 / ratio;
    let mut value = 0.0;
    let mut sum = 0.0;
    for k in -12..=12 {
        let x = k as f64 - frac;
        let wx = x / 13.0;
        if wx.abs() >= 1.0 {
            continue;
        }
        let sinc = if x.abs() < 1e-8 {
            cutoff
        } else {
            (std::f64::consts::PI * cutoff * x).sin() / (std::f64::consts::PI * x)
        };
        let window = 0.42
            + 0.5 * (std::f64::consts::PI * wx).cos()
            + 0.08 * (2.0 * std::f64::consts::PI * wx).cos();
        let coefficient = sinc * window;
        value += at(index + k) * coefficient;
        sum += coefficient;
    }
    (value / sum.max(1e-12)) as f32
}

fn read_pcm16_mono(path: &Path) -> Result<AudioSample, String> {
    let mut bytes = Vec::new();
    BufReader::new(
        File::open(path).map_err(|e| format!("cannot read sample {}: {e}", path.display()))?,
    )
    .read_to_end(&mut bytes)
    .map_err(|e| format!("cannot read sample {}: {e}", path.display()))?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("sample {} is not a RIFF/WAVE file", path.display()));
    }
    let mut format = None;
    let mut data = None;
    let mut channels = 0usize;
    let mut sample_rate = 0u32;
    let mut bits = 0u16;
    let mut cursor = 12usize;
    while cursor + 8 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        let begin = cursor + 8;
        let end = begin.saturating_add(size).min(bytes.len());
        match &bytes[cursor..cursor + 4] {
            b"fmt " if size >= 16 => {
                format = Some(u16::from_le_bytes(
                    bytes[begin..begin + 2].try_into().unwrap(),
                ));
                channels =
                    u16::from_le_bytes(bytes[begin + 2..begin + 4].try_into().unwrap()) as usize;
                sample_rate = u32::from_le_bytes(bytes[begin + 4..begin + 8].try_into().unwrap());
                bits = u16::from_le_bytes(bytes[begin + 14..begin + 16].try_into().unwrap());
            }
            b"data" => data = Some(&bytes[begin..end]),
            _ => {}
        }
        cursor = begin.saturating_add(size + (size & 1));
    }
    if !matches!((format, bits), (Some(1), 16 | 24 | 32) | (Some(3), 32))
        || !(1..=2).contains(&channels)
        || sample_rate == 0
    {
        return Err(format!(
            "sample {} must be mono/stereo PCM16/24/32 or float32 WAV",
            path.display()
        ));
    }
    let data = data.ok_or_else(|| format!("sample {} has no data chunk", path.display()))?;
    let sample_bytes = bits as usize / 8;
    let frame_bytes = channels * sample_bytes;
    let mut mono = Vec::with_capacity(data.len() / frame_bytes);
    let mut side = Vec::with_capacity(if channels == 2 {
        data.len() / frame_bytes
    } else {
        0
    });
    let decode = |bytes: &[u8]| -> f32 {
        match (format, bits) {
            (Some(3), 32) => f32::from_le_bytes(bytes[..4].try_into().unwrap()),
            (_, 16) => i16::from_le_bytes(bytes[..2].try_into().unwrap()) as f32 / 32768.0,
            (_, 24) => {
                ((i32::from_le_bytes([bytes[0], bytes[1], bytes[2], 0]) << 8) >> 8) as f32
                    / 8388608.0
            }
            _ => i32::from_le_bytes(bytes[..4].try_into().unwrap()) as f32 / 2147483648.0,
        }
    };
    for frame in data.chunks_exact(frame_bytes) {
        let left = decode(frame);
        let right = if channels == 2 {
            decode(&frame[sample_bytes..])
        } else {
            left
        };
        if !left.is_finite() || !right.is_finite() {
            return Err(format!("non-finite sample in {}", path.display()));
        }
        mono.push((left + right) * 0.5);
        if channels == 2 {
            side.push((left - right) * 0.5);
        }
    }
    if mono.is_empty() {
        return Err(format!("sample {} contains no audio", path.display()));
    }
    Ok(AudioSample {
        mono,
        side,
        sample_rate,
    })
}

fn automation_at(track: &Track, parameter: &str, beat: f64, fallback: f64, numerator: u8) -> f64 {
    let mut points: Vec<(f64, f64)> = track
        .automation
        .iter()
        .filter(|p| p.parameter == parameter)
        .map(|p| {
            (
                ((p.bar.saturating_sub(1)) as f64 * numerator as f64) + p.beat - 1.0,
                p.value as f64,
            )
        })
        .collect();
    if points.is_empty() {
        return fallback;
    }
    points.sort_by(|a, b| a.0.total_cmp(&b.0));
    if beat <= points[0].0 {
        return points[0].1;
    }
    for pair in points.windows(2) {
        if beat <= pair[1].0 {
            let width = (pair[1].0 - pair[0].0).max(1e-9);
            let t = ((beat - pair[0].0) / width).clamp(0.0, 1.0);
            return pair[0].1 + (pair[1].1 - pair[0].1) * t;
        }
    }
    points.last().map(|p| p.1).unwrap_or(fallback)
}

#[allow(clippy::too_many_arguments)] // Explicit DSP buffers and musical-time context.
fn synth_note(
    track: &Track,
    sound: &SoundSettings,
    pitch: u8,
    velocity: u8,
    start: usize,
    gate_dur: usize,
    bar0: u32,
    frames_per_bar: usize,
    sr: u32,
    table: &WavetableBank,
    out: &mut [f32],
    rng: &mut u64,
    out_side: &mut [f32],
) {
    let frequency = 440.0 * 2.0f32.powf((pitch as f32 - 69.0) / 12.0);
    // A role label no longer changes the level or inserts an undocumented sub oscillator.
    let amp = (velocity as f32 / 127.0).powf(1.2)
        * 0.30
        * crate::dsp::gain(sound.output_gain_db.clamp(-36.0, 18.0));
    let available = out.len().saturating_sub(start);
    let gate = gate_dur.min(available);
    let release = (sound.release_ms.clamp(0.0, 10000.0) * sr as f32 / 1000.0).max(1.0) as usize;
    let frames = gate.saturating_add(release).min(available);
    let attack = (sound.attack_ms.clamp(0.0, 5000.0) * sr as f32 / 1000.0).max(1.0);
    let decay = (sound.decay_ms.clamp(1.0, 10000.0) * sr as f32 / 1000.0).max(1.0);
    let envelope =
        |frame: usize| adsr_level(frame, gate, attack, decay, sound.sustain, release as f32);
    let gate_level = envelope(gate);
    let default_layer = SynthLayer {
        waveform: sound.waveform.clone(),
        unison: sound.unison,
        detune_cents: sound.detune_cents,
        ..SynthLayer::default()
    };
    let layers: Vec<&SynthLayer> = if sound.layers.is_empty() {
        vec![&default_layer]
    } else {
        sound.layers.iter().take(8).collect()
    };
    let layer_envelopes: Vec<(bool, f32, f32, f32, f32)> = layers
        .iter()
        .map(|layer| {
            let custom = layer.attack_ms.is_some()
                || layer.decay_ms.is_some()
                || layer.sustain.is_some()
                || layer.release_ms.is_some();
            (
                custom,
                (layer.attack_ms.unwrap_or(sound.attack_ms).max(0.0) * sr as f32 / 1000.0).max(1.0),
                (layer.decay_ms.unwrap_or(sound.decay_ms).max(1.0) * sr as f32 / 1000.0).max(1.0),
                layer.sustain.unwrap_or(sound.sustain).clamp(0.0, 1.0),
                (layer.release_ms.unwrap_or(sound.release_ms).max(0.0) * sr as f32 / 1000.0)
                    .max(1.0),
            )
        })
        .collect();
    let mut phases = [[0.0f32; 8]; 8];
    let mut modphases = [[0.0f32; 8]; 8];
    let mut ic1 = [[0.0f32; 2]; 8];
    let mut ic2 = [[0.0f32; 2]; 8];
    let absolute_start = bar0 as u64 * frames_per_bar as u64 + start as u64;
    for (li, layer) in layers.iter().enumerate() {
        for vi in 0..layer.unison.clamp(1, 8) as usize {
            let voices = layer.unison.clamp(1, 8) as usize;
            let spread = if voices == 1 {
                0.0
            } else {
                vi as f32 / (voices - 1) as f32 * 2.0 - 1.0
            };
            let ratio = 2.0f32.powf(
                layer.octave as f32
                    + (layer.tune_semitones + spread * layer.detune_cents / 100.0) / 12.0,
            );
            let cycles = absolute_start as f64 * frequency as f64 * ratio as f64 / sr as f64;
            phases[li][vi] = (if sound.phase_reset {
                li as f64 * 0.381966 + vi as f64 * 0.618034
            } else {
                cycles + li as f64 * 0.381966 + vi as f64 * 0.618034
            })
            .fract() as f32;
            modphases[li][vi] = if sound.phase_reset {
                0.0
            } else {
                (cycles * layer.fm_ratio as f64).fract() as f32
            };
        }
    }
    for i in 0..frames {
        let t = i as f32 / sr as f32;
        let absolute_t = (absolute_start + i as u64) as f64 / sr as f64;
        let env = if i < gate {
            envelope(i)
        } else {
            gate_level
                * (0.5 + 0.5 * (std::f32::consts::PI * (i - gate) as f32 / release as f32).cos())
        };
        let filter_env = (-t / (sound.filter_decay_ms.clamp(1.0, 10000.0) / 1000.0)).exp();
        let lfo = (std::f64::consts::TAU * sound.lfo_rate_hz as f64 * absolute_t).sin() as f32;
        let lfo_amp = if sound.lfo_target == "amplitude" {
            1.0 + lfo * sound.lfo_depth.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let mut sum = [0.0f32; 2];
        for (li, layer) in layers.iter().enumerate() {
            let (custom_envelope, layer_attack, layer_decay, layer_sustain, layer_release) =
                layer_envelopes[li];
            let layer_env = if custom_envelope {
                adsr_level(
                    i,
                    gate,
                    layer_attack,
                    layer_decay,
                    layer_sustain,
                    layer_release,
                )
            } else {
                env
            };
            let voices = layer.unison.clamp(1, 8) as usize;
            let mut layer_sum = [0.0f32; 2];
            for vi in 0..voices {
                let spread = if voices == 1 {
                    0.0
                } else {
                    vi as f32 / (voices - 1) as f32 * 2.0 - 1.0
                };
                let ratio = 2.0f32.powf(
                    layer.octave as f32
                        + (layer.tune_semitones
                            + spread * layer.detune_cents.clamp(0.0, 50.0) / 100.0)
                            / 12.0,
                );
                let pitch_envelope = sound.pitch_envelope_semitones.clamp(-24.0, 48.0)
                    * (-t / (sound.pitch_decay_ms.clamp(1.0, 5000.0) / 1000.0)).exp();
                let pitch_lfo = if sound.lfo_target == "pitch" {
                    (lfo + sound.lfo_bias.clamp(-1.0, 1.0)) * sound.lfo_depth.clamp(0.0, 4.0)
                } else {
                    0.0
                };
                let pitch_shift = pitch_envelope + pitch_lfo;
                let f = frequency * ratio * 2.0f32.powf(pitch_shift / 12.0);
                let dt = (f / sr as f32).clamp(0.0, 0.49);
                let phase = phases[li][vi];
                let modphase = modphases[li][vi];
                let fm_env = if layer.fm_decay_ms > 0.0 {
                    (-t / (layer.fm_decay_ms / 1000.0)).exp()
                } else {
                    1.0
                };
                let index = (layer.fm_amount.clamp(0.0, 1.0) * 8.0 * fm_env).min(
                    ((sr as f32 * 0.42 / f.max(1.0) - 1.0) / layer.fm_ratio.max(0.25)).max(0.0),
                );
                let pwm = (std::f64::consts::TAU * layer.pulse_width_rate_hz as f64 * absolute_t)
                    .sin() as f32
                    * layer.pulse_width_depth;
                let width = (layer.pulse_width + pwm).clamp(0.05, 0.95);
                let osc = match layer.waveform.as_str() {
                    "sine" => table_sample(&table[0][0], phase),
                    "saw" => 2.0 * phase - 1.0 - poly_blep(phase, dt),
                    "square" => {
                        let mut v = if phase < width { 1.0 } else { -1.0 };
                        v += poly_blep(phase, dt);
                        v -= poly_blep((phase - width).rem_euclid(1.0), dt);
                        v - (2.0 * width - 1.0)
                    }
                    "triangle" => formant_sample(&table[5], phase, f, sr as f32),
                    "choir" => formant_sample(&table[3], phase, f, sr as f32),
                    "brass" => formant_sample(&table[4], phase, f, sr as f32),
                    "fm" => (std::f32::consts::TAU * phase
                        + index * (std::f32::consts::TAU * modphase).sin())
                    .sin(),
                    "noise" => next_noise(rng),
                    _ => 0.0,
                };
                phases[li][vi] = (phase + dt).fract();
                modphases[li][vi] = (modphase + dt * layer.fm_ratio.clamp(0.25, 12.0)).fract();
                let noise = layer.noise_amount.clamp(0.0, 1.0);
                let x = osc * (1.0 - noise)
                    + if noise > 0.0 {
                        next_noise(rng) * noise
                    } else {
                        0.0
                    };
                let pan =
                    (spread * sound.stereo_spread.clamp(0.0, 1.0) + layer.pan).clamp(-1.0, 1.0);
                layer_sum[0] += x * (1.0 - pan).sqrt();
                layer_sum[1] += x * (1.0 + pan).sqrt();
            }
            let cutoff_mod = sound.filter_env_amount.clamp(-4.0, 4.0) * filter_env
                + if sound.lfo_target == "cutoff" {
                    lfo * sound.lfo_depth.clamp(0.0, 4.0)
                } else {
                    0.0
                };
            let cutoff = (layer.cutoff_hz.unwrap_or(sound.cutoff_hz) * 2.0f32.powf(cutoff_mod))
                .clamp(20.0, sr as f32 * 0.42);
            let g = (std::f32::consts::PI * cutoff / sr as f32).tan();
            let resonance = layer
                .filter_resonance
                .unwrap_or(sound.filter_resonance)
                .clamp(0.0, 0.90);
            let damping = 2.0 * (1.0 - resonance);
            let a1 = 1.0 / (1.0 + g * (g + damping));
            let a2 = g * a1;
            let a3 = g * a2;
            for channel in 0..2 {
                let x = layer_sum[channel] / (voices as f32).sqrt();
                let v3 = x - ic2[li][channel];
                let band = a1 * ic1[li][channel] + a2 * v3;
                let low = ic2[li][channel] + a2 * ic1[li][channel] + a3 * v3;
                ic1[li][channel] = 2.0 * band - ic1[li][channel];
                ic2[li][channel] = 2.0 * low - ic2[li][channel];
                let high = x - damping * band - low;
                let y = match sound.filter_mode.as_str() {
                    "highpass" => high,
                    "bandpass" => band * damping,
                    "notch" => low + high,
                    "allpass" => low + high - damping * band,
                    _ => low,
                };
                sum[channel] += y * layer.gain.clamp(0.0, 2.0) * layer_env;
            }
        }
        let amplitude = amp * lfo_amp;
        out[start + i] += (sum[0] + sum[1]) * 0.5 * amplitude;
        out_side[start + i] += (sum[0] - sum[1]) * 0.5 * amplitude;
    }
    let _ = track; // role names are metadata, never a hidden signal processor.
}

fn adsr_level(
    frame: usize,
    gate: usize,
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
) -> f32 {
    let level = |n: f32| {
        if n < attack {
            0.5 - 0.5 * (std::f32::consts::PI * n / attack).cos()
        } else {
            sustain.clamp(0.0, 1.0)
                + (1.0 - sustain.clamp(0.0, 1.0)) * (-3.0 * (n - attack) / decay).exp()
        }
    };
    if frame < gate {
        level(frame as f32)
    } else {
        level(gate as f32)
            * (0.5 + 0.5 * (std::f32::consts::PI * (frame - gate) as f32 / release).cos())
    }
}

fn next_noise(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 40) as f32 / 8_388_608.0) - 1.0
}

fn poly_blep(phase: f32, increment: f32) -> f32 {
    if increment <= 0.0 {
        return 0.0;
    }
    let p = phase.rem_euclid(1.0);
    if p < increment {
        let x = p / increment;
        x + x - x * x - 1.0
    } else if p > 1.0 - increment {
        let x = (p - 1.0) / increment;
        x * x + x + x + 1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod synth_regressions {
    use super::*;

    fn render_patch(sound: SoundSettings, sr: u32) -> Vec<f32> {
        let track: Track = serde_json::from_value(serde_json::json!({
            "id":"regression", "name":"Regression voice", "role":"arp"
        }))
        .unwrap();
        let mut out = vec![0.0; sr as usize / 2];
        let mut rng = 41;
        synth_note(
            &track,
            &sound,
            57,
            100,
            0,
            sr as usize * 2 / 5,
            0,
            sr as usize * 2,
            sr,
            &make_tables(),
            &mut out,
            &mut rng,
            &mut vec![0.0; sr as usize / 2],
        );
        out
    }

    #[test]
    fn high_cutoff_envelopes_do_not_poison_synth_audio() {
        for sr in [22_050, 48_000, 128_000] {
            for cutoff_hz in [5200.0, 20_000.0, 40_000.0] {
                let sound = SoundSettings {
                    cutoff_hz,
                    filter_resonance: 0.05,
                    filter_env_amount: 0.45,
                    attack_ms: 3.0,
                    decay_ms: 220.0,
                    sustain: 0.36,
                    layers: vec![SynthLayer {
                        waveform: "fm".into(),
                        fm_amount: 0.10,
                        ..SynthLayer::default()
                    }],
                    ..SoundSettings::default()
                };
                let out = render_patch(sound, sr);
                assert!(
                    out.iter().all(|x| x.is_finite() && x.abs() < 10.0),
                    "unstable synth at {sr} Hz, cutoff {cutoff_hz}"
                );
                assert!(
                    out[out.len() / 3..out.len() * 2 / 3]
                        .iter()
                        .any(|x| x.abs() > 0.001)
                );
            }
        }
    }

    #[test]
    fn oscillator_octave_is_a_frequency_doubling() {
        let crossings = |octave| {
            let sound = SoundSettings {
                cutoff_hz: 10_000.0,
                attack_ms: 1.0,
                sustain: 1.0,
                layers: vec![SynthLayer {
                    waveform: "sine".into(),
                    octave,
                    ..SynthLayer::default()
                }],
                ..SoundSettings::default()
            };
            let out = render_patch(sound, 48_000);
            out[2400..16800]
                .windows(2)
                .filter(|w| w[0] <= 0.0 && w[1] > 0.0)
                .count()
        };
        let base = crossings(0);
        let upper = crossings(1);
        assert!(
            (upper as i32 - 2 * base as i32).abs() <= 2,
            "expected double frequency, got {base} and {upper} crossings"
        );
    }
}

#[cfg(test)]
mod engine_regressions {
    use super::*;
    fn track(role: &str, sound: serde_json::Value) -> Track {
        serde_json::from_value(
            serde_json::json!({"id":"probe","name":"Probe","role":role,"sound":sound}),
        )
        .unwrap()
    }
    fn voice(t: &Track, gate: usize) -> (Vec<f32>, Vec<f32>) {
        let mut a = vec![0.0; 24000];
        let mut b = a.clone();
        let mut rng = 41;
        synth_note(
            t,
            &t.sound,
            69,
            100,
            0,
            gate,
            0,
            96000,
            48000,
            &make_tables(),
            &mut a,
            &mut rng,
            &mut b,
        );
        (a, b)
    }
    fn tempdir(name: &str) -> std::path::PathBuf {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("work")
            .join(format!("test-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
    #[test]
    fn noise_is_bipolar_full_scale_and_unbiased() {
        let mut rng = 41;
        let mut sum = 0.0;
        let mut sq = 0.0;
        let mut neg = 0;
        for _ in 0..100000 {
            let v = next_noise(&mut rng) as f64;
            sum += v;
            sq += v * v;
            if v < 0.0 {
                neg += 1;
            }
        }
        assert!((sum / 100000.0).abs() < 0.01);
        assert!((sq / 100000.0f64).sqrt() > 0.55);
        assert!((48000..52000).contains(&neg));
    }
    #[test]
    fn roles_and_silent_layers_do_not_change_level() {
        let sound = serde_json::json!({"layers":[{"waveform":"saw","gain":0.5}],"cutoff_hz":3000,"sustain":0.7});
        let a = track("bass", sound.clone());
        let b = track("lead", sound.clone());
        assert_eq!(voice(&a, 16000).0, voice(&b, 16000).0);
        let mut c = a.clone();
        c.sound.layers.push(SynthLayer {
            gain: 0.0,
            ..Default::default()
        });
        assert_eq!(voice(&a, 16000).0, voice(&c, 16000).0);
    }
    #[test]
    fn stereo_unison_produces_native_side_information() {
        let t = track(
            "synth",
            serde_json::json!({"layers":[{"waveform":"saw","gain":0.5,"unison":4,"detune_cents":10}],"stereo_spread":0.8,"cutoff_hz":5000,"sustain":0.8}),
        );
        let (m, s) = voice(&t, 16000);
        let mid: f32 = m.iter().map(|x| x * x).sum();
        let side: f32 = s.iter().map(|x| x * x).sum();
        assert!(side / mid > 0.05);
    }
    #[test]
    fn short_gate_during_attack_does_not_swell_after_note_off() {
        let t = track(
            "synth",
            serde_json::json!({"waveform":"sine","attack_ms":50,"release_ms":200,"cutoff_hz":16000,"sustain":1}),
        );
        let (y, _) = voice(&t, 480);
        let rms = |a: usize, b: usize| y[a..b].iter().map(|x| x * x).sum::<f32>() / (b - a) as f32;
        assert!(rms(6000, 6500) < rms(1000, 1500));
    }
    #[test]
    fn downsampling_rejects_frequencies_above_destination_nyquist() {
        let source: Vec<f32> = (0..9600)
            .map(|i| (std::f64::consts::TAU * 20000.0 * i as f64 / 48000.0).sin() as f32)
            .collect();
        let energy: f32 = (100..4000)
            .map(|i| sample_value(&source, i as f64 * 2.0, 2.0).powi(2))
            .sum::<f32>()
            / 3900.0;
        assert!(energy.sqrt() < 0.03, "alias RMS {}", energy.sqrt());
    }
    #[test]
    fn floating_point_stereo_samples_keep_their_side_channel() {
        let dir = tempdir("stereo");
        let file = dir.join("sample.wav");
        let mut w = WavWriter::new(&file, 48000, true).unwrap();
        for i in 0..1000 {
            let l = (i as f32 * 0.1).sin() * 0.4;
            let r = (i as f32 * 0.2).sin() * 0.2;
            w.push(l, r).unwrap();
        }
        w.finish().unwrap();
        let a = read_pcm16_mono(&file).unwrap();
        for i in 0..1000 {
            let l = (i as f32 * 0.1).sin() * 0.4;
            let r = (i as f32 * 0.2).sin() * 0.2;
            assert!((a.mono[i] + a.side[i] - l).abs() < 1e-6);
            assert!((a.mono[i] - a.side[i] - r).abs() < 1e-6);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn bus_stems_and_room_reconstruct_premaster_without_double_routing() {
        let dir = tempdir("routing");
        let mut song = Song::fresh(
            "Routing probe".into(),
            128.0,
            "D".into(),
            "minor".into(),
            48000,
            41,
        );
        song.buses.push(serde_json::from_value(serde_json::json!({"id":"music","name":"Music","gain_db":-3,"effects":{"eq":[{"frequency_hz":1000,"gain_db":2}]}})).unwrap());
        let mut t = track(
            "synth",
            serde_json::json!({"waveform":"sine","cutoff_hz":8000,"release_ms":100}),
        );
        t.gain_db = -6.0;
        t.bus_id = Some("music".into());
        t.patterns.push(serde_json::from_value(serde_json::json!({"id":"p","bars":1,"notes":[{"bar":1,"beat":1,"pitch":57,"velocity":100,"length_beats":3.5}]})).unwrap());
        t.clips.push(
            serde_json::from_value(
                serde_json::json!({"id":"c","pattern_id":"p","start_bar":1,"bars":1}),
            )
            .unwrap(),
        );
        t.effects.reverb_send = 0.2;
        song.tracks.push(t);
        let path = dir.join("mix.wav");
        let stats = render_wav_options(&song, &path, Some(&dir.join("stems")), true).unwrap();
        assert_eq!(stats.frames, 90000);
        let mix = read_pcm16_mono(&path).unwrap();
        let bus = read_pcm16_mono(&dir.join("stems/bus-music.wav")).unwrap();
        let room = read_pcm16_mono(&dir.join("stems/return-room.wav")).unwrap();
        for i in 0..mix.mono.len() {
            assert!((mix.mono[i] - bus.mono[i] - room.mono[i]).abs() < 1e-6);
            assert!((mix.side[i] - bus.side[i] - room.side[i]).abs() < 1e-6);
        }
        // Mastered PCM24 export has the same duration after FIR + limiter latency compensation.
        let mastered = render_wav(&song, &dir.join("master.wav")).unwrap();
        assert_eq!(mastered.frames, 90000);
        assert!(mastered.peak > 0.001 && mastered.peak < 1.0);
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[cfg(test)]
mod routing_and_peak_regressions {
    use super::*;
    #[test]
    fn reconstructed_peak_limiter_catches_between_sample_peaks() {
        let mut limiter = LookaheadLimiter::new(48000, 5.0, -1.0, 80.0);
        let mut output = Vec::new();
        for i in 0..10000 {
            let x = (std::f32::consts::FRAC_PI_2 * i as f32 + std::f32::consts::FRAC_PI_4).sin();
            if let Some((l, _)) = limiter.process(x, x) {
                output.push(l);
            }
        }
        let mut meter = crate::dsp::ReconstructedPeak::new();
        let peak = output
            .iter()
            .map(|&x| meter.push(x, x))
            .fold(0.0f32, f32::max);
        assert!(
            peak <= crate::dsp::gain(-0.95),
            "reconstructed peak {} dBFS",
            20.0 * peak.log10()
        );
        assert!(
            limiter.max_reduction_db > 0.8,
            "sample-only detector would miss this reduction"
        );
    }
    #[test]
    fn forced_synthesis_respects_the_patch_on_a_kick_role() {
        let mut song = Song::fresh(
            "Kick probe".into(),
            128.0,
            "D".into(),
            "minor".into(),
            48000,
            41,
        );
        let t:Track=serde_json::from_value(serde_json::json!({"id":"k","name":"Kick probe","role":"kick","source_type":"synth","sound":{"waveform":"sine","phase_reset":true,"cutoff_hz":10000,"sustain":1},"patterns":[{"id":"p","bars":1,"notes":[{"bar":1,"beat":2,"pitch":57,"length_beats":0.5,"velocity":100}]}],"clips":[{"id":"c","pattern_id":"p","start_bar":1,"bars":1}]})).unwrap();
        song.tracks.push(t.clone());
        let samples = load_samples(&song).unwrap();
        assert!(samples.is_empty());
        let (mut tail, mut side) = (vec![0.0; 12000], vec![0.0; 12000]);
        let (m, _) = render_source(
            &t,
            0,
            &song,
            90000,
            &make_tables(),
            &samples,
            &mut tail,
            &mut side,
        );
        assert!(m[..22500].iter().all(|x| *x == 0.0));
        assert!(m[23000..30000].iter().any(|x| x.abs() > 0.05));
    }
}

#[cfg(test)]
mod limiter_continuity_regression {
    use super::*;
    #[test]
    fn lookahead_attack_avoids_a_single_step_in_sustained_audio() {
        let mut limiter = LookaheadLimiter::new(48000, 5.0, -1.0, 100.0);
        let mut previous = 1.0f32;
        let mut max_step = 0.0f32;
        for i in 0..10000 {
            let envelope = 0.15 + 1.25 * (-((i as f32 - 5000.0) / 45.0).powi(2)).exp();
            let x = envelope * (std::f32::consts::TAU * 1000.0 * i as f32 / 48000.0).sin();
            if limiter.process(x, x).is_some() {
                if i > 1000 {
                    max_step = max_step.max((previous - limiter.gain).abs());
                }
                previous = limiter.gain;
            }
        }
        assert!(limiter.max_reduction_db > 2.0);
        assert!(max_step < 0.025, "gain jumped {max_step}");
    }
}
