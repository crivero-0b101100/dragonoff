// Copyright (C) 2026 OpenAI
// This program is free software under the GNU GPL v2 or (at your option) any later version.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Song {
    pub schema_version: u32,
    pub title: String,
    pub bpm: f64,
    pub key: String,
    pub scale: String,
    pub numerator: u8,
    pub denominator: u8,
    pub sample_rate: u32,
    pub seed: u64,
    #[serde(default = "default_personality")]
    pub personality: String,
    #[serde(default)]
    pub master: MasteringSettings,
    #[serde(default)]
    pub sections: Vec<Section>,
    #[serde(default)]
    pub tracks: Vec<Track>,
    #[serde(default)]
    pub buses: Vec<MixBus>,
    #[serde(default)]
    pub reverb: ReverbSettings,
    /// Analysis provenance retained by projects reconstructed from audio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub analysis_provenance: Option<serde_json::Value>,
}

fn default_personality() -> String {
    "auto".into()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MasteringSettings {
    #[serde(default = "default_knee")]
    pub compressor_knee_db: f32,
    #[serde(default = "default_master_detector_highpass")]
    pub compressor_sidechain_highpass_hz: f32,
    #[serde(default)]
    pub eq: Vec<EqBand>,
    #[serde(default = "default_master_output_gain")]
    pub output_gain_db: f32,
    #[serde(default)]
    pub clipper_drive_db: f32,
    #[serde(default = "default_master_eq_highpass")]
    pub eq_highpass_hz: f32,
    #[serde(default)]
    pub eq_low_shelf_db: f32,
    #[serde(default = "default_master_eq_low_mid_hz")]
    pub eq_low_mid_hz: f32,
    #[serde(default)]
    pub eq_low_mid_db: f32,
    #[serde(default = "default_master_eq_low_mid_q")]
    pub eq_low_mid_q: f32,
    #[serde(default = "default_master_eq_presence_hz")]
    pub eq_presence_hz: f32,
    #[serde(default)]
    pub eq_presence_db: f32,
    #[serde(default = "default_master_eq_presence_q")]
    pub eq_presence_q: f32,
    #[serde(default)]
    pub eq_high_shelf_db: f32,
    #[serde(default = "default_master_stereo_width")]
    pub stereo_width: f32,
    #[serde(default)]
    pub low_band_width: f32,
    #[serde(default = "default_master_compressor_threshold")]
    pub compressor_threshold_db: f32,
    #[serde(default = "default_master_compressor_ratio")]
    pub compressor_ratio: f32,
    #[serde(default = "default_master_compressor_attack")]
    pub compressor_attack_ms: f32,
    #[serde(default = "default_master_compressor_release")]
    pub compressor_release_ms: f32,
    #[serde(default = "default_master_limiter_ceiling")]
    pub limiter_ceiling_db: f32,
    #[serde(default = "default_master_limiter_release")]
    pub limiter_release_ms: f32,
    #[serde(default = "default_master_lookahead")]
    pub limiter_lookahead_ms: f32,
}

impl Default for MasteringSettings {
    fn default() -> Self {
        Self {
            eq: Vec::new(),
            compressor_knee_db: 6.0,
            compressor_sidechain_highpass_hz: 90.0,
            output_gain_db: default_master_output_gain(),
            clipper_drive_db: 0.0,
            eq_highpass_hz: default_master_eq_highpass(),
            eq_low_shelf_db: 0.0,
            eq_low_mid_hz: default_master_eq_low_mid_hz(),
            eq_low_mid_db: 0.0,
            eq_low_mid_q: default_master_eq_low_mid_q(),
            eq_presence_hz: default_master_eq_presence_hz(),
            eq_presence_db: 0.0,
            eq_presence_q: default_master_eq_presence_q(),
            eq_high_shelf_db: 0.0,
            stereo_width: default_master_stereo_width(),
            low_band_width: 0.0,
            compressor_threshold_db: default_master_compressor_threshold(),
            compressor_ratio: default_master_compressor_ratio(),
            compressor_attack_ms: default_master_compressor_attack(),
            compressor_release_ms: default_master_compressor_release(),
            limiter_ceiling_db: default_master_limiter_ceiling(),
            limiter_release_ms: default_master_limiter_release(),
            limiter_lookahead_ms: default_master_lookahead(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Section {
    pub name: String,
    pub start_bar: u32,
    pub bars: u32,
    #[serde(default = "default_energy")]
    pub energy: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Track {
    #[serde(default = "default_source_type")]
    pub source_type: String,
    pub id: String,
    pub name: String,
    pub role: String,
    #[serde(default)]
    pub gain_db: f32,
    #[serde(default)]
    pub pan: f32,
    #[serde(default)]
    pub muted: bool,
    #[serde(default)]
    pub bus_id: Option<String>,
    #[serde(default)]
    pub effects: Effects,
    #[serde(default)]
    pub sound: SoundSettings,
    /// Optional raw, normalized VST parameters. The native renderer ignores
    /// this map; plugin backends apply it after the role's starting patch.
    #[serde(default)]
    pub plugin_parameters: BTreeMap<String, f32>,
    #[serde(default)]
    pub sample_id: Option<String>,
    #[serde(default)]
    pub sample_alt_id: Option<String>,
    /// Optional external WAV path paired with sample_id. Paths may be absolute,
    /// or relative to the engine's asset directory. External files are not
    /// copied into project bundles; keep their source and license alongside it.
    #[serde(default)]
    pub sample_path: Option<String>,
    #[serde(default)]
    pub sample_alt_path: Option<String>,
    #[serde(default)]
    pub sample_tune_semitones: f32,
    #[serde(default)]
    pub drum: DrumSettings,
    #[serde(default)]
    pub patterns: Vec<Pattern>,
    #[serde(default)]
    pub clips: Vec<Clip>,
    #[serde(default)]
    pub automation: Vec<AutomationPoint>,
}

/// Per-hit controls for sampled drum and percussion tracks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DrumSettings {
    /// Skip this many milliseconds at the start of the source sample.
    #[serde(default)]
    pub sample_start_ms: f32,
    /// Limit playback to this many milliseconds, with a short end fade; zero keeps the full sample.
    #[serde(default)]
    pub decay_ms: f32,
    /// Add a short gain accent to the first milliseconds of the hit (0..2).
    #[serde(default)]
    pub transient: f32,
    /// Add a short noise transient for extra snare, hat, or impact bite (0..1).
    #[serde(default)]
    pub noise_mix: f32,
}

impl Default for DrumSettings {
    fn default() -> Self {
        Self {
            sample_start_ms: 0.0,
            decay_ms: 0.0,
            transient: 0.0,
            noise_mix: 0.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoundSettings {
    #[serde(default)]
    pub phase_reset: bool,
    /// Merge adjacent same-pitch notes into a held voice before synthesis.
    #[serde(default)]
    pub legato: bool,
    #[serde(default)]
    pub pitch_envelope_semitones: f32,
    #[serde(default = "default_pitch_decay")]
    pub pitch_decay_ms: f32,
    /// Explicit trim shared by all roles; no hidden role-dependent gain.
    #[serde(default)]
    pub output_gain_db: f32,
    #[serde(default)]
    pub stereo_spread: f32,
    /// Equal-power stereo orbit depth; 0 is fixed, 1 reaches the sides.
    #[serde(default)]
    pub stereo_motion_depth: f32,
    #[serde(default)]
    pub stereo_motion_rate_hz: f32,
    #[serde(default)]
    pub stereo_motion_phase_radians: f32,
    /// Maximum far-ear interaural delay for headphone-oriented orbit panning.
    #[serde(default)]
    pub binaural_itd_ms: f32,
    #[serde(default = "default_waveform")]
    pub waveform: String,
    #[serde(default = "default_unison")]
    pub unison: u8,
    #[serde(default)]
    pub detune_cents: f32,
    #[serde(default = "default_cutoff")]
    pub cutoff_hz: f32,
    #[serde(default = "default_attack")]
    pub attack_ms: f32,
    #[serde(default = "default_release")]
    pub release_ms: f32,
    #[serde(default = "default_decay")]
    pub decay_ms: f32,
    #[serde(default = "default_sustain")]
    pub sustain: f32,
    #[serde(default)]
    pub filter_resonance: f32,
    #[serde(default)]
    pub filter_env_amount: f32,
    #[serde(default = "default_filter_decay")]
    pub filter_decay_ms: f32,
    #[serde(default = "default_filter_mode")]
    pub filter_mode: String,
    #[serde(default)]
    pub lfo_rate_hz: f32,
    #[serde(default)]
    pub lfo_depth: f32,
    /// DC bias applied only to pitch LFO, shifting the cycle toward up or down bends.
    #[serde(default)]
    pub lfo_bias: f32,
    #[serde(default = "default_lfo_target")]
    pub lfo_target: String,
    #[serde(default)]
    pub layers: Vec<SynthLayer>,
}

/// An independently controllable oscillator layer inside a track's instrument.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynthLayer {
    #[serde(default)]
    pub tune_semitones: f32,
    #[serde(default = "default_fm_ratio")]
    pub fm_ratio: f32,
    #[serde(default)]
    pub fm_decay_ms: f32,
    #[serde(default)]
    pub pulse_width_depth: f32,
    #[serde(default)]
    pub pulse_width_rate_hz: f32,
    #[serde(default = "default_waveform")]
    pub waveform: String,
    #[serde(default)]
    pub octave: i8,
    #[serde(default = "default_layer_gain")]
    pub gain: f32,
    #[serde(default)]
    pub detune_cents: f32,
    #[serde(default = "default_unison")]
    pub unison: u8,
    #[serde(default = "default_pulse_width")]
    pub pulse_width: f32,
    #[serde(default)]
    pub fm_amount: f32,
    #[serde(default)]
    pub noise_amount: f32,
    /// Optional per-layer envelope/filter overrides; omitted values inherit the track patch.
    #[serde(default)]
    pub pan: f32,
    #[serde(default)]
    pub cutoff_hz: Option<f32>,
    #[serde(default)]
    pub filter_resonance: Option<f32>,
    #[serde(default)]
    pub attack_ms: Option<f32>,
    #[serde(default)]
    pub decay_ms: Option<f32>,
    #[serde(default)]
    pub sustain: Option<f32>,
    #[serde(default)]
    pub release_ms: Option<f32>,
}

impl Default for SynthLayer {
    fn default() -> Self {
        Self {
            tune_semitones: 0.0,
            fm_ratio: 2.0,
            fm_decay_ms: 0.0,
            pulse_width_depth: 0.0,
            pulse_width_rate_hz: 0.0,
            waveform: default_waveform(),
            octave: 0,
            gain: default_layer_gain(),
            detune_cents: 0.0,
            unison: 1,
            pulse_width: default_pulse_width(),
            fm_amount: 0.0,
            noise_amount: 0.0,
            pan: 0.0,
            cutoff_hz: None,
            filter_resonance: None,
            attack_ms: None,
            decay_ms: None,
            sustain: None,
            release_ms: None,
        }
    }
}

impl SoundSettings {
    pub fn for_role(role: &str) -> Self {
        match role {
            "sub" => Self {
                waveform: "sine".into(),
                unison: 1,
                detune_cents: 0.0,
                cutoff_hz: 180.0,
                attack_ms: 3.0,
                release_ms: 90.0,
                ..Self::default()
            },
            "bass" => Self {
                // Stable analog-style body: remove detuned PWM/FM brightness
                // from the default bass and keep its envelope connected.
                waveform: "sine".into(),
                unison: 1,
                detune_cents: 0.0,
                cutoff_hz: 760.0,
                attack_ms: 6.0,
                decay_ms: 320.0,
                sustain: 0.72,
                release_ms: 125.0,
                filter_resonance: 0.06,
                filter_env_amount: 0.08,
                filter_decay_ms: 700.0,
                layers: vec![
                    SynthLayer {
                        waveform: "saw".into(),
                        gain: 0.38,
                        unison: 1,
                        ..SynthLayer::default()
                    },
                    SynthLayer {
                        waveform: "triangle".into(),
                        gain: 0.22,
                        unison: 1,
                        ..SynthLayer::default()
                    },
                ],
                ..Self::default()
            },
            "pipe_bass" => Self {
                waveform: "triangle".into(),
                unison: 1,
                detune_cents: 0.0,
                cutoff_hz: 520.0,
                attack_ms: 3.0,
                release_ms: 75.0,
                ..Self::default()
            },
            "brass_bass" => Self {
                waveform: "brass".into(),
                unison: 2,
                detune_cents: 2.0,
                cutoff_hz: 1150.0,
                attack_ms: 5.0,
                release_ms: 130.0,
                ..Self::default()
            },
            "fm_bass" => Self {
                waveform: "fm".into(),
                unison: 1,
                detune_cents: 0.0,
                cutoff_hz: 2600.0,
                attack_ms: 1.0,
                release_ms: 55.0,
                ..Self::default()
            },
            "piano" => Self {
                waveform: "sine".into(),
                unison: 1,
                detune_cents: 0.0,
                cutoff_hz: 18_000.0,
                attack_ms: 5.0,
                release_ms: 2200.0,
                ..Self::default()
            },
            "flute" => Self {
                waveform: "sine".into(),
                unison: 1,
                detune_cents: 0.0,
                cutoff_hz: 12_000.0,
                attack_ms: 12.0,
                release_ms: 140.0,
                ..Self::default()
            },
            "steel_guitar" => Self {
                waveform: "triangle".into(),
                unison: 1,
                cutoff_hz: 12_000.0,
                attack_ms: 1.0,
                release_ms: 240.0,
                ..Self::default()
            },
            "chords" => Self {
                waveform: "saw".into(),
                unison: 2,
                detune_cents: 3.0,
                cutoff_hz: 2400.0,
                attack_ms: 24.0,
                decay_ms: 420.0,
                sustain: 0.52,
                release_ms: 360.0,
                ..Self::default()
            },
            "arp" => Self {
                waveform: "saw".into(),
                unison: 2,
                detune_cents: 4.0,
                cutoff_hz: 6000.0,
                attack_ms: 2.0,
                release_ms: 80.0,
                ..Self::default()
            },
            "lead" => Self {
                // A mono-friendly, band-limited saw voice with a short gate
                // keeps new leads punchy instead of drifting into bell-like
                // sine/FM tails. Producers can add extra layers or sustain
                // explicitly when the arrangement calls for it.
                waveform: "saw".into(),
                unison: 2,
                detune_cents: 3.5,
                cutoff_hz: 1750.0,
                attack_ms: 7.0,
                decay_ms: 280.0,
                sustain: 0.38,
                release_ms: 135.0,
                filter_resonance: 0.07,
                filter_env_amount: 0.22,
                filter_decay_ms: 420.0,
                lfo_rate_hz: 0.13,
                lfo_depth: 0.06,
                layers: vec![
                    SynthLayer {
                        waveform: "saw".into(),
                        gain: 0.52,
                        unison: 1,
                        detune_cents: 0.0,
                        ..SynthLayer::default()
                    },
                    SynthLayer {
                        waveform: "triangle".into(),
                        gain: 0.18,
                        octave: 0,
                        ..SynthLayer::default()
                    },
                ],
                ..Self::default()
            },
            "pad" => Self {
                waveform: "triangle".into(),
                unison: 5,
                detune_cents: 14.0,
                cutoff_hz: 5000.0,
                attack_ms: 140.0,
                release_ms: 650.0,
                ..Self::default()
            },
            "texture" => Self {
                waveform: "noise".into(),
                unison: 1,
                detune_cents: 0.0,
                cutoff_hz: 8000.0,
                attack_ms: 30.0,
                release_ms: 320.0,
                ..Self::default()
            },
            _ => Self::default(),
        }
    }
}

impl Default for SoundSettings {
    fn default() -> Self {
        Self {
            output_gain_db: 0.0,
            stereo_spread: 0.0,
            stereo_motion_depth: 0.0,
            stereo_motion_rate_hz: 0.0,
            stereo_motion_phase_radians: 0.0,
            binaural_itd_ms: 0.0,
            phase_reset: false,
            legato: false,
            pitch_envelope_semitones: 0.0,
            pitch_decay_ms: 80.0,
            waveform: default_waveform(),
            unison: default_unison(),
            detune_cents: 0.0,
            cutoff_hz: default_cutoff(),
            attack_ms: default_attack(),
            release_ms: default_release(),
            decay_ms: default_decay(),
            sustain: default_sustain(),
            filter_resonance: 0.0,
            filter_env_amount: 0.0,
            filter_decay_ms: default_filter_decay(),
            filter_mode: default_filter_mode(),
            lfo_rate_hz: 0.0,
            lfo_depth: 0.0,
            lfo_bias: 0.0,
            lfo_target: default_lfo_target(),
            layers: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Effects {
    #[serde(default)]
    pub eq: Vec<EqBand>,
    #[serde(default)]
    pub sidechain_source: Option<String>,
    #[serde(default = "default_sidechain_threshold")]
    pub sidechain_threshold_db: f32,
    #[serde(default = "default_knee")]
    pub compressor_knee_db: f32,
    #[serde(default)]
    pub compressor_makeup_db: f32,
    #[serde(default)]
    pub compressor_sidechain_highpass_hz: f32,
    #[serde(default)]
    pub lowpass_hz: Option<f32>,
    #[serde(default)]
    pub highpass_hz: Option<f32>,
    #[serde(default)]
    pub saturation: f32,
    #[serde(default = "default_saturation_mode")]
    pub saturation_mode: String,
    #[serde(default)]
    pub delay_ms: f32,
    #[serde(default)]
    pub delay_beats: Option<f32>,
    #[serde(default)]
    pub delay_feedback: f32,
    #[serde(default)]
    pub delay_wet: f32,
    #[serde(default)]
    pub reverb_send: f32,
    #[serde(default)]
    pub sidechain: bool,
    #[serde(default = "default_sidechain_amount")]
    pub sidechain_amount: f32,
    #[serde(default = "default_sidechain_release_ms")]
    pub sidechain_release_ms: f32,
    #[serde(default)]
    pub chorus_depth: f32,
    #[serde(default = "default_chorus_rate")]
    pub chorus_rate_hz: f32,
    #[serde(default = "default_compressor_threshold")]
    pub compressor_threshold_db: f32,
    #[serde(default = "default_compressor_ratio")]
    pub compressor_ratio: f32,
    #[serde(default = "default_compressor_attack")]
    pub compressor_attack_ms: f32,
    #[serde(default = "default_compressor_release")]
    pub compressor_release_ms: f32,
}

impl Default for Effects {
    fn default() -> Self {
        Self {
            eq: Vec::new(),
            sidechain_source: None,
            sidechain_threshold_db: -18.0,
            compressor_knee_db: 6.0,
            compressor_makeup_db: 0.0,
            compressor_sidechain_highpass_hz: 0.0,
            lowpass_hz: None,
            highpass_hz: None,
            saturation: 0.0,
            saturation_mode: default_saturation_mode(),
            delay_ms: 0.0,
            delay_beats: None,
            delay_feedback: 0.0,
            delay_wet: 0.0,
            reverb_send: 0.0,
            sidechain: false,
            sidechain_amount: default_sidechain_amount(),
            sidechain_release_ms: default_sidechain_release_ms(),
            chorus_depth: 0.0,
            chorus_rate_hz: default_chorus_rate(),
            compressor_threshold_db: default_compressor_threshold(),
            compressor_ratio: default_compressor_ratio(),
            compressor_attack_ms: default_compressor_attack(),
            compressor_release_ms: default_compressor_release(),
        }
    }
}

/// A parallel stereo subgroup. Tracks route to exactly one bus, then buses sum to master.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MixBus {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub gain_db: f32,
    #[serde(default)]
    pub effects: Effects,
    #[serde(default = "default_unity")]
    pub stereo_width: f32,
}

/// Parametric EQ, optionally level-dependent. Dynamic reduction affects only this band.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EqBand {
    #[serde(default = "default_eq_kind")]
    pub kind: String,
    pub frequency_hz: f32,
    #[serde(default = "default_eq_q")]
    pub q: f32,
    #[serde(default)]
    pub gain_db: f32,
    #[serde(default)]
    pub dynamic_threshold_db: Option<f32>,
    #[serde(default = "default_dynamic_ratio")]
    pub dynamic_ratio: f32,
    #[serde(default = "default_dynamic_range")]
    pub dynamic_range_db: f32,
    #[serde(default = "default_compressor_attack")]
    pub attack_ms: f32,
    #[serde(default = "default_compressor_release")]
    pub release_ms: f32,
}
fn default_unity() -> f32 {
    1.0
}
fn default_fm_ratio() -> f32 {
    2.0
}
fn default_eq_kind() -> String {
    "bell".into()
}
fn default_eq_q() -> f32 {
    0.70710677
}
fn default_dynamic_ratio() -> f32 {
    2.0
}
fn default_dynamic_range() -> f32 {
    4.0
}
fn default_sidechain_threshold() -> f32 {
    -18.0
}
fn default_knee() -> f32 {
    6.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pattern {
    pub id: String,
    pub bars: u32,
    #[serde(default)]
    pub notes: Vec<Note>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Clip {
    pub id: String,
    pub pattern_id: String,
    pub start_bar: u32,
    pub bars: u32,
    #[serde(default)]
    pub transpose: i16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Note {
    /// One-based bar within the pattern, following Ardour's bulk MIDI JSON convention.
    pub bar: u32,
    /// One-based beat within the bar; fractional beats are allowed.
    pub beat: f64,
    #[serde(default)]
    pub tick: u32,
    pub pitch: u8,
    pub length_beats: f64,
    #[serde(default = "default_velocity")]
    pub velocity: u8,
    #[serde(default = "default_channel")]
    pub channel: u8,
    /// Optional event-level synthesis controls for sound-by-sound reconstruction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synth_overrides: Option<NoteSynthOverrides>,
    /// Source-analysis evidence and uncertainty preserved through reconstruction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub analysis_evidence: Option<serde_json::Value>,
}

/// Per-note controls let transient and bass events vary without splitting a
/// musical part into one track per sound variation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NoteSynthOverrides {
    #[serde(default)]
    pub waveform: Option<String>,
    #[serde(default)]
    pub pitch_envelope_semitones: Option<f32>,
    #[serde(default)]
    pub pitch_decay_ms: Option<f32>,
    #[serde(default)]
    pub attack_ms: Option<f32>,
    #[serde(default)]
    pub decay_ms: Option<f32>,
    #[serde(default)]
    pub sustain: Option<f32>,
    #[serde(default)]
    pub release_ms: Option<f32>,
    #[serde(default)]
    pub cutoff_hz: Option<f32>,
    #[serde(default)]
    pub filter_env_amount: Option<f32>,
    #[serde(default)]
    pub noise_amount: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutomationPoint {
    #[serde(default)]
    pub parameter: String,
    pub bar: u32,
    pub beat: f64,
    pub value: f32,
}

fn default_energy() -> f32 {
    0.5
}
fn default_velocity() -> u8 {
    96
}
fn default_sidechain_amount() -> f32 {
    0.46
}
fn default_sidechain_release_ms() -> f32 {
    77.0
}
fn default_channel() -> u8 {
    1
}
fn default_waveform() -> String {
    "saw".into()
}
fn default_unison() -> u8 {
    1
}
fn default_cutoff() -> f32 {
    20_000.0
}
fn default_attack() -> f32 {
    8.0
}
fn default_release() -> f32 {
    100.0
}
fn default_decay() -> f32 {
    180.0
}
fn default_sustain() -> f32 {
    0.72
}
fn default_filter_decay() -> f32 {
    240.0
}
fn default_filter_mode() -> String {
    "lowpass".into()
}
fn default_saturation_mode() -> String {
    "soft_clip".into()
}
fn default_chorus_rate() -> f32 {
    0.23
}
fn default_lfo_target() -> String {
    "cutoff".into()
}
fn default_layer_gain() -> f32 {
    1.0
}
fn default_pulse_width() -> f32 {
    0.5
}
fn default_compressor_threshold() -> f32 {
    -6.0
}
fn default_compressor_ratio() -> f32 {
    1.0
}
fn default_compressor_attack() -> f32 {
    10.0
}
fn default_compressor_release() -> f32 {
    120.0
}
fn default_master_output_gain() -> f32 {
    0.0
}
fn default_master_eq_highpass() -> f32 {
    24.0
}
fn default_master_eq_low_mid_hz() -> f32 {
    280.0
}
fn default_master_eq_low_mid_q() -> f32 {
    0.85
}
fn default_master_eq_presence_hz() -> f32 {
    3600.0
}
fn default_master_eq_presence_q() -> f32 {
    0.75
}
fn default_master_stereo_width() -> f32 {
    1.0
}
fn default_master_compressor_threshold() -> f32 {
    -8.0
}
fn default_master_compressor_ratio() -> f32 {
    1.0
}
fn default_master_compressor_attack() -> f32 {
    25.0
}
fn default_master_compressor_release() -> f32 {
    160.0
}
fn default_master_limiter_ceiling() -> f32 {
    -1.0
}
fn default_master_limiter_release() -> f32 {
    100.0
}
fn default_master_lookahead() -> f32 {
    5.0
}

impl Song {
    pub fn fresh(
        title: String,
        bpm: f64,
        key: String,
        scale: String,
        sample_rate: u32,
        seed: u64,
    ) -> Self {
        Self {
            schema_version: 2,
            title,
            bpm,
            key,
            scale,
            numerator: 4,
            denominator: 4,
            sample_rate,
            seed,
            personality: "auto".into(),
            master: MasteringSettings::default(),
            sections: Vec::new(),
            tracks: Vec::new(),
            buses: Vec::new(),
            reverb: ReverbSettings::default(),
            analysis_provenance: None,
        }
    }

    pub fn total_bars(&self) -> u32 {
        self.sections
            .iter()
            .map(|s| s.start_bar.saturating_add(s.bars).saturating_sub(1))
            .chain(
                self.tracks
                    .iter()
                    .flat_map(|t| &t.clips)
                    .map(|c| c.start_bar.saturating_add(c.bars).saturating_sub(1)),
            )
            .max()
            .unwrap_or(1)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ReverbSettings {
    pub decay_seconds: f32,
    pub predelay_ms: f32,
    pub damping_hz: f32,
    pub highpass_hz: f32,
    pub lowpass_hz: f32,
    pub return_gain_db: f32,
}
impl Default for ReverbSettings {
    fn default() -> Self {
        Self {
            decay_seconds: 1.45,
            predelay_ms: 17.0,
            damping_hz: 5200.0,
            highpass_hz: 280.0,
            lowpass_hz: 7200.0,
            return_gain_db: -12.4,
        }
    }
}
fn default_master_detector_highpass() -> f32 {
    90.0
}

fn default_pitch_decay() -> f32 {
    80.0
}

fn default_source_type() -> String {
    "auto".into()
}

pub fn validate_sound(sound: &SoundSettings) -> Result<(), String> {
    if ![
        "sine", "saw", "square", "triangle", "choir", "brass", "fm", "noise",
    ]
    .contains(&sound.waveform.as_str())
    {
        return Err(
            "waveform must be sine, saw, square, triangle, choir, brass, fm or noise".into(),
        );
    }
    let valid_wave = |waveform: &str| {
        [
            "sine", "saw", "square", "triangle", "choir", "brass", "fm", "noise",
        ]
        .contains(&waveform)
    };
    if !sound.pitch_envelope_semitones.is_finite()
        || !(-24.0..=48.0).contains(&sound.pitch_envelope_semitones)
        || !sound.pitch_decay_ms.is_finite()
        || !(1.0..=5000.0).contains(&sound.pitch_decay_ms)
    {
        return Err("invalid pitch envelope".into());
    }
    if !sound.output_gain_db.is_finite()
        || !(-36.0..=18.0).contains(&sound.output_gain_db)
        || !sound.stereo_spread.is_finite()
        || !(0.0..=1.0).contains(&sound.stereo_spread)
        || !sound.stereo_motion_depth.is_finite()
        || !(0.0..=1.0).contains(&sound.stereo_motion_depth)
        || !sound.stereo_motion_rate_hz.is_finite()
        || !(0.0..=8.0).contains(&sound.stereo_motion_rate_hz)
        || !sound.stereo_motion_phase_radians.is_finite()
        || !(-std::f32::consts::TAU..=std::f32::consts::TAU)
            .contains(&sound.stereo_motion_phase_radians)
        || !sound.binaural_itd_ms.is_finite()
        || !(0.0..=1.2).contains(&sound.binaural_itd_ms)
    {
        return Err("invalid output trim or stereo spread".into());
    }
    for l in &sound.layers {
        if !l.tune_semitones.is_finite()
            || !(-24.0..=24.0).contains(&l.tune_semitones)
            || !l.fm_ratio.is_finite()
            || !(0.25..=12.0).contains(&l.fm_ratio)
            || !l.fm_decay_ms.is_finite()
            || !(0.0..=10000.0).contains(&l.fm_decay_ms)
            || !l.pulse_width_depth.is_finite()
            || !(0.0..=0.4).contains(&l.pulse_width_depth)
            || !l.pulse_width_rate_hz.is_finite()
            || !(0.0..=20.0).contains(&l.pulse_width_rate_hz)
        {
            return Err("invalid FM, tuning or PWM controls".into());
        }
    }
    if !(1..=8).contains(&sound.unison)
        || !sound.detune_cents.is_finite()
        || !(0.0..=50.0).contains(&sound.detune_cents)
        || !sound.cutoff_hz.is_finite()
        || !(20.0..=40_000.0).contains(&sound.cutoff_hz)
        || !sound.attack_ms.is_finite()
        || !(0.0..=5000.0).contains(&sound.attack_ms)
        || !sound.release_ms.is_finite()
        || !(0.0..=10_000.0).contains(&sound.release_ms)
        || !sound.decay_ms.is_finite()
        || !(0.0..=10_000.0).contains(&sound.decay_ms)
        || !sound.sustain.is_finite()
        || !(0.0..=1.0).contains(&sound.sustain)
        || !["lowpass", "highpass", "bandpass", "notch", "allpass"]
            .contains(&sound.filter_mode.as_str())
        || !sound.filter_resonance.is_finite()
        || !(0.0..=0.95).contains(&sound.filter_resonance)
        || !sound.filter_env_amount.is_finite()
        || !(-4.0..=4.0).contains(&sound.filter_env_amount)
        || !sound.filter_decay_ms.is_finite()
        || !(1.0..=10_000.0).contains(&sound.filter_decay_ms)
        || !sound.lfo_rate_hz.is_finite()
        || !(0.0..=30.0).contains(&sound.lfo_rate_hz)
        || !sound.lfo_depth.is_finite()
        || !(0.0..=4.0).contains(&sound.lfo_depth)
        || !sound.lfo_bias.is_finite()
        || !(-1.0..=1.0).contains(&sound.lfo_bias)
        || !["cutoff", "amplitude", "pitch"].contains(&sound.lfo_target.as_str())
        || sound.layers.len() > 8
        || sound
            .layers
            .iter()
            .map(|layer| layer.unison as usize)
            .sum::<usize>()
            > 32
        || sound.layers.iter().any(|layer| {
            !valid_wave(&layer.waveform)
                || !(-3..=3).contains(&layer.octave)
                || !layer.gain.is_finite()
                || !(0.0..=2.0).contains(&layer.gain)
                || !layer.detune_cents.is_finite()
                || !(0.0..=50.0).contains(&layer.detune_cents)
                || !(1..=8).contains(&layer.unison)
                || !layer.pulse_width.is_finite()
                || !(0.05..=0.95).contains(&layer.pulse_width)
                || !layer.fm_amount.is_finite()
                || !(0.0..=1.0).contains(&layer.fm_amount)
                || !layer.noise_amount.is_finite()
                || !(0.0..=1.0).contains(&layer.noise_amount)
                || !layer.pan.is_finite()
                || !(-1.0..=1.0).contains(&layer.pan)
                || layer
                    .cutoff_hz
                    .is_some_and(|v| !v.is_finite() || !(20.0..=40_000.0).contains(&v))
                || layer
                    .filter_resonance
                    .is_some_and(|v| !v.is_finite() || !(0.0..=0.95).contains(&v))
                || layer
                    .attack_ms
                    .is_some_and(|v| !v.is_finite() || !(0.0..=5000.0).contains(&v))
                || layer
                    .decay_ms
                    .is_some_and(|v| !v.is_finite() || !(0.0..=10_000.0).contains(&v))
                || layer
                    .sustain
                    .is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
                || layer
                    .release_ms
                    .is_some_and(|v| !v.is_finite() || !(0.0..=10_000.0).contains(&v))
        })
    {
        return Err(
            "sound patch controls or oscillator layers are outside their supported ranges".into(),
        );
    }
    Ok(())
}

pub fn validate_note_synth_overrides(overrides: &NoteSynthOverrides) -> Result<(), String> {
    if overrides.waveform.as_ref().is_some_and(|waveform| {
        ![
            "sine", "saw", "square", "triangle", "choir", "brass", "fm", "noise",
        ]
        .contains(&waveform.as_str())
    }) || overrides
        .pitch_envelope_semitones
        .is_some_and(|value| !value.is_finite() || !(-24.0..=48.0).contains(&value))
        || overrides
            .pitch_decay_ms
            .is_some_and(|value| !value.is_finite() || !(1.0..=5000.0).contains(&value))
        || overrides
            .attack_ms
            .is_some_and(|value| !value.is_finite() || !(0.0..=5000.0).contains(&value))
        || overrides
            .decay_ms
            .is_some_and(|value| !value.is_finite() || !(0.0..=10000.0).contains(&value))
        || overrides
            .sustain
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        || overrides
            .release_ms
            .is_some_and(|value| !value.is_finite() || !(0.0..=10000.0).contains(&value))
        || overrides
            .cutoff_hz
            .is_some_and(|value| !value.is_finite() || !(20.0..=40000.0).contains(&value))
        || overrides
            .filter_env_amount
            .is_some_and(|value| !value.is_finite() || !(-4.0..=4.0).contains(&value))
        || overrides
            .noise_amount
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err("note synth overrides contain an unsupported value".into());
    }
    Ok(())
}

pub fn validate_drum(drum: &DrumSettings) -> Result<(), String> {
    if !drum.sample_start_ms.is_finite()
        || !(0.0..=500.0).contains(&drum.sample_start_ms)
        || !drum.decay_ms.is_finite()
        || !(0.0..=5000.0).contains(&drum.decay_ms)
        || !drum.transient.is_finite()
        || !(0.0..=2.0).contains(&drum.transient)
        || !drum.noise_mix.is_finite()
        || !(0.0..=1.0).contains(&drum.noise_mix)
    {
        return Err("drum controls are outside their supported ranges".into());
    }
    Ok(())
}

pub fn validate_effects(effects: &Effects) -> Result<(), String> {
    validate_eq(&effects.eq)?;
    for (value, min, max) in [
        (effects.compressor_knee_db, 0.0, 24.0),
        (effects.compressor_makeup_db, -24.0, 24.0),
        (effects.compressor_sidechain_highpass_hz, 0.0, 1000.0),
        (effects.sidechain_threshold_db, -60.0, 0.0),
    ] {
        if !value.is_finite() || value < min || value > max {
            return Err("invalid dynamics controls".into());
        }
    }

    if effects
        .lowpass_hz
        .is_some_and(|v| !v.is_finite() || !(20.0..=40_000.0).contains(&v))
        || effects
            .highpass_hz
            .is_some_and(|v| !v.is_finite() || !(20.0..=20_000.0).contains(&v))
        || !effects.saturation.is_finite()
        || !(0.0..=1.0).contains(&effects.saturation)
        || !["soft_clip", "hard_clip", "foldback"].contains(&effects.saturation_mode.as_str())
        || !effects.delay_ms.is_finite()
        || !(0.0..=1200.0).contains(&effects.delay_ms)
        || effects
            .delay_beats
            .is_some_and(|v| !v.is_finite() || !(0.125..=2.0).contains(&v))
        || !effects.delay_feedback.is_finite()
        || !(0.0..=0.85).contains(&effects.delay_feedback)
        || !effects.delay_wet.is_finite()
        || !(0.0..=0.8).contains(&effects.delay_wet)
        || !effects.reverb_send.is_finite()
        || !(0.0..=0.8).contains(&effects.reverb_send)
        || !effects.chorus_depth.is_finite()
        || !(0.0..=1.0).contains(&effects.chorus_depth)
        || !effects.chorus_rate_hz.is_finite()
        || !(0.02..=8.0).contains(&effects.chorus_rate_hz)
        || !effects.sidechain_amount.is_finite()
        || !(0.0..=0.95).contains(&effects.sidechain_amount)
        || !effects.sidechain_release_ms.is_finite()
        || !(5.0..=2000.0).contains(&effects.sidechain_release_ms)
        || !effects.compressor_threshold_db.is_finite()
        || !(-60.0..=0.0).contains(&effects.compressor_threshold_db)
        || !effects.compressor_ratio.is_finite()
        || !(1.0..=20.0).contains(&effects.compressor_ratio)
        || !effects.compressor_attack_ms.is_finite()
        || !(0.1..=100.0).contains(&effects.compressor_attack_ms)
        || !effects.compressor_release_ms.is_finite()
        || !(5.0..=2000.0).contains(&effects.compressor_release_ms)
    {
        return Err("effect controls are outside their supported ranges".into());
    }
    Ok(())
}

pub fn validate_eq(bands: &[EqBand]) -> Result<(), String> {
    if bands.len() > 12 {
        return Err("at most 12 EQ bands per processor".into());
    }
    for b in bands {
        if ![
            "bell",
            "low_shelf",
            "high_shelf",
            "notch",
            "highpass",
            "lowpass",
        ]
        .contains(&b.kind.as_str())
            || !b.frequency_hz.is_finite()
            || !(10.0..=40000.0).contains(&b.frequency_hz)
            || !b.q.is_finite()
            || !(0.1..=16.0).contains(&b.q)
            || !b.gain_db.is_finite()
            || !(-24.0..=24.0).contains(&b.gain_db)
            || b.dynamic_threshold_db
                .is_some_and(|v| !v.is_finite() || !(-72.0..=0.0).contains(&v))
            || !b.dynamic_ratio.is_finite()
            || !(1.0..=12.0).contains(&b.dynamic_ratio)
            || !b.dynamic_range_db.is_finite()
            || !(0.0..=12.0).contains(&b.dynamic_range_db)
            || !b.attack_ms.is_finite()
            || !(0.1..=200.0).contains(&b.attack_ms)
            || !b.release_ms.is_finite()
            || !(5.0..=2000.0).contains(&b.release_ms)
        {
            return Err("invalid EQ band".into());
        }
        if b.dynamic_threshold_db.is_some() && b.kind != "bell" {
            return Err("dynamic EQ currently supports bell bands".into());
        }
    }
    Ok(())
}
