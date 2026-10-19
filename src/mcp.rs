// Copyright (C) 2026 OpenAI
// This program is free software under the GNU GPL v2 or (at your option) any later version.

use crate::analysis;
use crate::audio;
use crate::midi::parse_bulk_notes;
use crate::model::{
    AutomationPoint, Clip, DrumSettings, Effects, EqBand, MixBus, Pattern, Section, Song,
    SoundSettings, Track, validate_drum, validate_effects, validate_eq, validate_sound,
};
use crate::render;
use serde_json::{Value, json};
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn serve() -> Result<(), String> {
    let stdin = io::stdin();
    let mut active: Option<PathBuf> = std::env::var_os("DRAGONOFF_SESSION").map(PathBuf::from);
    for line in stdin.lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let request: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("invalid MCP JSON: {e}");
                continue;
            }
        };
        let has_id = request.get("id").is_some();
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        let method = request.get("method").and_then(Value::as_str).unwrap_or("");
        let params = request.get("params").cloned().unwrap_or_else(|| json!({}));
        let response = match method {
            "initialize" => Some(json!({
                "jsonrpc":"2.0", "id":id,
                "result":{
                    "protocolVersion":"2025-03-26",
                    "capabilities":{"tools":{"listChanged":false}},
                    "serverInfo":{"name":"dragonoff","version":env!("CARGO_PKG_VERSION")},
                    "instructions":"Headless composition and rendering engine. Generated arrangements use four explicit, independently clipped drum lanes (kick, closed hat, backbeat, shaker), plus bass, a foreground lead, a distinct response synth, and a syncopated chord riff. Lead layers use independent clips and effect automation. Rendering plays only written note events. Evaluation is technical diagnostics only; do not use scores as a musical-quality target. Arrangement generation is an editable starting point. Session files are ordinary JSON projects."
                }
            })),
            "notifications/initialized" => None,
            "ping" => Some(json!({"jsonrpc":"2.0","id":id,"result":{}})),
            "tools/list" => {
                Some(json!({"jsonrpc":"2.0","id":id,"result":{"tools":tool_definitions()}}))
            }
            "tools/call" => {
                let tool_name = params.get("name").and_then(Value::as_str).unwrap_or("");
                let args = params
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                match call_tool(tool_name, &args, &mut active) {
                    Ok(value) => Some(
                        json!({"jsonrpc":"2.0","id":id,"result":{"content":[{"type":"text","text":serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".into())}],"structuredContent":value,"isError":false}}),
                    ),
                    Err(error) => Some(
                        json!({"jsonrpc":"2.0","id":id,"result":{"content":[{"type":"text","text":error}],"isError":true}}),
                    ),
                }
            }
            _ if has_id => Some(
                json!({"jsonrpc":"2.0","id":id,"error":{"code":-32601,"message":format!("unknown method: {method}")}}),
            ),
            _ => None,
        };
        if let Some(response) = response {
            println!("{}", response);
            io::stdout().flush().map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn tool_definitions() -> Vec<Value> {
    let mut definitions = vec![
        tool(
            "dragonoff.project_create",
            "Create and open a melodic-techno project.",
            json!({"type":"object","properties":{"path":{"type":"string"},"title":{"type":"string"},"bpm":{"type":"number","minimum":60,"maximum":180},"key":{"type":"string"},"scale":{"type":"string"},"sample_rate":{"type":"integer","enum":[44100,48000,96000,128000]},"seed":{"type":"integer"},"personality":{"type":"string","enum":["auto","noir","hypnotic","industrial","acid"]}},"required":["path","title"]}),
        ),
        tool(
            "dragonoff.project_get",
            "Read the current project or a project at path.",
            json!({"type":"object","properties":{"path":{"type":"string"}}}),
        ),
        tool(
            "dragonoff.instrument_catalog",
            "Discover all built-in sound sources and described patch recipes. Apply a recipe with dragonoff.patch_apply, then edit any controls.",
            json!({"type":"object","properties":{"family":{"type":"string","description":"Optional family filter, e.g. bass, lead, chords, pad, arp, keys, texture."}}}),
        ),
        tool(
            "dragonoff.sample_catalog",
            "List installed samples, or scan an external sample-pack directory for WAVs. External results include absolute paths for sample_path and the caller-supplied license label; verify license terms before distribution.",
            json!({"type":"object","properties":{"type":{"type":"string","description":"Optional installed sample type filter, such as kick, hat_closed, perc, or instrument_sample."},"tag":{"type":"string","description":"Optional installed sample character tag filter, such as deep, metallic, or airy."},"directory":{"type":"string","description":"Optional external sample-pack directory to scan recursively for WAV files."},"license":{"type":"string","description":"Optional license label to preserve with the catalog result, e.g. CC0 1.0. The server does not verify the claim."}}}),
        ),
        tool(
            "dragonoff.patch_apply",
            "Apply one editable starting patch to a track. This changes only its synth sound settings; track effects and musical content remain agent-controlled.",
            json!({"type":"object","properties":{"track_id":{"type":"string"},"preset_id":{"type":"string"}},"required":["track_id","preset_id"]}),
        ),
        tool(
            "dragonoff.composition_catalog",
            "Browse alternative harmonic arcs, groove devices, phrase-development ideas, and arrangement forms. These are adaptable starting ideas, not fixed templates.",
            json!({"type":"object","properties":{}}),
        ),
        tool(
            "dragonoff.arrangement_steer",
            "Offer optional arrangement ideas for the current track combinations. Suggestions never add tracks or constrain the producer's choices.",
            json!({"type":"object","properties":{"path":{"type":"string"}}}),
        ),
        tool(
            "dragonoff.musical_context",
            "Read the written MIDI events already sounding in a bar range, expanded through their arrangement clips. Use this to make new parts answer the actual harmony, bass rhythm and drum pocket. This is descriptive context only: it never filters notes or imposes a scale, rhythm or style.",
            json!({"type":"object","properties":{"path":{"type":"string"},"start_bar":{"type":"integer","minimum":1},"bars":{"type":"integer","minimum":1,"maximum":128},"track_ids":{"type":"array","items":{"type":"string"},"description":"Optional subset such as [\"kick\",\"bass\",\"chords\"]. Omit to inspect all tracks."}}}),
        ),
        tool(
            "dragonoff.project_set",
            "Set global musical controls: title, tempo, key, scale, meter, sample rate or random seed.",
            json!({"type":"object","properties":{"path":{"type":"string"},"title":{"type":"string"},"bpm":{"type":"number","minimum":60,"maximum":180},"key":{"type":"string"},"scale":{"type":"string"},"numerator":{"type":"integer","minimum":1,"maximum":12},"denominator":{"type":"integer","enum":[2,4,8,16]},"sample_rate":{"type":"integer","enum":[44100,48000,96000,128000]},"seed":{"type":"integer"},"personality":{"type":"string","enum":["auto","noir","hypnotic","industrial","acid"]}}}),
        ),
        tool(
            "dragonoff.generate_arrangement",
            "Optional fixed-profile starter arrangement. For a track authored to the user's request, create tracks and write your own notes with track_add, track_set, pattern_import_json, clip_place and sections_set; this starter is not required.",
            json!({"type":"object","properties":{"path":{"type":"string"},"seed":{"type":"integer"},"profile":{"type":"string","enum":["club_drive"],"description":"The phrase-first club profile."},"personality":{"type":"string","enum":["auto","noir","hypnotic","industrial","acid"],"description":"Phrase vocabulary: noir, hypnotic, industrial or acid. Auto selects reproducibly from the seed."}}}),
        ),
        tool(
            "dragonoff.sections_set",
            "Set the song form using named sections and energy targets.",
            json!({
                "type":"object",
                "properties":{"sections":{"type":"array","items":{
                    "type":"object",
                    "properties":{
                        "name":{"type":"string"},
                        "start_bar":{"type":"integer","minimum":1},
                        "bars":{"type":"integer","minimum":1},
                        "energy":{"type":"number","minimum":0,"maximum":1}
                    },
                    "required":["name","start_bar","bars"]
                }}},
                "required":["sections"]
            }),
        ),
        tool(
            "dragonoff.track_add",
            "Add a track with a built-in role or any custom role name. Built-in roles receive convenient starting patches; custom roles get the same full patch controls.",
            json!({"type":"object","properties":{"id":{"type":"string"},"name":{"type":"string"},"role":{"type":"string"}},"required":["name","role"]}),
        ),
        tool(
            "dragonoff.track_set",
            "Set track controls, sample choices, synth patch and effects in one operation. Synth layers have independent oscillator, pan, envelope and filter controls; sampled roles include piano, flute, nylon guitar and steel_guitar.",
            json!({"type":"object","properties":{"track_id":{"type":"string"},"gain_db":{"type":"number","minimum":-60,"maximum":12},"pan":{"type":"number","minimum":-1,"maximum":1},"muted":{"type":"boolean"},"sample_id":{"type":"string","description":"Installed sample ID, e.g. house.kick.punch-02. Sampled roles piano, flute, guitar and steel_guitar select their internal multi-sample. For sampled synthesis roles choose instrument.sf2.ghost-mult, instrument.sf2.saw-morph, instrument.sf2.supra-saw, or instrument.sf2.fuzz-bass."},"sample_alt_id":{"type":"string","description":"Optional installed alternate sample; alternates each trigger."},"sample_tune_semitones":{"type":"number","minimum":-24,"maximum":24},"drum":{"type":"object","description":"Per-hit controls for sampled drum and percussion tracks.","properties":{"sample_start_ms":{"type":"number","minimum":0,"maximum":500},"decay_ms":{"type":"number","minimum":0,"maximum":5000,"description":"Maximum sample duration; zero preserves the full source."},"transient":{"type":"number","minimum":0,"maximum":2},"noise_mix":{"type":"number","minimum":0,"maximum":1}}},"sound":{"type":"object","description":"Replace the track synthesizer patch. Use 1-8 layers with independent oscillator, octave, level, tuning, unison, pulse width, FM, noise, pan, envelope and filter controls. Omitted layer envelopes and filters inherit the track patch.","properties":{"waveform":{"type":"string","enum":["sine","saw","square","triangle","choir","brass","fm","noise"]},"unison":{"type":"integer","minimum":1,"maximum":8},"detune_cents":{"type":"number","minimum":0,"maximum":50},"cutoff_hz":{"type":"number","minimum":20,"maximum":40000},"attack_ms":{"type":"number","minimum":0,"maximum":5000},"decay_ms":{"type":"number","minimum":0,"maximum":10000},"sustain":{"type":"number","minimum":0,"maximum":1},"release_ms":{"type":"number","minimum":0,"maximum":10000},"filter_mode":{"type":"string","enum":["lowpass","highpass","bandpass","notch","allpass"]},"filter_resonance":{"type":"number","minimum":0,"maximum":0.95},"filter_env_amount":{"type":"number","minimum":-4,"maximum":4,"description":"Filter envelope depth in octaves."},"filter_decay_ms":{"type":"number","minimum":1,"maximum":10000},"stereo_motion_depth":{"type":"number","minimum":0,"maximum":1,"description":"Equal-power stereo orbit depth; 0 is fixed, 1 reaches the sides."},"stereo_motion_rate_hz":{"type":"number","minimum":0,"maximum":8},"stereo_motion_phase_radians":{"type":"number","minimum":-std::f64::consts::TAU,"maximum":std::f64::consts::TAU},"binaural_itd_ms":{"type":"number","minimum":0,"maximum":1.2,"description":"Maximum far-ear interaural delay for headphone-oriented motion; this is a compact binaural cue, not HRTF rendering."},"lfo_rate_hz":{"type":"number","minimum":0,"maximum":30},"lfo_depth":{"type":"number","minimum":0,"maximum":4},"lfo_target":{"type":"string","enum":["cutoff","amplitude","pitch"],"description":"Pitch uses lfo_depth in semitones."},"layers":{"type":"array","maxItems":8,"items":{"type":"object","properties":{"waveform":{"type":"string","enum":["sine","saw","square","triangle","choir","brass","fm","noise"]},"octave":{"type":"integer","minimum":-3,"maximum":3},"gain":{"type":"number","minimum":0,"maximum":2},"detune_cents":{"type":"number","minimum":0,"maximum":50},"unison":{"type":"integer","minimum":1,"maximum":8},"pulse_width":{"type":"number","minimum":0.05,"maximum":0.95},"fm_amount":{"type":"number","minimum":0,"maximum":1},"noise_amount":{"type":"number","minimum":0,"maximum":1}},"additionalProperties":false}}}},"effects":{"type":"object","properties":{"lowpass_hz":{"type":"number"},"highpass_hz":{"type":"number"},"saturation":{"type":"number","minimum":0,"maximum":1},"saturation_mode":{"type":"string","enum":["soft_clip","hard_clip","foldback"]},"delay_ms":{"type":"number","minimum":0,"maximum":1200},"delay_beats":{"type":"number","minimum":0.125,"maximum":2},"delay_feedback":{"type":"number","minimum":0,"maximum":0.85},"delay_wet":{"type":"number","minimum":0,"maximum":0.8},"reverb_send":{"type":"number","minimum":0,"maximum":0.8},"chorus_depth":{"type":"number","minimum":0,"maximum":1},"chorus_rate_hz":{"type":"number","minimum":0.02,"maximum":8},"compressor_threshold_db":{"type":"number","minimum":-36,"maximum":0},"compressor_ratio":{"type":"number","minimum":1,"maximum":20},"compressor_attack_ms":{"type":"number","minimum":0.1,"maximum":100},"compressor_release_ms":{"type":"number","minimum":10,"maximum":1000},"sidechain":{"type":"boolean"},"sidechain_amount":{"type":"number","minimum":0,"maximum":0.95},"sidechain_release_ms":{"type":"number","minimum":20,"maximum":1000}}}} ,"required":["track_id"]}),
        ),
        tool(
            "dragonoff.pattern_import_json",
            "Bulk-import a complete motif, chord, bassline or drum phrase. Accepts Ardour-style midi_events with one-based region-relative bars/beats and n/v/l/t aliases, plus descriptive field aliases.",
            json!({"type":"object","properties":{"track_id":{"type":"string"},"pattern_id":{"type":"string"},"bars":{"type":"integer","minimum":1},"midi":{"type":"object","properties":{"ticks_per_quarter":{"type":"integer"},"is_drum_mode":{"type":"boolean"},"channel_base":{"type":"string","enum":["one","zero"]},"midi_events":{"type":"array","items":{"type":"object"}},"events":{"type":"array","items":{"type":"object"}}}}},"required":["track_id","pattern_id","bars","midi"]}),
        ),
        tool(
            "dragonoff.clip_place",
            "Place or replace a pattern clip on the arrangement timeline.",
            json!({"type":"object","properties":{"track_id":{"type":"string"},"clip_id":{"type":"string"},"pattern_id":{"type":"string"},"start_bar":{"type":"integer","minimum":1},"bars":{"type":"integer","minimum":1},"transpose":{"type":"integer"}},"required":["track_id","clip_id","pattern_id","start_bar","bars"]}),
        ),
        tool(
            "dragonoff.clip_remove",
            "Remove an arrangement clip by track and clip ID.",
            json!({"type":"object","properties":{"track_id":{"type":"string"},"clip_id":{"type":"string"}},"required":["track_id","clip_id"]}),
        ),
        tool(
            "dragonoff.automation_set",
            "Replace a parameter lane with musical-time points. Supports gain_db, lowpass_hz, reverb_send, delay_feedback, delay_wet, chorus_depth, pitch_envelope_semitones, pitch LFO depth/rate/bias, and stereo motion depth/rate.",
            json!({"type":"object","properties":{"track_id":{"type":"string"},"parameter":{"type":"string"},"points":{"type":"array","items":{"type":"object","properties":{"bar":{"type":"integer","minimum":1},"beat":{"type":"number","minimum":1},"value":{"type":"number"}},"required":["bar","beat","value"]}}},"required":["track_id","parameter","points"]}),
        ),
        tool(
            "dragonoff.master_set",
            "Set the master bus: five-band corrective/tone EQ, stereo width, glue compression and lookahead peak limiter.",
            json!({"type":"object","properties":{"path":{"type":"string"},"output_gain_db":{"type":"number","minimum":-24,"maximum":18},"clipper_drive_db":{"type":"number","minimum":0,"maximum":6,"description":"Smooth master soft clipper drive before the peak limiter; start near 1-3 dB for density."},"eq_highpass_hz":{"type":"number","minimum":10,"maximum":120},"eq_low_shelf_db":{"type":"number","minimum":-6,"maximum":6},"eq_low_mid_hz":{"type":"number","minimum":80,"maximum":800},"eq_low_mid_db":{"type":"number","minimum":-6,"maximum":6},"eq_low_mid_q":{"type":"number","minimum":0.3,"maximum":4},"eq_presence_hz":{"type":"number","minimum":1000,"maximum":10000},"eq_presence_db":{"type":"number","minimum":-6,"maximum":6},"eq_presence_q":{"type":"number","minimum":0.3,"maximum":4},"eq_high_shelf_db":{"type":"number","minimum":-6,"maximum":6},"stereo_width":{"type":"number","minimum":0,"maximum":3.0,"description":"High-frequency M/S width; keep the low end centered with low_band_width."},"low_band_width":{"type":"number","minimum":0,"maximum":1},"compressor_threshold_db":{"type":"number","minimum":-36,"maximum":0},"compressor_ratio":{"type":"number","minimum":1,"maximum":8},"compressor_attack_ms":{"type":"number","minimum":0.1,"maximum":100},"compressor_release_ms":{"type":"number","minimum":10,"maximum":1000},"limiter_ceiling_db":{"type":"number","minimum":-12,"maximum":0},"limiter_release_ms":{"type":"number","minimum":10,"maximum":1000},"limiter_lookahead_ms":{"type":"number","minimum":0,"maximum":20}},"required":[]}),
        ),
        tool(
            "dragonoff.chord_options",
            "Return playable chord qualities plus close, open and drop-2 voicings across all inversions for an agent-selected root note.",
            json!({"type":"object","properties":{"root_midi":{"type":"integer","minimum":36,"maximum":84},"quality":{"type":"string","description":"Optional quality filter, e.g. minor9, major9, sus4, dominant7b9"}},"required":["root_midi"]}),
        ),
        tool(
            "dragonoff.project_render",
            "Render the current arrangement to stereo WAV. Optional Surge XT backend renders melodic roles through a local VST3.",
            json!({"type":"object","properties":{"output":{"type":"string"},"path":{"type":"string"},"backend":{"type":"string","enum":["native","surge_xt"]},"plugin_path":{"type":"string"}},"required":["output"]}),
        ),
        tool(
            "dragonoff.vst_parameter_catalog",
            "List the raw parameter IDs and current values exposed by a VST3 plugin. Use these IDs with track_set.plugin_parameters for direct per-track sound design on the Surge XT backend.",
            json!({"type":"object","properties":{"plugin_path":{"type":"string"},"prefix":{"type":"string","description":"Optional parameter ID prefix, such as a_osc_1 or a_filter_1."}},"required":["plugin_path"]}),
        ),
        tool(
            "dragonoff.audio_measure",
            "Measure objective WAV signal properties such as loudness, peak, clipping and stereo energy. It does not score musical quality or arrangement choices.",
            json!({"type":"object","properties":{"wav":{"type":"string"}},"required":["wav"]}),
        ),
        tool(
            "dragonoff.track_reconstruction_compare",
            "Compare a synthesized render with its source WAV. Reports tempo, key/chord, bar-energy, spectral-balance, loudness and stereo differences, plus suggested parameter adjustments for an automated refinement pass.",
            json!({"type":"object","properties":{"reference_wav":{"type":"string","description":"Original source WAV used for track_analyze."},"rendered_wav":{"type":"string","description":"Synthesized WAV exported from the reconstructed Dragonoff project."}},"required":["reference_wav","rendered_wav"]}),
        ),
        tool(
            "dragonoff.track_analyze",
            "Analyze a finished WAV mix for tempo/key, bars, chords, transients, spectral balance and energy. Set include_ml=true for Demucs separation and Basic Pitch note transcription; the report includes bass per-note envelope, harmonic and pitch-bend evidence, drum-hit decay and pitch trajectories, per-event synth overrides, bar-level mix targets and a native-synth reconstruction spec.",
            json!({"type":"object","properties":{"wav":{"type":"string","description":"Path to a PCM16/24/32 or float32 RIFF/WAVE file."},"include_ml":{"type":"boolean","description":"Run optional pretrained Demucs and Basic Pitch. Adds bass note envelope/harmonic/pitch-bend observations, per-hit drum envelope and low-frequency pitch trajectories, event-level synth controls, and a Dragonoff native-synth reconstruction spec. Requires the optional Python packages."},"output_directory":{"type":"string","description":"Directory for separated stems, MIDI/CSV note transcriptions, and the ML analysis report. Required when include_ml is true."}},"required":["wav"]}),
        ),
        tool(
            "dragonoff.track_reconstruct",
            "Create a Dragonoff project from a track analysis report. Builds every analyzed part as a native synth track from its reconstruction spec, preserving the extracted notes and estimated sound controls.",
            json!({"type":"object","properties":{"analysis_json":{"type":"string","description":"Path to track_analysis_ml.json produced by dragonoff.track_analyze with include_ml=true."},"project_path":{"type":"string","description":"Path where the new .dragonoff.json project will be saved."},"title":{"type":"string","description":"Optional project title."}},"required":["analysis_json","project_path"]}),
        ),
        tool(
            "dragonoff.project_save",
            "Save the current project.",
            json!({"type":"object","properties":{"path":{"type":"string"}}}),
        ),
    ];
    let track = definitions
        .iter_mut()
        .find(|d| d["name"] == "dragonoff.track_set")
        .unwrap();
    let props = track["inputSchema"]["properties"].as_object_mut().unwrap();
    props.insert("source_type".into(),json!({"type":"string","enum":["auto","synth","one_shot"],"description":"Choose synthesis independently of semantic role, including synthesized kicks or snares."}));
    props.insert("bus_id".into(),json!({"type":["string","null"],"description":"Route to one declared bus; null routes directly to master."}));
    for key in ["sample_id", "sample_alt_id"] {
        props.get_mut(key).unwrap()["type"] = json!(["string", "null"]);
    }
    props.insert("sample_path".into(), json!({"type":["string","null"],"description":"Optional WAV path for this sample_id. Absolute paths support external royalty-free libraries; the file is read locally and is not copied into the project."}));
    props.insert("plugin_parameters".into(), json!({"type":["object","null"],"description":"Optional plugin-specific raw parameter overrides, keyed by the exact ID returned by dragonoff.vst_parameter_catalog. Values are normalized 0..1 and override the backend's starting patch. Null clears overrides.","additionalProperties":{"type":"number","minimum":0,"maximum":1}}));
    props.insert(
        "sample_alt_path".into(),
        json!({"type":["string","null"],"description":"Optional WAV path for sample_alt_id."}),
    );
    let sound = props.get_mut("sound").unwrap()["properties"]
        .as_object_mut()
        .unwrap();
    sound.insert("phase_reset".into(), json!({"type":"boolean"}));
    sound.insert("legato".into(), json!({"type":"boolean","description":"Merge touching notes of the same pitch so the amp/filter envelope does not retrigger."}));
    sound.insert(
        "pitch_envelope_semitones".into(),
        json!({"type":"number","minimum":-24,"maximum":48}),
    );
    sound.insert(
        "pitch_decay_ms".into(),
        json!({"type":"number","minimum":1,"maximum":5000}),
    );
    sound.insert(
        "lfo_bias".into(),
        json!({"type":"number","minimum":-1,"maximum":1,"description":"Bias pitch LFO toward downward motion when negative; used when lfo_target is pitch."}),
    );
    sound.insert(
        "output_gain_db".into(),
        json!({"type":"number","minimum":-36,"maximum":18}),
    );
    sound.insert("stereo_spread".into(),json!({"type":"number","minimum":0,"maximum":1,"description":"Actual pan spread between unison voices, before effects."}));
    let layers = sound.get_mut("layers").unwrap()["items"]["properties"]
        .as_object_mut()
        .unwrap();
    for (key, min, max) in [
        ("tune_semitones", -24.0, 24.0),
        ("fm_ratio", 0.25, 12.0),
        ("fm_decay_ms", 0.0, 10000.0),
        ("pulse_width_depth", 0.0, 0.4),
        ("pulse_width_rate_hz", 0.0, 20.0),
    ] {
        layers.insert(
            key.into(),
            json!({"type":"number","minimum":min,"maximum":max}),
        );
    }
    layers.insert(
        "pan".into(),
        json!({"type":"number","minimum":-1,"maximum":1}),
    );
    for (key, min, max) in [
        ("cutoff_hz", 20.0, 40000.0),
        ("filter_resonance", 0.0, 0.95),
        ("attack_ms", 0.0, 5000.0),
        ("decay_ms", 0.0, 10000.0),
        ("sustain", 0.0, 1.0),
        ("release_ms", 0.0, 10000.0),
    ] {
        layers.insert(
            key.into(),
            json!({"type":["number","null"],"minimum":min,"maximum":max,"description":"Optional per-layer override; null or omitted inherits the track patch."}),
        );
    }
    let effects = props.get_mut("effects").unwrap()["properties"]
        .as_object_mut()
        .unwrap();
    effects.extend(processor_properties());
    let master = definitions
        .iter_mut()
        .find(|d| d["name"] == "dragonoff.master_set")
        .unwrap();
    master["inputSchema"]["properties"]["eq"] = eq_schema();
    master["inputSchema"]["properties"]["compressor_knee_db"] =
        json!({"type":"number","minimum":0,"maximum":24});
    master["inputSchema"]["properties"]["compressor_sidechain_highpass_hz"] =
        json!({"type":"number","minimum":0,"maximum":1000});
    master["inputSchema"]["properties"]["stereo_width"]["maximum"] = json!(2.0);
    definitions.push(tool("dragonoff.reverb_set","Set the shared diffuse stereo room return: decay, pre-delay, damping, filtering and level.",json!({"type":"object","properties":{"decay_seconds":{"type":"number","minimum":0.1,"maximum":12},"predelay_ms":{"type":"number","minimum":0,"maximum":250},"damping_hz":{"type":"number","minimum":500,"maximum":20000},"highpass_hz":{"type":"number","minimum":10,"maximum":2000},"lowpass_hz":{"type":"number","minimum":500,"maximum":20000},"return_gain_db":{"type":"number","minimum":-60,"maximum":6}}})));
    let render = definitions
        .iter_mut()
        .find(|d| d["name"] == "dragonoff.project_render")
        .unwrap();
    render["inputSchema"]["properties"]["stems_directory"] = json!({"type":"string","description":"Optional folder for aligned float32 track, bus and premaster stems."});
    render["inputSchema"]["properties"]["master_bypass"] = json!({"type":"boolean","description":"Export the premaster without master EQ, compression, clipper or limiter."});
    definitions.push(tool("dragonoff.bus_set","Create or update a parallel stereo bus. Track routing is explicit. Bus EQ, dynamic EQ, saturation, compression and audio-triggered sidechain are independent.",json!({"type":"object","properties":{"id":{"type":"string"},"name":{"type":"string"},"gain_db":{"type":"number","minimum":-60,"maximum":12},"stereo_width":{"type":"number","minimum":0,"maximum":2},"effects":{"type":"object","properties":processor_properties()}},"required":["id"]})));
    definitions.push(tool(
        "dragonoff.bus_remove",
        "Remove an unused bus; route its tracks elsewhere first.",
        json!({"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}),
    ));
    definitions.push(tool("dragonoff.track_clone","Create an independent mixing layer from a track. By default the new layer has no MIDI or clips; copy_content can copy them too.",json!({"type":"object","properties":{"source_id":{"type":"string"},"id":{"type":"string"},"name":{"type":"string"},"copy_content":{"type":"boolean"}},"required":["source_id","id"]})));
    definitions
}

fn eq_schema() -> Value {
    json!({"type":"array","maxItems":12,"items":{"type":"object","properties":{"kind":{"type":"string","enum":["bell","highpass","lowpass","low_shelf","high_shelf","notch"]},"frequency_hz":{"type":"number","minimum":10,"maximum":40000},"q":{"type":"number","minimum":0.1,"maximum":16},"gain_db":{"type":"number","minimum":-24,"maximum":24},"dynamic_threshold_db":{"type":["number","null"],"minimum":-72,"maximum":0},"dynamic_ratio":{"type":"number","minimum":1,"maximum":12},"dynamic_range_db":{"type":"number","minimum":0,"maximum":12},"attack_ms":{"type":"number","minimum":0.1,"maximum":200},"release_ms":{"type":"number","minimum":5,"maximum":2000}},"required":["frequency_hz"]}})
}
fn processor_properties() -> serde_json::Map<String, Value> {
    let mut map = serde_json::Map::new();
    map.insert("eq".into(), eq_schema());
    for (key, min, max) in [
        ("compressor_knee_db", 0.0, 24.0),
        ("compressor_makeup_db", -24.0, 24.0),
        ("compressor_sidechain_highpass_hz", 0.0, 1000.0),
        ("sidechain_threshold_db", -60.0, 0.0),
        ("highpass_hz", 10.0, 20000.0),
        ("lowpass_hz", 20.0, 40000.0),
        ("saturation", 0.0, 1.0),
        ("compressor_threshold_db", -60.0, 0.0),
        ("compressor_ratio", 1.0, 20.0),
        ("compressor_attack_ms", 0.1, 100.0),
        ("compressor_release_ms", 5.0, 2000.0),
        ("sidechain_amount", 0.0, 0.95),
        ("sidechain_release_ms", 5.0, 2000.0),
    ] {
        map.insert(
            key.into(),
            json!({"type":"number","minimum":min,"maximum":max}),
        );
    }
    for key in ["highpass_hz", "lowpass_hz"] {
        map.get_mut(key).unwrap()["type"] = json!(["number", "null"]);
    }
    map.insert(
        "saturation_mode".into(),
        json!({"type":"string","enum":["soft_clip","hard_clip","foldback"]}),
    );
    map.insert("sidechain".into(), json!({"type":"boolean"}));
    map.insert("sidechain_source".into(),json!({"type":["string","null"],"description":"Track ID used as an actual audio detector; defaults to the first kick."}));
    map
}

fn merge_settings<T: serde::Serialize + serde::de::DeserializeOwned>(
    current: &T,
    changes: &Value,
) -> Result<T, String> {
    let mut value = serde_json::to_value(current).map_err(|e| e.to_string())?;
    let target = value.as_object_mut().ok_or("settings must be an object")?;
    for (k, v) in changes
        .as_object()
        .ok_or("settings changes must be an object")?
    {
        if !target.contains_key(k) {
            return Err(format!("unknown settings field {k}"));
        }
        target.insert(k.clone(), v.clone());
    }
    serde_json::from_value(value).map_err(|e| e.to_string())
}

fn tool(name: &str, description: &str, input_schema: Value) -> Value {
    json!({"name":name,"description":description,"inputSchema":input_schema})
}

fn fresh_project_seed() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    (nanos ^ (u64::from(std::process::id()) << 32)).max(1)
}

fn external_sample_catalog(directory: &Path, license: &str) -> Result<Vec<Value>, String> {
    fn visit(directory: &Path, license: &str, output: &mut Vec<Value>) -> Result<(), String> {
        let entries = std::fs::read_dir(directory).map_err(|error| {
            format!(
                "cannot read sample directory {}: {error}",
                directory.display()
            )
        })?;
        for entry in entries {
            let entry =
                entry.map_err(|error| format!("cannot read sample directory entry: {error}"))?;
            let kind = entry.file_type().map_err(|error| error.to_string())?;
            if kind.is_dir() {
                visit(&entry.path(), license, output)?;
            } else if kind.is_file()
                && entry
                    .path()
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
            {
                let path = entry
                    .path()
                    .canonicalize()
                    .map_err(|error| error.to_string())?;
                let sample_type = path
                    .parent()
                    .and_then(Path::file_name)
                    .and_then(|name| name.to_str())
                    .unwrap_or("sample");
                output.push(json!({
                    "id": format!("external.{}", path.file_stem().and_then(|name| name.to_str()).unwrap_or("sample")),
                    "type": sample_type,
                    "path": path,
                    "license": license,
                }));
                if output.len() > 2048 {
                    return Err("sample catalog is limited to 2048 WAV files per directory".into());
                }
            }
        }
        Ok(())
    }

    let directory = directory.canonicalize().map_err(|error| {
        format!(
            "cannot resolve sample directory {}: {error}",
            directory.display()
        )
    })?;
    if !directory.is_dir() {
        return Err(format!(
            "sample catalog path is not a directory: {}",
            directory.display()
        ));
    }
    let mut samples = Vec::new();
    visit(&directory, license, &mut samples)?;
    samples.sort_by(|left, right| left["path"].as_str().cmp(&right["path"].as_str()));
    Ok(samples)
}

fn call_tool(name: &str, args: &Value, active: &mut Option<PathBuf>) -> Result<Value, String> {
    match name {
        "dragonoff.project_create" => {
            let path = string(args, "path")?;
            let title = string(args, "title")?;
            let bpm = args.get("bpm").and_then(Value::as_f64).unwrap_or(124.0);
            let key = args
                .get("key")
                .and_then(Value::as_str)
                .unwrap_or("F#")
                .to_string();
            let scale = args
                .get("scale")
                .and_then(Value::as_str)
                .unwrap_or("minor")
                .to_string();
            let sr = args
                .get("sample_rate")
                .and_then(Value::as_u64)
                .unwrap_or(48_000) as u32;
            let seed = args
                .get("seed")
                .and_then(Value::as_u64)
                .unwrap_or_else(fresh_project_seed);
            if !(60.0..=180.0).contains(&bpm) {
                return Err("bpm must be in 60..180".into());
            }
            let mut song = Song::fresh(title.to_string(), bpm, key, scale, sr, seed);
            if let Some(personality) = args.get("personality").and_then(Value::as_str) {
                song.personality = personality.to_ascii_lowercase();
                if !matches!(
                    song.personality.as_str(),
                    "auto" | "noir" | "hypnotic" | "industrial" | "acid"
                ) {
                    return Err(
                        "personality must be auto, noir, hypnotic, industrial or acid".into(),
                    );
                }
            }
            save_song(Path::new(&path), &song)?;
            *active = Some(PathBuf::from(&path));
            Ok(
                json!({"created":path,"bpm":bpm,"seed":seed,"personality":song.personality,"memory_mode":"bar-streamed"}),
            )
        }
        "dragonoff.project_get" => {
            let p = path_for(args, active)?;
            let song = load_song(&p)?;
            Ok(serde_json::to_value(song).map_err(|e| e.to_string())?)
        }
        "dragonoff.instrument_catalog" => {
            let mut library: Value =
                serde_json::from_str(include_str!("../assets/catalog/instruments.json"))
                    .map_err(|e| format!("invalid embedded instrument catalog: {e}"))?;
            if let Some(family) = args.get("family").and_then(Value::as_str) {
                let presets = library
                    .get_mut("presets")
                    .and_then(Value::as_array_mut)
                    .ok_or("instrument catalog is missing presets")?;
                presets
                    .retain(|preset| preset.get("family").and_then(Value::as_str) == Some(family));
            }
            Ok(library)
        }
        "dragonoff.sample_catalog" => {
            if let Some(directory) = args.get("directory").and_then(Value::as_str) {
                let license = args
                    .get("license")
                    .and_then(Value::as_str)
                    .unwrap_or("unspecified; verify before distribution");
                let samples = external_sample_catalog(Path::new(directory), license)?;
                return Ok(json!({"source_directory": directory, "samples": samples}));
            }
            let mut library: Value =
                serde_json::from_str(include_str!("../assets/samples/manifest.json"))
                    .map_err(|e| format!("invalid embedded sample catalog: {e}"))?;
            let samples = library
                .get_mut("samples")
                .and_then(Value::as_object_mut)
                .ok_or("sample manifest is missing samples")?;
            let wanted_type = args.get("type").and_then(Value::as_str);
            let wanted_tag = args.get("tag").and_then(Value::as_str);
            samples.retain(|_, item| {
                let type_matches =
                    wanted_type.is_none_or(|t| item.get("type").and_then(Value::as_str) == Some(t));
                let tag_matches = wanted_tag.is_none_or(|tag| {
                    item.get("character_tags")
                        .and_then(Value::as_array)
                        .is_some_and(|tags| tags.iter().any(|value| value.as_str() == Some(tag)))
                });
                type_matches && tag_matches
            });
            Ok(library)
        }
        "dragonoff.vst_parameter_catalog" => {
            let plugin = string(args, "plugin_path")?;
            let python = std::env::var("DRAGONOFF_PYTHON").unwrap_or_else(|_| "python3".into());
            let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/plugin_catalog.py");
            let mut command = Command::new(python);
            command.arg(script).arg(plugin);
            if let Some(prefix) = args.get("prefix").and_then(Value::as_str) {
                command.arg("--prefix").arg(prefix);
            }
            let result = command
                .output()
                .map_err(|e| format!("could not start VST parameter catalog: {e}"))?;
            if !result.status.success() {
                return Err(format!(
                    "VST parameter catalog failed: {}",
                    String::from_utf8_lossy(&result.stderr)
                ));
            }
            serde_json::from_slice(&result.stdout)
                .map_err(|e| format!("VST parameter catalog returned invalid JSON: {e}"))
        }
        "dragonoff.composition_catalog" => {
            serde_json::from_str(include_str!("../assets/catalog/compositions.json"))
                .map_err(|e| format!("invalid embedded composition catalog: {e}"))
        }
        "dragonoff.patch_apply" => {
            let p = path_for(args, active)?;
            let track_id = string(args, "track_id")?;
            let preset_id = string(args, "preset_id")?;
            let library: Value =
                serde_json::from_str(include_str!("../assets/catalog/instruments.json"))
                    .map_err(|e| e.to_string())?;
            let preset = library["presets"].as_array().and_then(|items| items.iter().find(|item| item["id"] == preset_id)).ok_or("unknown preset_id; call dragonoff.instrument_catalog to discover available patches")?;
            let sound: SoundSettings = serde_json::from_value(preset["sound"].clone())
                .map_err(|e| format!("preset sound is invalid: {e}"))?;
            validate_sound(&sound)?;
            let mut song = load_song(&p)?;
            let track = song
                .tracks
                .iter_mut()
                .find(|t| t.id == track_id)
                .ok_or("track not found")?;
            track.sound = sound.clone();
            save_song(&p, &song)?;
            Ok(
                json!({"track_id":track_id,"preset_id":preset_id,"name":preset["name"],"description":preset["description"],"sound":sound,"editable":true}),
            )
        }
        "dragonoff.arrangement_steer" => {
            let p = path_for(args, active)?;
            let song = load_song(&p)?;
            Ok(arrangement_steering(&song))
        }
        "dragonoff.musical_context" => {
            let p = path_for(args, active)?;
            let song = load_song(&p)?;
            let start_bar = args.get("start_bar").and_then(Value::as_u64).unwrap_or(1) as u32;
            let bars = args.get("bars").and_then(Value::as_u64).unwrap_or(16) as u32;
            if start_bar == 0 || bars == 0 || bars > 128 {
                return Err("start_bar must be positive and bars must be in 1..128".into());
            }
            let end_bar = start_bar.saturating_add(bars);
            let selected: Option<Vec<String>> = args
                .get("track_ids")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                });
            let tracks = song
                .tracks
                .iter()
                .filter(|track| {
                    selected
                        .as_ref()
                        .is_none_or(|ids| ids.iter().any(|id| id == &track.id))
                })
                .map(|track| {
                    let mut events = Vec::new();
                    for clip in &track.clips {
                        let Some(pattern) = track
                            .patterns
                            .iter()
                            .find(|pattern| pattern.id == clip.pattern_id)
                        else {
                            continue;
                        };
                        if pattern.bars == 0 {
                            continue;
                        }
                        let from = start_bar.max(clip.start_bar);
                        let to = end_bar.min(clip.start_bar.saturating_add(clip.bars));
                        for global_bar in from..to {
                            let local_bar = (global_bar - clip.start_bar) % pattern.bars + 1;
                            for note in pattern.notes.iter().filter(|note| note.bar == local_bar) {
                                events.push(json!({
                                    "bar":global_bar,
                                    "beat":note.beat,
                                    "pitch":(note.pitch as i16 + clip.transpose).clamp(0,127),
                                    "length_beats":note.length_beats,
                                    "velocity":note.velocity,
                                    "clip_id":clip.id,
                                    "pattern_id":pattern.id
                                }));
                            }
                        }
                    }
                    events.sort_by(|a, b| {
                        let ab = a["bar"]
                            .as_u64()
                            .unwrap_or(0)
                            .cmp(&b["bar"].as_u64().unwrap_or(0));
                        ab.then_with(|| {
                            a["beat"]
                                .as_f64()
                                .unwrap_or(0.0)
                                .total_cmp(&b["beat"].as_f64().unwrap_or(0.0))
                        })
                    });
                    json!({"track_id":track.id,"name":track.name,"role":track.role,"events":events})
                })
                .collect::<Vec<_>>();
            let sections = song
                .sections
                .iter()
                .filter(|section| {
                    section.start_bar < end_bar && section.start_bar + section.bars > start_bar
                })
                .collect::<Vec<_>>();
            Ok(
                json!({"title":song.title,"bpm":song.bpm,"key_label":song.key,"scale_label":song.scale,"meter":[song.numerator,song.denominator],"range":{"start_bar":start_bar,"bars":bars},"sections":sections,"tracks":tracks,"note":"Events are descriptive. Choose any notes, rhythms, voicings or outside tones you want."}),
            )
        }
        "dragonoff.project_set" => {
            let p = path_for(args, active)?;
            let mut song = load_song(&p)?;
            if let Some(v) = args.get("title").and_then(Value::as_str) {
                song.title = v.to_string();
            }
            if let Some(v) = args.get("bpm").and_then(Value::as_f64) {
                if !(60.0..=180.0).contains(&v) {
                    return Err("bpm must be in 60..180".into());
                }
                song.bpm = v;
            }
            if let Some(v) = args.get("key").and_then(Value::as_str) {
                song.key = v.to_string();
            }
            if let Some(v) = args.get("scale").and_then(Value::as_str) {
                song.scale = v.to_string();
            }
            if let Some(v) = args.get("numerator").and_then(Value::as_u64) {
                if !(1..=12).contains(&v) {
                    return Err("numerator must be in 1..12".into());
                }
                song.numerator = v as u8;
            }
            if let Some(v) = args.get("denominator").and_then(Value::as_u64) {
                if ![2, 4, 8, 16].contains(&v) {
                    return Err("denominator must be 2, 4, 8 or 16".into());
                }
                song.denominator = v as u8;
            }
            if let Some(v) = args.get("sample_rate").and_then(Value::as_u64) {
                if ![44100, 48000, 96000, 128000].contains(&v) {
                    return Err("unsupported sample rate".into());
                }
                song.sample_rate = v as u32;
            }
            if let Some(v) = args.get("seed").and_then(Value::as_u64) {
                song.seed = v.max(1);
            }
            if let Some(v) = args.get("personality").and_then(Value::as_str) {
                let v = v.to_ascii_lowercase();
                if !matches!(
                    v.as_str(),
                    "auto" | "noir" | "hypnotic" | "industrial" | "acid"
                ) {
                    return Err(
                        "personality must be auto, noir, hypnotic, industrial or acid".into(),
                    );
                }
                song.personality = v;
            }
            save_song(&p, &song)?;
            Ok(
                json!({"title":song.title,"bpm":song.bpm,"key":song.key,"scale":song.scale,"meter":format!("{}/{}",song.numerator,song.denominator),"sample_rate":song.sample_rate,"seed":song.seed,"personality":song.personality}),
            )
        }
        "dragonoff.master_set" => {
            let p = path_for(args, active)?;
            let mut song = load_song(&p)?;
            let mut master = song.master.clone();
            set_master_f32(
                args,
                "output_gain_db",
                -24.0,
                18.0,
                &mut master.output_gain_db,
            )?;
            set_master_f32(
                args,
                "clipper_drive_db",
                0.0,
                6.0,
                &mut master.clipper_drive_db,
            )?;
            set_master_f32(
                args,
                "eq_highpass_hz",
                10.0,
                120.0,
                &mut master.eq_highpass_hz,
            )?;
            set_master_f32(
                args,
                "eq_low_shelf_db",
                -6.0,
                6.0,
                &mut master.eq_low_shelf_db,
            )?;
            set_master_f32(
                args,
                "eq_low_mid_hz",
                80.0,
                800.0,
                &mut master.eq_low_mid_hz,
            )?;
            set_master_f32(args, "eq_low_mid_db", -6.0, 6.0, &mut master.eq_low_mid_db)?;
            set_master_f32(args, "eq_low_mid_q", 0.3, 4.0, &mut master.eq_low_mid_q)?;
            set_master_f32(
                args,
                "eq_presence_hz",
                1000.0,
                10000.0,
                &mut master.eq_presence_hz,
            )?;
            set_master_f32(
                args,
                "eq_presence_db",
                -6.0,
                6.0,
                &mut master.eq_presence_db,
            )?;
            set_master_f32(args, "eq_presence_q", 0.3, 4.0, &mut master.eq_presence_q)?;
            set_master_f32(
                args,
                "eq_high_shelf_db",
                -6.0,
                6.0,
                &mut master.eq_high_shelf_db,
            )?;
            set_master_f32(args, "stereo_width", 0.0, 2.0, &mut master.stereo_width)?;
            set_master_f32(args, "low_band_width", 0.0, 1.0, &mut master.low_band_width)?;
            set_master_f32(
                args,
                "compressor_threshold_db",
                -36.0,
                0.0,
                &mut master.compressor_threshold_db,
            )?;
            set_master_f32(
                args,
                "compressor_ratio",
                1.0,
                8.0,
                &mut master.compressor_ratio,
            )?;
            set_master_f32(
                args,
                "compressor_attack_ms",
                0.1,
                100.0,
                &mut master.compressor_attack_ms,
            )?;
            set_master_f32(
                args,
                "compressor_release_ms",
                10.0,
                1000.0,
                &mut master.compressor_release_ms,
            )?;
            set_master_f32(
                args,
                "limiter_ceiling_db",
                -12.0,
                0.0,
                &mut master.limiter_ceiling_db,
            )?;
            set_master_f32(
                args,
                "limiter_release_ms",
                10.0,
                1000.0,
                &mut master.limiter_release_ms,
            )?;
            set_master_f32(
                args,
                "limiter_lookahead_ms",
                0.0,
                20.0,
                &mut master.limiter_lookahead_ms,
            )?;
            song.master = master;
            set_master_f32(
                args,
                "compressor_knee_db",
                0.0,
                24.0,
                &mut song.master.compressor_knee_db,
            )?;
            set_master_f32(
                args,
                "compressor_sidechain_highpass_hz",
                0.0,
                1000.0,
                &mut song.master.compressor_sidechain_highpass_hz,
            )?;
            if let Some(eq) = args.get("eq") {
                let bands: Vec<EqBand> =
                    serde_json::from_value(eq.clone()).map_err(|e| e.to_string())?;
                validate_eq(&bands)?;
                song.master.eq = bands;
            }
            save_song(&p, &song)?;
            Ok(serde_json::to_value(&song.master).map_err(|e| e.to_string())?)
        }
        "dragonoff.chord_options" => {
            let root = args
                .get("root_midi")
                .and_then(Value::as_u64)
                .filter(|v| (36..=84).contains(v))
                .ok_or("root_midi must be in 36..84")? as i32;
            let requested = args.get("quality").and_then(Value::as_str);
            let options = chord_quality_options()
                .into_iter()
                .filter(|(quality, _, _)| requested.is_none_or(|r| r == *quality))
                .map(|(quality, label, intervals)| {
                    let mut voicings = Vec::new();
                    for inversion in 0..intervals.len() {
                        let mut close: Vec<i32> = intervals.iter().cycle().skip(inversion).take(intervals.len()).enumerate().map(|(i, interval)| root + *interval + if i + inversion >= intervals.len() { 12 } else { 0 }).collect();
                        close.sort_unstable();
                        let mut open = close.clone();
                        for i in (2..open.len()).step_by(2) { open[i] += 12; }
                        open.sort_unstable();
                        let mut drop2 = close.clone();
                        if drop2.len() >= 4 { let at = drop2.len() - 2; drop2[at] -= 12; drop2.sort_unstable(); }
                        voicings.push(json!({"inversion":inversion,"layout":"close","midi_notes":close}));
                        voicings.push(json!({"inversion":inversion,"layout":"open","midi_notes":open}));
                        voicings.push(json!({"inversion":inversion,"layout":"drop2","midi_notes":drop2}));
                    }
                    json!({"quality":quality,"label":label,"intervals":intervals,"midi_notes":intervals.iter().map(|i|root + *i).collect::<Vec<_>>(),"voicings":voicings})
                })
                .collect::<Vec<_>>();
            if options.is_empty() {
                return Err(
                    "unknown quality; call dragonoff.chord_options without quality to list choices"
                        .into(),
                );
            }
            Ok(json!({"root_midi":root,"options":options}))
        }
        "dragonoff.generate_arrangement" => {
            let p = path_for(args, active)?;
            let mut song = load_song(&p)?;
            if let Some(seed) = args.get("seed").and_then(Value::as_u64) {
                song.seed = seed.max(1);
            }
            if let Some(personality) = args.get("personality").and_then(Value::as_str) {
                song.personality = personality.to_ascii_lowercase();
            }
            if !matches!(
                song.personality.as_str(),
                "auto" | "noir" | "hypnotic" | "industrial" | "acid"
            ) {
                return Err("personality must be auto, noir, hypnotic, industrial or acid".into());
            }
            let profile = args
                .get("profile")
                .and_then(Value::as_str)
                .unwrap_or("club_drive");
            if profile != "club_drive" {
                return Err("the phrase-first engine supports the club_drive profile".into());
            }
            // Use the phrase-first engine. The previous path stacked template
            // roles, expanded loops with hash-based note mutations, then
            // filtered the result; that pipeline is intentionally bypassed.
            crate::composer::build(&mut song);
            save_song(&p, &song)?;
            Ok(
                json!({"generated":true,"profile":profile,"personality":song.personality,"seed":song.seed,"bars":song.total_bars(),"duration_minutes":(song.total_bars() as f64 * song.numerator as f64 * 4.0 / song.denominator as f64 / song.bpm),"tracks":song.tracks.iter().map(|t|t.name.clone()).collect::<Vec<_>>(),"path":p}),
            )
        }
        "dragonoff.sections_set" => {
            let p = path_for(args, active)?;
            let mut song = load_song(&p)?;
            let sections: Vec<Section> =
                serde_json::from_value(args.get("sections").cloned().ok_or("sections required")?)
                    .map_err(|e| e.to_string())?;
            if sections
                .iter()
                .any(|s| s.start_bar == 0 || s.bars == 0 || !(0.0..=1.0).contains(&s.energy))
            {
                return Err("section bars/start_bar must be positive; energy must be 0..1".into());
            }
            let mut sorted = sections.clone();
            sorted.sort_by_key(|s| s.start_bar);
            for pair in sorted.windows(2) {
                if pair[0].start_bar + pair[0].bars > pair[1].start_bar {
                    return Err("sections overlap".into());
                }
            }
            song.sections = sorted;
            save_song(&p, &song)?;
            Ok(json!({"sections":song.sections,"total_bars":song.total_bars()}))
        }
        "dragonoff.track_add" => {
            let p = path_for(args, active)?;
            let mut song = load_song(&p)?;
            let name = string(args, "name")?;
            let role = string(args, "role")?;
            if !valid_role(role) {
                return Err(
                    "role must be 1..48 characters using letters, digits, underscore or hyphen"
                        .into(),
                );
            }
            let id = args
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| format!("track-{}", song.tracks.len() + 1));
            if song.tracks.iter().any(|t| t.id == id) {
                return Err("track id already exists".into());
            }
            song.tracks
                .push(new_track(id.clone(), name.to_string(), role.to_string()));
            save_song(&p, &song)?;
            Ok(json!({"track_id":id,"track_count":song.tracks.len()}))
        }
        "dragonoff.reverb_set" => {
            let p = path_for(args, active)?;
            let mut song = load_song(&p)?;
            song.reverb = merge_settings(&song.reverb, args)?;
            let r = &song.reverb;
            for (v, min, max) in [
                (r.decay_seconds, 0.1, 12.0),
                (r.predelay_ms, 0.0, 250.0),
                (r.damping_hz, 500.0, 20000.0),
                (r.highpass_hz, 10.0, 2000.0),
                (r.lowpass_hz, 500.0, 20000.0),
                (r.return_gain_db, -60.0, 6.0),
            ] {
                if !v.is_finite() || v < min || v > max {
                    return Err("invalid reverb settings".into());
                }
            }
            save_song(&p, &song)?;
            serde_json::to_value(&song.reverb).map_err(|e| e.to_string())
        }
        "dragonoff.bus_set" => {
            let p = path_for(args, active)?;
            let mut song = load_song(&p)?;
            let id = string(args, "id")?;
            if !valid_role(id) {
                return Err("invalid bus id".into());
            }
            let mut bus = song
                .buses
                .iter()
                .find(|b| b.id == id)
                .cloned()
                .unwrap_or(MixBus {
                    id: id.into(),
                    name: id.into(),
                    gain_db: 0.0,
                    effects: Effects::default(),
                    stereo_width: 1.0,
                });
            if let Some(name) = args.get("name").and_then(Value::as_str) {
                bus.name = name.into();
            }
            set_master_f32(args, "gain_db", -60.0, 12.0, &mut bus.gain_db)?;
            set_master_f32(args, "stereo_width", 0.0, 2.0, &mut bus.stereo_width)?;
            if let Some(e) = args.get("effects") {
                bus.effects = merge_settings(&bus.effects, e)?;
                validate_effects(&bus.effects)?;
            }
            if bus.effects.delay_ms > 0.0
                || bus.effects.delay_beats.is_some()
                || bus.effects.delay_wet > 0.0
                || bus.effects.reverb_send > 0.0
                || bus.effects.chorus_depth > 0.0
            {
                return Err("Bus inserts support EQ, dynamics and saturation. Put time effects on independent tracks.".into());
            }
            song.buses.retain(|b| b.id != id);
            song.buses.push(bus.clone());
            save_song(&p, &song)?;
            serde_json::to_value(bus).map_err(|e| e.to_string())
        }
        "dragonoff.bus_remove" => {
            let p = path_for(args, active)?;
            let mut song = load_song(&p)?;
            let id = string(args, "id")?;
            if song.tracks.iter().any(|t| t.bus_id.as_deref() == Some(id)) {
                return Err("bus still has routed tracks".into());
            }
            song.buses.retain(|b| b.id != id);
            save_song(&p, &song)?;
            Ok(json!({"removed":id}))
        }
        "dragonoff.track_clone" => {
            let p = path_for(args, active)?;
            let mut song = load_song(&p)?;
            let id = string(args, "id")?;
            let source = string(args, "source_id")?;
            if !valid_role(id) || song.tracks.iter().any(|t| t.id == id) {
                return Err("track id invalid or already exists".into());
            }
            let mut track = song
                .tracks
                .iter()
                .find(|t| t.id == source)
                .cloned()
                .ok_or("source track not found")?;
            track.id = id.into();
            track.name = args
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(id)
                .into();
            if !args
                .get("copy_content")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                track.patterns.clear();
                track.clips.clear();
                track.automation.clear();
            }
            song.tracks.push(track);
            save_song(&p, &song)?;
            Ok(json!({"created":id,"source":source}))
        }
        "dragonoff.track_set" => {
            let p = path_for(args, active)?;
            let mut song = load_song(&p)?;
            let id = string(args, "track_id")?;
            let (
                gain_db,
                pan,
                effects,
                sound,
                drum,
                sample_id,
                sample_alt_id,
                sample_path,
                sample_alt_path,
                sample_tune_semitones,
                plugin_parameters,
            ) = {
                let track = song
                    .tracks
                    .iter_mut()
                    .find(|t| t.id == id)
                    .ok_or("track not found")?;
                if let Some(source) = args.get("source_type").and_then(Value::as_str) {
                    if !["auto", "synth", "one_shot"].contains(&source) {
                        return Err("invalid source_type".into());
                    }
                    track.source_type = source.into();
                }
                if let Some(v) = args.get("gain_db").and_then(Value::as_f64) {
                    track.gain_db = v.clamp(-60.0, 12.0) as f32;
                }
                if let Some(v) = args.get("pan").and_then(Value::as_f64) {
                    track.pan = v.clamp(-1.0, 1.0) as f32;
                }
                if let Some(v) = args.get("bus_id") {
                    track.bus_id = if v.is_null() {
                        None
                    } else {
                        Some(
                            v.as_str()
                                .ok_or("bus_id must be string or null")?
                                .to_string(),
                        )
                    };
                }
                if let Some(v) = args.get("muted").and_then(Value::as_bool) {
                    track.muted = v;
                }
                if let Some(v) = args.get("sample_id") {
                    track.sample_id = if v.is_null() {
                        None
                    } else {
                        Some(
                            v.as_str()
                                .ok_or("sample_id must be a string or null")?
                                .to_string(),
                        )
                    };
                }
                if let Some(v) = args.get("sample_alt_id") {
                    track.sample_alt_id = if v.is_null() {
                        None
                    } else {
                        Some(
                            v.as_str()
                                .ok_or("sample_alt_id must be a string or null")?
                                .to_string(),
                        )
                    };
                }
                if let Some(v) = args.get("sample_path") {
                    track.sample_path = if v.is_null() {
                        None
                    } else {
                        Some(
                            v.as_str()
                                .ok_or("sample_path must be a string or null")?
                                .to_string(),
                        )
                    };
                }
                if let Some(v) = args.get("sample_alt_path") {
                    track.sample_alt_path = if v.is_null() {
                        None
                    } else {
                        Some(
                            v.as_str()
                                .ok_or("sample_alt_path must be a string or null")?
                                .to_string(),
                        )
                    };
                }
                if let Some(v) = args.get("sample_tune_semitones").and_then(Value::as_f64) {
                    track.sample_tune_semitones = v.clamp(-24.0, 24.0) as f32;
                }
                if let Some(d) = args.get("drum") {
                    track.drum = merge_settings(&track.drum, d)?;
                    validate_drum(&track.drum)?;
                }
                if let Some(s) = args.get("sound") {
                    track.sound = merge_settings(&track.sound, s)?;
                    validate_sound(&track.sound)?;
                }
                if let Some(parameters) = args.get("plugin_parameters") {
                    if parameters.is_null() {
                        track.plugin_parameters.clear();
                    } else {
                        let values = parameters
                            .as_object()
                            .ok_or("plugin_parameters must be an object or null")?;
                        for (key, value) in values {
                            let normalized = value
                                .as_f64()
                                .ok_or("plugin parameter values must be numbers")?;
                            if key.is_empty()
                                || key.len() > 128
                                || !normalized.is_finite()
                                || !(0.0..=1.0).contains(&normalized)
                            {
                                return Err(format!("invalid normalized plugin parameter `{key}`"));
                            }
                            track
                                .plugin_parameters
                                .insert(key.clone(), normalized as f32);
                        }
                    }
                }
                if let Some(e) = args.get("effects") {
                    track.effects = merge_settings(&track.effects, e)?;
                    validate_effects(&track.effects)?;
                }
                (
                    track.gain_db,
                    track.pan,
                    track.effects.clone(),
                    track.sound.clone(),
                    track.drum.clone(),
                    track.sample_id.clone(),
                    track.sample_alt_id.clone(),
                    track.sample_path.clone(),
                    track.sample_alt_path.clone(),
                    track.sample_tune_semitones,
                    track.plugin_parameters.clone(),
                )
            };
            for t in &song.tracks {
                if let Some(id) = &t.bus_id
                    && !song.buses.iter().any(|b| &b.id == id)
                {
                    return Err(format!("missing bus {id}"));
                }
            }
            save_song(&p, &song)?;
            Ok(
                json!({"track_id":id,"gain_db":gain_db,"pan":pan,"effects":effects,"sound":sound,"plugin_parameters":plugin_parameters,"drum":drum,"sample_id":sample_id,"sample_alt_id":sample_alt_id,"sample_path":sample_path,"sample_alt_path":sample_alt_path,"sample_tune_semitones":sample_tune_semitones}),
            )
        }
        "dragonoff.pattern_import_json" => {
            let p = path_for(args, active)?;
            let mut song = load_song(&p)?;
            let id = string(args, "track_id")?;
            let pattern_id = string(args, "pattern_id")?;
            let bars = args.get("bars").and_then(Value::as_u64).unwrap_or(1) as u32;
            if bars == 0 {
                return Err("bars must be >= 1".into());
            }
            let (numerator, denominator) = (song.numerator, song.denominator);
            let notes = parse_bulk_notes(
                args.get("midi").ok_or("midi required")?,
                numerator,
                denominator,
            )?;
            if notes.iter().any(|n| n.bar > bars) {
                return Err("note bar exceeds pattern length".into());
            }
            let note_count = notes.len();
            let pattern = Pattern {
                id: pattern_id.to_string(),
                bars,
                notes,
            };
            {
                let track = song
                    .tracks
                    .iter_mut()
                    .find(|t| t.id == id)
                    .ok_or("track not found")?;
                if let Some(existing) = track.patterns.iter_mut().find(|p| p.id == pattern_id) {
                    *existing = pattern;
                } else {
                    track.patterns.push(pattern);
                }
            }
            save_song(&p, &song)?;
            Ok(json!({"track_id":id,"pattern_id":pattern_id,"notes":note_count}))
        }
        "dragonoff.clip_place" => {
            let p = path_for(args, active)?;
            let mut song = load_song(&p)?;
            let id = string(args, "track_id")?;
            let track = song
                .tracks
                .iter_mut()
                .find(|t| t.id == id)
                .ok_or("track not found")?;
            let clip_id = string(args, "clip_id")?;
            let clip = Clip {
                id: clip_id.to_string(),
                pattern_id: string(args, "pattern_id")?.to_string(),
                start_bar: args.get("start_bar").and_then(Value::as_u64).unwrap_or(1) as u32,
                bars: args.get("bars").and_then(Value::as_u64).unwrap_or(1) as u32,
                transpose: args
                    .get("transpose")
                    .and_then(Value::as_i64)
                    .unwrap_or(0)
                    .clamp(-48, 48) as i16,
            };
            if clip.start_bar == 0 || clip.bars == 0 {
                return Err("clip start_bar and bars must be positive".into());
            }
            if !track.patterns.iter().any(|p| p.id == clip.pattern_id) {
                return Err("pattern not found on track".into());
            }
            if let Some(old) = track.clips.iter_mut().find(|c| c.id == clip_id) {
                *old = clip.clone();
            } else {
                track.clips.push(clip.clone());
            }
            save_song(&p, &song)?;
            Ok(serde_json::to_value(clip).map_err(|e| e.to_string())?)
        }
        "dragonoff.clip_remove" => {
            let p = path_for(args, active)?;
            let mut song = load_song(&p)?;
            let track_id = string(args, "track_id")?;
            let clip_id = string(args, "clip_id")?;
            let track = song
                .tracks
                .iter_mut()
                .find(|t| t.id == track_id)
                .ok_or("track not found")?;
            let before = track.clips.len();
            track.clips.retain(|c| c.id != clip_id);
            if before == track.clips.len() {
                return Err("clip not found".into());
            }
            save_song(&p, &song)?;
            Ok(json!({"removed":clip_id,"track_id":track_id}))
        }
        "dragonoff.automation_set" => {
            let p = path_for(args, active)?;
            let mut song = load_song(&p)?;
            let track_id = string(args, "track_id")?;
            let parameter = string(args, "parameter")?;
            if ![
                "gain_db",
                "lowpass_hz",
                "reverb_send",
                "delay_feedback",
                "delay_wet",
                "chorus_depth",
                "pitch_envelope_semitones",
                "lfo_depth",
                "lfo_bias",
                "lfo_rate_hz",
                "stereo_motion_depth",
                "stereo_motion_rate_hz",
            ]
            .contains(&parameter)
            {
                return Err("unsupported automation parameter".into());
            }
            let points: Vec<AutomationPoint> =
                serde_json::from_value(args.get("points").cloned().ok_or("points required")?)
                    .map_err(|e| e.to_string())?;
            if points
                .iter()
                .any(|p| p.bar == 0 || !p.beat.is_finite() || p.beat < 1.0 || !p.value.is_finite())
            {
                return Err("invalid automation point".into());
            }
            let point_count = points.len();
            {
                let track = song
                    .tracks
                    .iter_mut()
                    .find(|t| t.id == track_id)
                    .ok_or("track not found")?;
                track.automation.retain(|p| p.parameter != parameter);
                track.automation.extend(points.into_iter().map(|mut p| {
                    p.parameter = parameter.to_string();
                    p
                }));
            }
            save_song(&p, &song)?;
            Ok(json!({"track_id":track_id,"parameter":parameter,"points":point_count}))
        }
        "dragonoff.project_render" => {
            let p = path_for(args, active)?;
            let song = load_song(&p)?;
            let output = string(args, "output")?;
            if args.get("backend").and_then(Value::as_str) == Some("surge_xt") {
                let plugin = args
                    .get("plugin_path")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .or_else(|| std::env::var("DRAGONOFF_SURGE_VST3").ok())
                    .ok_or("Surge XT backend needs plugin_path or DRAGONOFF_SURGE_VST3")?;
                let python = std::env::var("DRAGONOFF_PYTHON").unwrap_or_else(|_| "python3".into());
                let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/surge_render.py");
                let result = Command::new(python)
                    .arg(script)
                    .arg(&p)
                    .arg(output)
                    .arg("--plugin")
                    .arg(plugin)
                    .output()
                    .map_err(|e| format!("could not start Surge XT renderer: {e}"))?;
                if !result.status.success() {
                    return Err(format!(
                        "Surge XT render failed: {}",
                        String::from_utf8_lossy(&result.stderr)
                    ));
                }
                let stdout = String::from_utf8_lossy(&result.stdout);
                let report = stdout
                    .lines()
                    .rev()
                    .find_map(|line| serde_json::from_str::<Value>(line).ok())
                    .ok_or_else(|| format!("Surge XT renderer returned no report: {stdout}"))?;
                return Ok(report);
            }
            let stems = args
                .get("stems_directory")
                .and_then(Value::as_str)
                .map(Path::new);
            let bypass = args
                .get("master_bypass")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let stats = if stems.is_none() && !bypass {
                render::render_wav(&song, Path::new(output))?
            } else {
                render::render_wav_options(&song, Path::new(output), stems, bypass)?
            };
            let mut report = serde_json::to_value(&stats).map_err(|e| e.to_string())?;
            report["output"] = json!(output);
            report["duration_seconds"] = json!(stats.frames as f64 / stats.sample_rate as f64);
            report["bit_depth"] = json!(if bypass { 32 } else { 24 });
            report["master_bypass"] = json!(bypass);
            report["stems_format"] = json!("aligned float32, pre-master");
            report["memory_strategy"] =
                json!("one-bar audio; per-source tails; sidechain donors cached only");
            report["limiter_detector"] = json!(
                "4x FIR reconstruction estimate; 0.2 dB guard; not a certified true-peak meter"
            );
            report["clipper_oversampling"] = json!(4);
            report["latency_compensated"] = json!(true);
            report["quality_assessment"] = json!(
                "Engineering measurements only. This does not assess musical taste, sound-design quality, or release readiness."
            );
            let report_path = format!("{output}.mix.json");
            fs::write(
                &report_path,
                serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            report["report_path"] = json!(report_path);
            Ok(report)
        }
        "dragonoff.audio_measure" => {
            let wav = PathBuf::from(string(args, "wav")?);
            Ok(serde_json::to_value(audio::analyze_wav(&wav)?).map_err(|e| e.to_string())?)
        }
        "dragonoff.track_analyze" => {
            let wav = PathBuf::from(string(args, "wav")?);
            let mut result =
                serde_json::to_value(analysis::analyze_wav(&wav)?).map_err(|e| e.to_string())?;
            if args
                .get("include_ml")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                let output_directory = string(args, "output_directory")?;
                let python = std::env::var("DRAGONOFF_ANALYSIS_PYTHON")
                    .or_else(|_| std::env::var("DRAGONOFF_PYTHON"))
                    .unwrap_or_else(|_| "python3".into());
                let script =
                    Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/track_analysis_ml.py");
                let tempo = result["tempo_bpm"].as_f64().unwrap_or(120.0);
                let process = Command::new(python)
                    .arg(script)
                    .arg("--audio")
                    .arg(&wav)
                    .arg("--output")
                    .arg(output_directory)
                    .arg("--tempo")
                    .arg(tempo.to_string())
                    .output()
                    .map_err(|e| format!("could not start optional ML analysis: {e}"))?;
                if !process.status.success() {
                    return Err(format!(
                        "optional ML analysis failed: {}",
                        String::from_utf8_lossy(&process.stderr)
                    ));
                }
                let ml: Value = serde_json::from_slice(&process.stdout)
                    .map_err(|e| format!("optional ML analysis returned invalid JSON: {e}"))?;
                let ml_bars = ml
                    .get("bar_event_map")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let native_bars = result
                    .get("bars")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let bar_count = ml_bars.len().max(native_bars.len());
                let reconstruction_bars: Vec<Value> = (0..bar_count)
                    .map(|index| {
                        let native = native_bars.get(index).cloned().unwrap_or(Value::Null);
                        let deep = ml_bars.get(index).cloned().unwrap_or(Value::Null);
                        json!({
                            "bar": index + 1,
                            "start_seconds": native.get("start_seconds").cloned().or_else(|| deep.get("start_seconds_estimate").cloned()).unwrap_or(Value::Null),
                            "end_seconds": native.get("end_seconds").cloned().or_else(|| deep.get("end_seconds_estimate").cloned()).unwrap_or(Value::Null),
                            "relative_energy": native.get("relative_energy").cloned().unwrap_or(Value::Null),
                            "native_mix_chord_candidate": native.get("chord").cloned().unwrap_or(Value::Null),
                            "native_mix_chord_score_margin": native.get("chord_confidence").cloned().unwrap_or(Value::Null),
                            "ml_stem_chord_candidate": deep.get("chord_candidate").cloned().unwrap_or(Value::Null),
                            "ml_stem_chord_interpretation": deep.get("chord_interpretation").cloned().unwrap_or(Value::Null),
                            "ml_stem_chord_alternatives": deep.get("chord_top_candidates").cloned().unwrap_or_else(|| json!([])),
                            "strongest_pitch_classes": native.get("strongest_pitch_classes").cloned().unwrap_or_else(|| json!([])),
                            "note_event_count_by_layer": deep.get("note_event_count_by_layer").cloned().unwrap_or_else(|| json!({})),
                            "pitch_range_by_layer_midi": deep.get("pitch_range_by_layer_midi").cloned().unwrap_or_else(|| json!({}))
                        })
                    })
                    .collect();
                result["reconstruction_spec"] = ml
                    .get("reconstruction_spec")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                result["reconstruction_map"] = json!({
                    "description":"Bar-indexed arrangement map combining full-mix energy/harmony estimates with stem-separated event counts, chord alternatives, and the synth reconstruction specification. Detailed timed note and sound evidence remain in ml_analysis.",
                    "bars":reconstruction_bars,
                    "event_array_paths":ml.pointer("/reconstruction_spec/layer_event_paths").cloned().unwrap_or_else(|| json!({})),
                    "timing_reference":ml.pointer("/reconstruction_spec/timing").cloned().unwrap_or_else(|| json!({})),
                    "synth_reconstruction_spec_path":"synth_reconstruction_spec"
                });
                result["analysis_schema_version"] = json!("4.0");
                result["synth_reconstruction_spec"] = ml
                    .get("synth_reconstruction_spec")
                    .cloned()
                    .unwrap_or(Value::Null);
                result["ml_analysis"] = ml;
            }
            Ok(result)
        }
        "dragonoff.track_reconstruct" => {
            let analysis_path = PathBuf::from(string(args, "analysis_json")?);
            let project_path = PathBuf::from(string(args, "project_path")?);
            let analysis_text = fs::read_to_string(&analysis_path)
                .map_err(|e| format!("cannot read analysis JSON: {e}"))?;
            let analysis: Value = serde_json::from_str(&analysis_text)
                .map_err(|e| format!("invalid analysis JSON: {e}"))?;
            let spec = analysis
                .get("synth_reconstruction_spec")
                .ok_or("analysis JSON has no synth_reconstruction_spec; rerun track_analyze with include_ml=true")?;
            if spec.get("target_renderer").and_then(Value::as_str) != Some("dragonoff_native_synth")
            {
                return Err("analysis synth spec does not target Dragonoff native synth".into());
            }
            let bpm = spec
                .get("tempo_bpm")
                .and_then(Value::as_f64)
                .ok_or("synth spec is missing tempo_bpm")?;
            if !(60.0..=180.0).contains(&bpm) {
                return Err(
                    "analysis tempo must be in 60..180 BPM for project reconstruction".into(),
                );
            }
            let key = spec
                .get("key")
                .and_then(Value::as_str)
                .unwrap_or("C")
                .to_string();
            let scale = spec
                .get("scale")
                .and_then(Value::as_str)
                .unwrap_or("minor")
                .to_string();
            let title = args
                .get("title")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| {
                    analysis
                        .get("input_audio")
                        .and_then(Value::as_str)
                        .and_then(|p| Path::new(p).file_stem())
                        .and_then(|s| s.to_str())
                        .map(|s| format!("{s} Synth Reconstruction"))
                })
                .unwrap_or_else(|| "Synth Reconstruction".into());
            let mut song = Song::fresh(title, bpm, key, scale, 48_000, fresh_project_seed());
            if let Some(sections) = spec.get("sections") {
                song.sections = serde_json::from_value(sections.clone())
                    .map_err(|e| format!("invalid synth spec sections: {e}"))?;
            }
            let parts = spec
                .get("parts")
                .and_then(Value::as_array)
                .ok_or("synth spec parts must be an array")?;
            if parts.is_empty() {
                return Err("synth spec contains no parts".into());
            }
            for part in parts {
                let id = part
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or("synth part is missing id")?
                    .to_string();
                let name = part
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or(&id)
                    .to_string();
                let role = part
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("texture")
                    .to_string();
                if part.get("source_type").and_then(Value::as_str) != Some("synth") {
                    return Err(format!("synth part `{id}` must use source_type=synth"));
                }
                let pattern_value = part
                    .get("pattern")
                    .ok_or_else(|| format!("synth part `{id}` is missing pattern"))?;
                let pattern_id = pattern_value
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("synth part `{id}` pattern is missing id"))?
                    .to_string();
                let pattern_bars = pattern_value
                    .get("bars")
                    .and_then(Value::as_u64)
                    .unwrap_or(1) as u32;
                if pattern_bars == 0 {
                    return Err(format!(
                        "synth part `{id}` pattern must span at least one bar"
                    ));
                }
                let source_notes = pattern_value
                    .get("notes")
                    .cloned()
                    .unwrap_or_else(|| json!([]));
                let note_value = json!({"ticks_per_quarter":480,"midi_events":source_notes});
                let notes = parse_bulk_notes(&note_value, 4, 4)?;
                if notes.iter().any(|note| note.bar > pattern_bars) {
                    return Err(format!("synth part `{id}` has a note outside its pattern"));
                }
                let mut track = new_track(id.clone(), name, role);
                track.source_type = "synth".into();
                if let Some(sound) = part.get("sound") {
                    track.sound = merge_settings(&track.sound, sound)?;
                }
                validate_sound(&track.sound)?;
                track.patterns.push(Pattern {
                    id: pattern_id.clone(),
                    bars: pattern_bars,
                    notes,
                });
                track.clips.push(Clip {
                    id: format!("{id}_analysis_clip"),
                    pattern_id,
                    start_bar: 1,
                    bars: pattern_bars,
                    transpose: 0,
                });
                if let Some(points) = part.get("automation").and_then(Value::as_array) {
                    for point in points {
                        let automation: AutomationPoint = serde_json::from_value(point.clone())
                            .map_err(|e| format!("invalid automation in synth part `{id}`: {e}"))?;
                        let valid_parameter_value = match automation.parameter.as_str() {
                            "gain_db" => (-60.0..=12.0).contains(&automation.value),
                            "lowpass_hz" => (20.0..=20_000.0).contains(&automation.value),
                            _ => false,
                        };
                        if !valid_parameter_value
                            || automation.bar == 0
                            || automation.bar > pattern_bars
                            || !(1.0..=5.0).contains(&automation.beat)
                            || !automation.value.is_finite()
                        {
                            return Err(format!("invalid automation in synth part `{id}`"));
                        }
                        track.automation.push(automation);
                    }
                }
                song.tracks.push(track);
            }
            let path_string = project_path.display().to_string();
            save_song(&project_path, &song)?;
            *active = Some(project_path.clone());
            Ok(
                json!({"created":path_string,"title":song.title,"bpm":song.bpm,"key":song.key,"scale":song.scale,"bars":song.total_bars(),"track_count":song.tracks.len(),"all_tracks_source_type":"synth","analysis_json":analysis_path}),
            )
        }
        "dragonoff.track_reconstruction_compare" => {
            let reference = PathBuf::from(string(args, "reference_wav")?);
            let rendered = PathBuf::from(string(args, "rendered_wav")?);
            serde_json::to_value(analysis::compare_wavs(&reference, &rendered)?)
                .map_err(|e| format!("could not serialize reconstruction comparison: {e}"))
        }
        "dragonoff.project_save" => {
            let p = path_for(args, active)?;
            let song = load_song(&p)?;
            save_song(&p, &song)?;
            Ok(json!({"saved":p}))
        }
        _ => Err(format!("unknown tool: {name}")),
    }
}

fn new_track(id: String, name: String, role: String) -> Track {
    let sound = SoundSettings::for_role(&role);
    let mut effects = Effects::default();
    match role.as_str() {
        "hat" => effects.highpass_hz = Some(6500.0),
        "open_hat" => effects.highpass_hz = Some(6000.0),
        "clap" => effects.highpass_hz = Some(500.0),
        "perc" => effects.highpass_hz = Some(400.0),
        "pad" => {
            effects.reverb_send = 0.28;
            effects.chorus_depth = 0.32;
        }
        "chords" => {
            effects.reverb_send = 0.2;
            effects.chorus_depth = 0.22;
        }
        "lead" => {
            effects.reverb_send = 0.16;
            effects.delay_beats = Some(1.5);
            effects.delay_wet = 0.18;
            effects.delay_feedback = 0.2;
        }
        "arp" => {
            effects.reverb_send = 0.12;
            effects.delay_beats = Some(0.75);
            effects.delay_wet = 0.2;
            effects.delay_feedback = 0.22;
        }
        _ => {}
    }
    Track {
        source_type: "auto".into(),
        id,
        name,
        role,
        gain_db: 0.0,
        pan: 0.0,
        muted: false,
        bus_id: None,
        effects,
        sound,
        plugin_parameters: Default::default(),
        sample_id: None,
        sample_alt_id: None,
        sample_path: None,
        sample_alt_path: None,
        sample_tune_semitones: 0.0,
        drum: DrumSettings::default(),
        patterns: Vec::new(),
        clips: Vec::new(),
        automation: Vec::new(),
    }
}

fn arrangement_steering(song: &Song) -> Value {
    let present = |role: &str| {
        song.tracks
            .iter()
            .any(|t| t.role == role && !t.muted && !t.clips.is_empty())
    };
    let low_roles = ["sub", "bass", "pipe_bass", "brass_bass", "fm_bass"];
    let active_low: Vec<&str> = low_roles.iter().copied().filter(|r| present(r)).collect();
    let mut low_collisions = Vec::<Value>::new();
    let low_tracks: Vec<&Track> = song
        .tracks
        .iter()
        .filter(|t| low_roles.contains(&t.role.as_str()) && !t.muted)
        .collect();
    for i in 0..low_tracks.len() {
        for j in i + 1..low_tracks.len() {
            let a = placed_note_onsets(low_tracks[i]);
            let b = placed_note_onsets(low_tracks[j]);
            let count = a
                .iter()
                .filter(|(bar_a, beat_a, pitch_a)| {
                    *pitch_a < 60
                        && b.iter().any(|(bar_b, beat_b, pitch_b)| {
                            *pitch_b < 60 && bar_a == bar_b && (beat_a - beat_b).abs() < 0.08
                        })
                })
                .count();
            if count > 0 {
                low_collisions.push(json!({"tracks":[low_tracks[i].id,low_tracks[j].id],"roles":[low_tracks[i].role,low_tracks[j].role],"near_simultaneous_low_note_onsets":count}));
            }
        }
    }
    let mut findings = Vec::<Value>::new();
    let mut suggestions = Vec::<Value>::new();

    if !present("kick") {
        findings.push(json!({"level":"gap","area":"groove","evidence":"No active kick clip."}));
        suggestions.push(json!({"idea":"Add a kick pattern or use a different explicit pulse source.","why":"A consistent pulse gives melodic layers a stable rhythmic reference."}));
    }
    if active_low.is_empty() {
        findings.push(json!({"level":"gap","area":"low_end","evidence":"No active low-end role."}));
        suggestions.push(json!({"idea":"Try kick + one sustained sub role, then add one rhythmically distinct mid-bass if the groove needs more movement.","why":"Separate sub weight from mid-bass rhythm and harmonics."}));
    } else if active_low.len() >= 4 {
        findings.push(json!({"level":"review","area":"low_end","evidence":format!("{} low-end roles are active: {}",active_low.len(),active_low.join(", "))}));
        suggestions.push(json!({"idea":"Audition the low-end roles in pairs and mute any role that does not contribute a distinct register, rhythm, or transient.","why":"Several simultaneous bass roles can mask the kick and flatten the groove."}));
    }
    if !low_collisions.is_empty() {
        findings.push(json!({"level":"review","area":"low_end_timing","evidence":"Low-register bass roles have note onsets within 0.08 beats of each other.","pairs":low_collisions}));
        suggestions.push(json!({"idea":"Keep simultaneous low notes only when they reinforce the same harmonic event; otherwise shift one part, shorten its gate, or move its layer upward.","why":"Onset and register overlap can blur kick clearance and bass articulation."}));
    }
    if !present("chords") {
        findings.push(json!({"level":"gap","area":"harmony","evidence":"No active chord track."}));
        suggestions.push(json!({"idea":"Add a chord instrument, piano, or another harmonic voice with explicit voicing and register.","why":"A melodic hook benefits from a harmonic frame, which can be sparse."}));
    }
    if !present("lead") && !present("piano") && !present("flute") && !present("arp") {
        findings.push(
            json!({"level":"gap","area":"hook","evidence":"No active melodic or motif track."}),
        );
        suggestions.push(json!({"idea":"Introduce one recognizable motif and vary its register, rhythm, or orchestration across sections.","why":"A motif gives the arrangement continuity through breakdowns and drops."}));
    }
    if present("pad") && present("chords") {
        suggestions.push(json!({"idea":"Let the pad sustain the emotional harmony while chord stabs supply rhythm; automate density so both are not equally prominent throughout.","why":"Distinct note lengths and section roles make the layers read as an arrangement."}));
    }
    if present("piano") && present("flute") {
        suggestions.push(json!({"idea":"Use piano and flute as alternating foreground colors: piano can define the hook, with flute answering or taking the breakdown.","why":"Distinctive timbres stay legible when they take turns in the foreground."}));
    }

    let section_report: Vec<Value> = song.sections.iter().map(|section| {
        let end = section.start_bar.saturating_add(section.bars);
        let active: Vec<&str> = song.tracks.iter().filter(|t| !t.muted && t.clips.iter().any(|c| c.start_bar < end && section.start_bar < c.start_bar.saturating_add(c.bars))).map(|t| t.role.as_str()).collect();
        let unique: std::collections::BTreeSet<&str> = active.iter().copied().collect();
        let density = if section.energy >= 0.75 { "high-energy target" } else if section.energy <= 0.30 { "low-energy target" } else { "middle-energy target" };
        let alignment = if section.energy >= 0.75 && unique.len() < 5 { "sparse_for_target" } else if section.energy <= 0.30 && unique.len() > 8 { "dense_for_target" } else { "plausible" };
        json!({"name":section.name,"start_bar":section.start_bar,"bars":section.bars,"energy_target":section.energy,"target_read":density,"active_track_count":unique.len(),"active_roles":unique,"density_read":alignment})
    }).collect();
    if section_report.iter().any(|s| {
        s["density_read"] == "sparse_for_target" || s["density_read"] == "dense_for_target"
    }) {
        findings.push(json!({"level":"review","area":"section_density","evidence":"At least one section's active layer count is far from its stated energy target."}));
        suggestions.push(json!({"idea":"For dense sections, add or brighten a small number of layers; for low-energy sections, remove rhythmic or low-frequency roles while retaining a recognizable motif.","why":"Energy targets become audible through changing density and register, not loudness alone."}));
    }

    let chord_notes: Vec<u8> = song
        .tracks
        .iter()
        .filter(|t| t.role == "chords")
        .flat_map(|t| t.patterns.iter())
        .flat_map(|p| p.notes.iter().map(|n| n.pitch % 12))
        .collect();
    let chord_colors: std::collections::BTreeSet<u8> = chord_notes.iter().copied().collect();
    json!({
        "mode":"advisory; no edits made",
        "summary":{"tracks":song.tracks.len(),"sections":song.sections.len(),"active_low_end_roles":active_low,"distinct_chord_pitch_classes":chord_colors.len()},
        "findings":findings,
        "section_density":section_report,
        "combination_ideas":[
            {"name":"focused_drive","parts":["kick","mono sub","one syncopated mid-bass","closed hat"],"steering":"Use one primary bass rhythm; reserve additional bass colors for fills or transitions."},
            {"name":"melodic_drop","parts":["kick","sub or bass","chord stab","lead motif","one high rhythmic accent"],"steering":"Keep the hook readable by leaving a register or rhythmic gap around its strongest notes."},
            {"name":"operatic_breakdown","parts":["sustained chord voice","piano or flute motif","filtered texture"],"steering":"Reduce kick and sub density; let one distinctive voice carry the phrase."},
            {"name":"layered_transition","parts":["existing motif","rising filter or delay automation","short fill or noise layer"],"steering":"Use one or two changes to point into the next section; avoid changing every layer at once."}
        ],
        "suggestions":suggestions
    })
}

fn placed_note_onsets(track: &Track) -> Vec<(u32, f64, u8)> {
    let mut placed = Vec::new();
    for clip in &track.clips {
        let Some(pattern) = track.patterns.iter().find(|p| p.id == clip.pattern_id) else {
            continue;
        };
        for offset in 0..clip.bars {
            let bar = clip.start_bar + offset;
            let pattern_bar = (offset % pattern.bars.max(1)) + 1;
            for note in pattern.notes.iter().filter(|n| n.bar == pattern_bar) {
                placed.push((
                    bar,
                    note.beat + note.tick as f64 / 480.0,
                    (note.pitch as i16 + clip.transpose).clamp(0, 127) as u8,
                ));
            }
        }
    }
    placed
}

fn valid_role(role: &str) -> bool {
    !role.is_empty()
        && role.len() <= 48
        && role
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

fn chord_quality_options() -> Vec<(&'static str, &'static str, Vec<i32>)> {
    vec![
        ("minor7", "minor seventh", vec![0, 3, 7, 10]),
        ("minor9", "minor ninth", vec![0, 3, 7, 10, 14]),
        ("minor11", "minor eleventh", vec![0, 3, 7, 10, 14, 17]),
        ("minor6", "minor sixth", vec![0, 3, 7, 9]),
        ("major7", "major seventh", vec![0, 4, 7, 11]),
        ("major9", "major ninth", vec![0, 4, 7, 11, 14]),
        ("add9", "major add ninth", vec![0, 4, 7, 14]),
        ("sus2", "suspended second ninth", vec![0, 2, 7, 14]),
        ("sus4", "suspended fourth seventh", vec![0, 5, 7, 10]),
        ("dominant7", "dominant seventh", vec![0, 4, 7, 10]),
        ("dominant9", "dominant ninth", vec![0, 4, 7, 10, 14]),
        (
            "dominant7b9",
            "dominant seventh flat ninth",
            vec![0, 4, 7, 10, 13],
        ),
        ("minor7b5", "half-diminished seventh", vec![0, 3, 6, 10]),
        ("diminished7", "diminished seventh", vec![0, 3, 6, 9]),
        ("augmented", "augmented triad", vec![0, 4, 8]),
    ]
}

fn set_master_f32(
    args: &Value,
    key: &str,
    min: f32,
    max: f32,
    value: &mut f32,
) -> Result<(), String> {
    if let Some(v) = args.get(key).and_then(Value::as_f64) {
        if !v.is_finite() || v < min as f64 || v > max as f64 {
            return Err(format!("{key} must be in {min}..{max}"));
        }
        *value = v as f32;
    }
    Ok(())
}

fn save_song(path: &Path, song: &Song) -> Result<(), String> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_string_pretty(song).map_err(|e| e.to_string())?;
    fs::write(path, text).map_err(|e| format!("cannot save project: {e}"))
}
fn load_song(path: &Path) -> Result<Song, String> {
    let s = fs::read_to_string(path).map_err(|e| format!("cannot read project: {e}"))?;
    serde_json::from_str(&s).map_err(|e| format!("invalid project: {e}"))
}
fn path_for(args: &Value, active: &mut Option<PathBuf>) -> Result<PathBuf, String> {
    if let Some(p) = args.get("path").and_then(Value::as_str) {
        let pb = PathBuf::from(p);
        *active = Some(pb.clone());
        return Ok(pb);
    }
    active
        .clone()
        .ok_or_else(|| "create or select a project first".into())
}
fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{key} is required"))
}

#[cfg(test)]
mod controls_regressions {
    use super::*;
    #[test]
    fn partial_updates_preserve_eq_sidechain_and_envelope() {
        let fx: Effects = serde_json::from_value(
            json!({"sidechain":true,"eq":[{"frequency_hz":320,"gain_db":-3}],"chorus_depth":0.4}),
        )
        .unwrap();
        let changed: Effects = merge_settings(&fx, &json!({"saturation":0.2})).unwrap();
        assert!(changed.sidechain);
        assert_eq!(changed.eq[0].gain_db, -3.0);
        assert_eq!(changed.chorus_depth, 0.4);
        let sound = SoundSettings {
            attack_ms: 123.0,
            ..Default::default()
        };
        let changed: SoundSettings = merge_settings(&sound, &json!({"cutoff_hz":1200})).unwrap();
        assert_eq!(changed.attack_ms, 123.0);
        assert!(merge_settings::<Effects>(&fx, &json!({"saturaton":0.5})).is_err());
    }
    #[test]
    fn new_controls_are_discoverable_and_dynamic_eq_is_validated() {
        let tools = tool_definitions();
        assert!(tools.iter().any(|v| v["name"] == "dragonoff.bus_set"));
        let track = tools
            .iter()
            .find(|v| v["name"] == "dragonoff.track_set")
            .unwrap();
        assert!(track["inputSchema"]["properties"]["effects"]["properties"]["eq"].is_object());
        assert!(track["inputSchema"]["properties"]["sound"]["properties"]["layers"]["items"]["properties"]["fm_ratio"].is_object());
        let wrong: Vec<EqBand> = serde_json::from_value(
            json!([{"kind":"lowpass","frequency_hz":2000,"dynamic_threshold_db":-20}]),
        )
        .unwrap();
        assert!(validate_eq(&wrong).is_err());
    }
}
