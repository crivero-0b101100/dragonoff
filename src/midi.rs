// Copyright (C) 2026 OpenAI
// This program is free software under the GNU GPL v2 or (at your option) any later version.
//
// The one-based bar/beat/tick convention and repeated-bar expansion follow the
// Ardour MCP bulk MIDI JSON behavior (libs/surfaces/mcp_http/mcp_http_server.cc).

use crate::model::{Note, NoteSynthOverrides, validate_note_synth_overrides};
use serde_json::Value;

pub fn parse_bulk_notes(
    input: &Value,
    numerator: u8,
    denominator: u8,
) -> Result<Vec<Note>, String> {
    let ticks_per_quarter = input
        .get("ticks_per_quarter")
        .and_then(Value::as_u64)
        .unwrap_or(480);
    if ticks_per_quarter == 0 || ticks_per_quarter > 96_000 {
        return Err("ticks_per_quarter must be in 1..96000".into());
    }
    let drum_mode = input
        .get("is_drum_mode")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let channel_default = if drum_mode { 10 } else { 1 };
    let events = input
        .get("midi_events")
        .or_else(|| input.get("events"))
        .and_then(Value::as_array)
        .ok_or("midi_events (or events) must be an array")?;
    let channel_base = input
        .get("channel_base")
        .and_then(Value::as_str)
        .unwrap_or("one");
    if channel_base != "one" && channel_base != "zero" {
        return Err("channel_base must be 'one' or 'zero'".into());
    }
    let mut notes = Vec::new();

    for event in events {
        let bar = event
            .get("bar")
            .and_then(Value::as_u64)
            .ok_or("each event needs a positive bar")? as u32;
        if bar == 0 {
            return Err("bar is one-based and must be >= 1".into());
        }
        if let Some(source_bar) = event.get("repeat").and_then(Value::as_u64) {
            if event
                .as_object()
                .map(|o| o.keys().all(|k| k == "bar" || k == "repeat"))
                != Some(true)
            {
                return Err("repeat entries must contain only bar and repeat".into());
            }
            if source_bar == 0 {
                return Err("repeat source bar must be >= 1".into());
            }
            let source_notes: Vec<Note> = notes
                .iter()
                .filter(|n: &&Note| n.bar == source_bar as u32)
                .cloned()
                .collect();
            if source_notes.is_empty() {
                return Err(format!(
                    "repeat source bar {source_bar} has no earlier notes"
                ));
            }
            for mut note in source_notes {
                note.bar = bar;
                notes.push(note);
            }
            continue;
        }

        let beat = event
            .get("beat")
            .and_then(Value::as_f64)
            .or_else(|| event.get("b").and_then(Value::as_f64))
            .unwrap_or(1.0);
        let tick = event
            .get("tick")
            .or_else(|| event.get("t"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let pitch = event
            .get("pitch")
            .or_else(|| event.get("n"))
            .and_then(Value::as_u64)
            .ok_or("each note needs pitch (or n)")?;
        if pitch > 127 {
            return Err("pitch must be in MIDI range 0..127".into());
        }
        let raw_channel =
            event
                .get("channel")
                .and_then(Value::as_u64)
                .unwrap_or(if channel_base == "zero" {
                    (channel_default - 1) as u64
                } else {
                    channel_default as u64
                });
        let channel = if channel_base == "zero" {
            raw_channel + 1
        } else {
            raw_channel
        };
        if !(1..=16).contains(&channel) {
            return Err(if channel_base == "zero" {
                "channel must be in 0..15"
            } else {
                "channel must be in 1..16"
            }
            .into());
        }
        let velocity = event
            .get("velocity")
            .or_else(|| event.get("v"))
            .and_then(Value::as_u64)
            .unwrap_or(96);
        let synth_overrides = event
            .get("synth_overrides")
            .cloned()
            .map(serde_json::from_value::<NoteSynthOverrides>)
            .transpose()
            .map_err(|error| format!("invalid note synth_overrides: {error}"))?;
        let analysis_evidence = event.get("analysis_evidence").cloned();
        if let Some(overrides) = &synth_overrides {
            validate_note_synth_overrides(overrides)?;
        }
        if velocity > 127 {
            return Err("velocity must be in 0..127".into());
        }
        let length = event
            .get("length_beats")
            .or_else(|| event.get("l"))
            .and_then(Value::as_f64)
            .or_else(|| event.get("l").and_then(Value::as_f64))
            .unwrap_or(if drum_mode { 0.0 } else { 1.0 });
        if !beat.is_finite() || beat < 1.0 || beat > numerator as f64 + 1.0 {
            return Err("beat is outside this bar".into());
        }
        if !length.is_finite() || length < 0.0 || length > 64.0 {
            return Err("length_beats must be in 0..64".into());
        }
        if tick as u64 >= ticks_per_quarter {
            return Err("tick must be smaller than ticks_per_quarter".into());
        }
        // The project model stores ticks at 480 PPQ so patterns can be imported
        // from arbitrary input resolutions without carrying a per-pattern clock.
        let normalized_tick = ((tick as u64 * 480) / ticks_per_quarter) as u32;
        notes.push(Note {
            bar,
            beat,
            tick: normalized_tick.min(479),
            pitch: pitch as u8,
            length_beats: length,
            velocity: velocity as u8,
            channel: channel as u8,
            synth_overrides,
            analysis_evidence,
        });
    }

    // Use the same quarter-note conversion Ardour uses: time signatures change
    // the length of a bar without changing the meaning of beat/tick positions.
    let quarters_per_bar = numerator as f64 * (4.0 / denominator as f64);
    let quarters_per_beat = 4.0 / denominator as f64;
    for n in &notes {
        let q = (n.bar as f64 - 1.0) * quarters_per_bar
            + (n.beat - 1.0) * quarters_per_beat
            + n.tick as f64 / ticks_per_quarter as f64;
        if !q.is_finite() {
            return Err("note time is not finite".into());
        }
    }
    Ok(notes)
}
