//! Minimal, phrase-first composition engine.
//!
//! The former starter generator built a twelve-track checklist, copied an
//! eight-bar loop into longer blocks, then mutated note positions with hashes.
//! That produced activity without musical intent. This engine starts from a
//! small set of explicit phrase choices and only creates parts that have a
//! defined job. Seed variation selects a complete groove/phrase vocabulary;
//! it never adds arbitrary drum hits or instruments.

use crate::model::{
    AutomationPoint, Clip, DrumSettings, Effects, Note, Pattern, Section, Song, SoundSettings,
    Track,
};

pub fn build(song: &mut Song) {
    song.sections = vec![
        section("Opening", 1, 16, 0.18),
        section("First lift", 17, 20, 0.70),
        section("Breath", 37, 12, 0.35),
        section("Main passage", 49, 40, 0.84),
        section("Breakdown", 89, 30, 0.20),
        section("Return", 119, 32, 0.92),
        section("Short release", 151, 10, 0.44),
        section("Mix out", 161, 32, 0.22),
    ];
    song.tracks.clear();

    let root = key_pitch(&song.key, 3);
    let archetype = match song.personality.as_str() {
        "noir" => 0,
        "hypnotic" => 1,
        "industrial" => 2,
        "acid" => 3,
        _ => (song.seed ^ song.seed.rotate_left(17)) as usize % 4,
    };

    // The groove family chooses a coherent kit palette. This keeps a given
    // render internally consistent while ensuring generated projects do not
    // all inherit the same three samples.
    let kick_samples = [
        "house.kick.deep-02",
        "house.kick.deep-02",
        "house.kick.punch-04",
        "house.kick.punch-02",
    ];
    let hat_samples = [
        "house.hat.closed-04",
        "house.hat.closed-05",
        "house.hat.closed-06",
        "house.hat.closed-08",
    ];
    let clap_samples = [
        "house.clap.classic-02",
        "house.snare.snap-01",
        "house.clap.classic-02",
        "house.snare.snap-01",
    ];

    let mut kick = track("kick", "Kick", "kick", kick_samples[archetype], -1.0);
    kick.source_type = "one_shot".into();
    kick.patterns.push(pattern(
        "kick-sparse",
        1,
        vec![note(1, 1.0, 36, 0.78, 108), note(1, 3.0, 36, 0.78, 94)],
    ));
    let mut kick_drive = Vec::new();
    for bar in 1..=4 {
        for (beat, velocity) in [(1.0, 108), (2.0, 98), (3.0, 103), (4.0, 97)] {
            kick_drive.push(note(bar, beat, 36, 0.74, velocity));
        }
    }
    kick.patterns.push(pattern("kick-drive", 4, kick_drive));
    kick.clips = clips(
        "kick-sparse",
        &[(1, 16), (37, 12), (105, 14), (151, 10), (177, 16)],
    );
    kick.clips.extend(clips(
        "kick-drive",
        &[(17, 20), (49, 40), (119, 32), (161, 32)],
    ));
    song.tracks.push(kick);

    let mut hat = track("hat", "Closed hat", "hat", hat_samples[archetype], -8.0);
    hat.source_type = "one_shot".into();
    hat.pan = -0.18;
    hat.sample_alt_id = Some(
        match archetype {
            0 => "house.hat.closed-06",
            1 => "house.hat.closed-08",
            2 => "house.hat.closed-04",
            _ => "house.hat.closed-05",
        }
        .into(),
    );
    hat.patterns.push(pattern(
        "hat-sparse",
        1,
        vec![note(1, 2.5, 42, 0.14, 44), note(1, 4.5, 42, 0.14, 39)],
    ));
    let mut hat_groove = Vec::new();
    for bar in 1..=4 {
        for (i, beat) in [1.5, 2.5, 3.5, 4.5].into_iter().enumerate() {
            let velocity = if (i + bar as usize).is_multiple_of(2) {
                58
            } else {
                43
            };
            hat_groove.push(note(bar, beat, 42, 0.12, velocity));
        }
        // A paired sixteenth pickup appears at the ends of bars two and four,
        // giving the four-bar cell a clear internal shape.
        if bar == 2 || bar == 4 {
            hat_groove.push(note(bar, 2.75, 42, 0.07, 30));
            hat_groove.push(note(bar, 4.75, 42, 0.07, 36));
        }
    }
    hat.patterns.push(pattern("hat-groove", 4, hat_groove));
    hat.clips = clips(
        "hat-sparse",
        &[(1, 16), (37, 12), (89, 30), (151, 10), (177, 16)],
    );
    hat.clips.extend(clips(
        "hat-groove",
        &[(17, 20), (49, 40), (119, 32), (161, 16)],
    ));
    song.tracks.push(hat);

    let mut clap = track("clap", "Clap", "clap", clap_samples[archetype], -9.0);
    clap.source_type = "one_shot".into();
    clap.sample_alt_id = Some(if archetype % 2 == 0 {
        "house.snare.snap-01".into()
    } else {
        "house.clap.classic-02".into()
    });
    let clap_a: Vec<Note> = (1..=4)
        .flat_map(|bar| {
            let (two, four) = if bar % 2 == 0 { (68, 74) } else { (70, 76) };
            [
                note(bar, 2.0, 39, 0.22, two),
                note(bar, 4.0, 39, 0.22, four),
            ]
        })
        .collect();
    let mut clap_b = clap_a.clone();
    clap_b.extend([note(4, 3.75, 39, 0.10, 38), note(4, 4.5, 39, 0.10, 44)]);
    clap.patterns.push(pattern("clap-backbeat", 4, clap_a));
    clap.patterns.push(pattern("clap-turn", 4, clap_b));
    clap.clips = clips(
        "clap-backbeat",
        &[(17, 20), (49, 8), (65, 16), (119, 26), (161, 8)],
    );
    clap.clips
        .extend(clips("clap-turn", &[(57, 8), (81, 8), (145, 6), (169, 8)]));
    song.tracks.push(clap);

    // The fourth lane is a quiet shaker. Its entrance is deliberately sparse
    // and interlocks with the closed hat; it drops out in breaks and mix-out.
    let mut shaker = track(
        "shaker",
        "Shaker pulse",
        "perc",
        "house.perc.shaker-02",
        -16.0,
    );
    shaker.source_type = "one_shot".into();
    shaker.pan = 0.28;
    let mut shaker_notes = Vec::new();
    for bar in 1..=4 {
        for (beat, velocity) in [(1.75, 30), (2.75, 25), (3.75, 34), (4.75, 28)] {
            if bar % 2 == 0 || beat == 2.75 || beat == 4.75 {
                shaker_notes.push(note(bar, beat, 54, 0.08, velocity));
            }
        }
        if bar == 4 {
            shaker_notes.extend([note(bar, 3.5, 54, 0.07, 37), note(bar, 4.5, 54, 0.07, 42)]);
        }
    }
    shaker
        .patterns
        .push(pattern("shaker-four", 4, shaker_notes));
    shaker.clips = clips(
        "shaker-four",
        &[(25, 12), (49, 16), (73, 16), (125, 12), (145, 6)],
    );
    song.tracks.push(shaker);

    // Keep the kick as the only downbeat low-frequency source. The sampled
    // bass uses a restrained, offbeat answer with intentional phrase gaps.
    let bass_notes = rolling_bass_phrase(root, archetype, false);
    let bass_answer = rolling_bass_phrase(root, archetype, true);
    let mut bass = track(
        "bass",
        "Rolling analog sub bass",
        "sf2_bass",
        "instrument.sf2.fuzz-bass",
        -2.5,
    );
    bass.sound = SoundSettings {
        release_ms: 135.0,
        ..SoundSettings::default()
    };
    bass.effects.sidechain = true;
    bass.effects.sidechain_source = Some("kick".into());
    bass.effects.sidechain_amount = 0.58;
    bass.patterns.push(pattern("bass-a", 8, bass_notes));
    bass.patterns.push(pattern("bass-b", 8, bass_answer));
    bass.clips = clips("bass-a", &[(17, 20), (49, 16), (137, 14), (161, 32)]);
    bass.clips.extend(clips("bass-b", &[(65, 24), (119, 32)]));
    song.tracks.push(bass);

    // A chord-aware 16-bar lead phrase. Its rhythm stays editable, while the
    // pitch targets follow the chord cycle and the second half develops the
    // first instead of looping the same eight-bar cell.
    let hook = hook_phrase(root, archetype);
    let hook_answer = hook_answer(root, archetype);
    let mut lead = track(
        "hook",
        "Rolling modal lead riff",
        "sf2_lead",
        "instrument.sf2.saw-morph",
        -8.0,
    );
    lead.pan = -0.18;
    lead.sound = SoundSettings {
        binaural_itd_ms: 0.48,
        ..SoundSettings::default()
    };
    lead.effects.delay_beats = Some(0.75);
    lead.effects.delay_feedback = 0.12;
    lead.effects.delay_wet = 0.06;
    lead.effects.reverb_send = 0.34;
    lead.effects.chorus_depth = 0.28;
    lead.patterns.push(pattern("hook-a", 16, hook));
    lead.clips = clips("hook-a", &[(17, 72), (119, 74)]);
    lead.automation = automation_curve(
        "reverb_send",
        &[
            (17, 0.30),
            (49, 0.38),
            (73, 0.50),
            (89, 0.30),
            (119, 0.42),
            (145, 0.54),
            (151, 0.32),
            (161, 0.24),
        ],
    );
    song.tracks.push(lead);

    // A separate, softer response voice overlaps the main motif at phrase
    // boundaries. Its airy patch and rising reverb curve keep it distinct from
    // the dry foreground line; the two parts trade space instead of stacking
    // the same MIDI notes on one instrument.
    let mut response = track(
        "hook_response",
        "String lead response",
        "sf2_lead",
        "instrument.sf2.supra-saw",
        -10.0,
    );
    response.pan = 0.22;
    response.sound = SoundSettings {
        binaural_itd_ms: 0.72,
        ..SoundSettings::default()
    };
    response.effects.delay_beats = Some(0.75);
    response.effects.delay_feedback = 0.19;
    response.effects.delay_wet = 0.12;
    response.effects.reverb_send = 0.30;
    response.effects.chorus_depth = 0.34;
    response
        .patterns
        .push(pattern("hook-response", 8, hook_answer));
    response.clips = clips("hook-response", &[(25, 64), (89, 30), (119, 74)]);
    response.automation = automation_curve(
        "reverb_send",
        &[
            (25, 0.28),
            (49, 0.38),
            (65, 0.48),
            (89, 0.36),
            (119, 0.46),
            (145, 0.54),
            (151, 0.38),
            (161, 0.30),
        ],
    );
    song.tracks.push(response);

    // A written, syncopated chord riff supplies the harmonic motion. The
    // Cm–Ab–Bb–G progression changes every two bars; short voicing stabs use
    // complementary rhythms in each bar instead of long sustained pads.
    let progression = [
        [root + 12, root + 15, root + 19], // Cm
        [root + 12, root + 15, root + 20], // Ab/C
        [root + 10, root + 14, root + 17], // Bb
        [root + 7, root + 11, root + 14],  // G
    ];
    let mut riff_notes = Vec::new();
    for (chord_index, chord) in progression.iter().enumerate() {
        let first_bar = chord_index as u32 * 2 + 1;
        for (beat, voices, length, velocity) in [
            (1.75, [0, 1, 2].as_slice(), 0.42, 66),
            (3.25, [1, 2].as_slice(), 0.30, 53),
            (4.25, [0, 1, 2].as_slice(), 0.34, 59),
        ] {
            for &voice in voices {
                riff_notes.push(note(
                    first_bar,
                    beat,
                    chord[voice],
                    length,
                    (velocity as u8).saturating_add(5),
                ));
            }
        }
        let second_bar = first_bar + 1;
        for (beat, voices, length, velocity) in [
            (1.5, [0, 2].as_slice(), 0.35, 57),
            (2.75, [0, 1, 2].as_slice(), 0.30, 64),
            (4.5, [1, 2].as_slice(), 0.28, 51),
        ] {
            for &voice in voices {
                riff_notes.push(note(
                    second_bar,
                    beat,
                    chord[voice],
                    length,
                    (velocity as u8).saturating_add(5),
                ));
            }
        }
    }
    let mut color = track(
        "color",
        "Syncopated chord riff",
        "sf2_chords",
        "instrument.sf2.ghost-mult",
        -7.0,
    );
    color.pan = -0.12;
    color.sound = SoundSettings {
        binaural_itd_ms: 0.35,
        ..SoundSettings::default()
    };
    color.effects.delay_beats = Some(0.375);
    color.effects.delay_feedback = 0.16;
    color.effects.delay_wet = 0.12;
    color.effects.reverb_send = 0.12;
    color.patterns.push(pattern("chord-riff", 8, riff_notes));
    color.clips = clips(
        "chord-riff",
        &[(25, 12), (49, 40), (89, 30), (119, 32), (161, 16)],
    );
    song.tracks.push(color);

    let pad_chords = [
        [root, root + 7, root + 12, root + 15],
        [root + 8, root + 12, root + 15, root + 20],
        [root + 10, root + 14, root + 17, root + 22],
        [root + 7, root + 11, root + 14, root + 17],
    ];
    let mut air_notes = Vec::new();
    for (index, chord) in pad_chords.iter().enumerate() {
        let bar = index as u32 * 2 + 1;
        for &pitch in chord {
            air_notes.push(note(bar, 1.0, pitch, 7.8, 46));
        }
    }
    let mut air = track(
        "atmosphere",
        "Dark harmonic atmosphere",
        "pad",
        "instrument.sf2.synth-pad",
        -15.5,
    );
    air.pan = 0.04;
    air.sound = SoundSettings {
        binaural_itd_ms: 0.62,
        ..SoundSettings::default()
    };
    air.effects.highpass_hz = Some(180.0);
    air.effects.lowpass_hz = Some(4_800.0);
    air.effects.reverb_send = 0.48;
    air.effects.chorus_depth = 0.12;
    air.effects.chorus_rate_hz = 0.12;
    air.patterns.push(pattern("air-bed", 8, air_notes));
    air.clips = clips("air-bed", &[(1, 88), (89, 30), (119, 74)]);
    air.automation = automation_curve(
        "reverb_send",
        &[
            (25, 0.36),
            (49, 0.45),
            (73, 0.52),
            (89, 0.58),
            (119, 0.48),
            (151, 0.56),
            (161, 0.42),
            (193, 0.32),
        ],
    );
    song.tracks.push(air);

    song.master.output_gain_db = 0.0;
    song.master.stereo_width = 1.85;
    song.master.low_band_width = 0.0;
    song.master.compressor_threshold_db = -8.0;
    song.master.compressor_ratio = 1.25;
    song.master.clipper_drive_db = 0.0;
    song.personality = ["noir", "hypnotic", "industrial", "acid"][archetype].into();
}

fn hook_phrase(root: i32, variant: usize) -> Vec<Note> {
    // Degree offsets relative to C5: root, third, fifth, color tone. Each
    // chord has its own targets, so a note can answer the bass change rather
    // than merely obeying a global scale filter.
    let chord_tones: [[i32; 4]; 4] = [[0, 3, 7, 2], [-4, 0, 3, 7], [-2, 2, 5, 7], [-5, -1, 2, 5]];
    let rhythms: [&[(f64, usize, f64, u8)]; 4] = [
        &[(1.5, 0, 0.28, 88), (2.75, 2, 0.42, 101), (4.5, 1, 0.25, 82)],
        &[(1.75, 2, 0.30, 91), (3.0, 1, 0.38, 102), (4.5, 0, 0.22, 81)],
        &[
            (1.5, 1, 0.25, 88),
            (2.5, 3, 0.30, 94),
            (3.75, 2, 0.42, 102),
            (4.75, 0, 0.16, 76),
        ],
        &[
            (2.0, 0, 0.32, 90),
            (3.25, 2, 0.44, 101),
            (4.75, 3, 0.17, 79),
        ],
    ];
    let mut notes = Vec::new();
    for bar_index in 0usize..16 {
        let bar = bar_index as u32 + 1;
        let cycle = bar_index / 8;
        let chord = (bar_index / 2) % chord_tones.len();
        let rhythm_id = (variant + cycle + (bar_index / 4) % 2) % rhythms.len();
        for (event_index, &(beat, degree, length, velocity)) in
            rhythms[rhythm_id].iter().enumerate()
        {
            // Leave the first beat of selected answer bars to the bass.
            if cycle == 1 && bar_index % 4 == 1 && event_index == 0 {
                continue;
            }
            let degree = (degree + cycle + if bar_index % 4 >= 2 { 1 } else { 0 }) % 4;
            let pitch = root + 24 + chord_tones[chord][degree];
            notes.push(note(bar, beat, pitch, length, velocity));
        }
        if cycle == 1 && bar_index % 4 == 3 {
            notes.push(note(bar, 4.75, root + 24 + chord_tones[chord][3], 0.14, 74));
        }
    }
    notes
}

fn rolling_bass_phrase(root: i32, archetype: usize, variation: bool) -> Vec<Note> {
    let groove: [(f64, i32, f64, u8); 4] = match archetype {
        0 => [
            (1.5, 0, 0.42, 84),
            (2.5, 0, 0.34, 68),
            (3.5, 0, 0.42, 78),
            (4.5, 4, 0.34, 72),
        ],
        1 => [
            (1.5, 0, 0.40, 82),
            (2.5, 0, 0.34, 67),
            (3.5, 0, 0.42, 77),
            (4.5, 4, 0.32, 70),
        ],
        2 => [
            (1.5, 0, 0.44, 86),
            (2.5, 4, 0.32, 72),
            (3.5, 0, 0.40, 80),
            (4.5, 0, 0.30, 68),
        ],
        _ => [
            (1.5, 0, 0.38, 80),
            (2.5, 0, 0.32, 66),
            (3.5, 4, 0.40, 77),
            (4.5, 0, 0.34, 72),
        ],
    };
    let mut notes = Vec::new();
    for bar in 1..=8 {
        for (index, &(beat, degree, gate, velocity)) in groove.iter().enumerate() {
            if (bar + index as u32).is_multiple_of(8) && index == 1 {
                continue;
            }
            let shift = if variation && bar % 2 == 0 && index == 3 {
                -0.125
            } else {
                0.0
            };
            let degree = if variation && bar % 4 == 0 && index == 3 {
                0
            } else {
                degree
            };
            let accent = if variation && index == 1 { 3 } else { 0 };
            notes.push(note(
                bar,
                beat + shift,
                root - 12 + interval(degree),
                gate,
                velocity.saturating_sub(accent),
            ));
        }
    }
    notes
}

fn hook_answer(root: i32, variant: usize) -> Vec<Note> {
    let rows: &[&[(f64, i32, f64, u8)]] = &[
        &[
            (1.5, 17, 0.72, 62),
            (2.75, 22, 0.58, 58),
            (4.0, 24, 0.72, 66),
        ],
        &[
            (1.25, 24, 0.70, 64),
            (2.5, 22, 0.54, 58),
            (4.0, 19, 0.80, 62),
        ],
        &[
            (1.5, 17, 0.68, 60),
            (2.75, 20, 0.56, 58),
            (4.0, 24, 0.72, 66),
        ],
        &[
            (1.25, 27, 0.70, 67),
            (2.5, 24, 0.58, 61),
            (4.0, 22, 0.82, 64),
        ],
        &[
            (1.5, 19, 0.70, 62),
            (2.75, 24, 0.56, 64),
            (4.0, 27, 0.72, 68),
        ],
        &[
            (1.25, 29, 0.68, 67),
            (2.5, 27, 0.58, 64),
            (4.0, 24, 0.82, 66),
        ],
        &[
            (1.5, 22, 0.70, 62),
            (2.75, 24, 0.56, 65),
            (4.0, 27, 0.72, 70),
        ],
        &[
            (1.25, 24, 0.64, 64),
            (2.5, 22, 0.56, 60),
            (4.0, 19, 1.10, 62),
        ],
    ];
    let mut notes = Vec::new();
    for (bar_index, events) in rows.iter().enumerate() {
        let bar = bar_index as u32 + 1;
        for &(beat, degree, length, velocity) in *events {
            let shift = if variant == 3 && bar_index % 2 == 1 {
                -0.125
            } else {
                0.0
            };
            notes.push(note(
                bar,
                beat + shift,
                root + degree,
                length,
                velocity.saturating_add(4),
            ));
        }
    }
    notes
}

fn track(id: &str, name: &str, role: &str, sample: &str, gain_db: f32) -> Track {
    let mut effects = Effects::default();
    if matches!(role, "sf2_bass") {
        effects.lowpass_hz = Some(1_200.0);
    }
    Track {
        source_type: "auto".into(),
        id: id.into(),
        name: name.into(),
        role: role.into(),
        gain_db,
        pan: 0.0,
        muted: false,
        bus_id: None,
        effects,
        sound: SoundSettings::default(),
        plugin_parameters: Default::default(),
        sample_id: Some(sample.into()),
        sample_alt_id: None,
        sample_path: None,
        sample_alt_path: None,
        sample_tune_semitones: 0.0,
        drum: DrumSettings::default(),
        patterns: Vec::new(),
        clips: Vec::new(),
        automation: Vec::<AutomationPoint>::new(),
    }
}

fn pattern(id: &str, bars: u32, notes: Vec<Note>) -> Pattern {
    Pattern {
        id: id.into(),
        bars,
        notes,
    }
}

fn clips(id: &str, ranges: &[(u32, u32)]) -> Vec<Clip> {
    ranges
        .iter()
        .enumerate()
        .map(|(i, &(start_bar, bars))| Clip {
            id: format!("{id}-section-{}", i + 1),
            pattern_id: id.into(),
            start_bar,
            bars,
            transpose: 0,
        })
        .collect()
}

fn automation_curve(parameter: &str, points: &[(u32, f32)]) -> Vec<AutomationPoint> {
    points
        .iter()
        .map(|&(bar, value)| AutomationPoint {
            parameter: parameter.into(),
            bar,
            beat: 1.0,
            value,
        })
        .collect()
}

fn section(name: &str, start_bar: u32, bars: u32, energy: f32) -> Section {
    Section {
        name: name.into(),
        start_bar,
        bars,
        energy,
    }
}

fn note(bar: u32, beat: f64, pitch: i32, length_beats: f64, velocity: u8) -> Note {
    Note {
        bar,
        beat,
        tick: 0,
        pitch: pitch.clamp(0, 127) as u8,
        length_beats,
        velocity,
        channel: 1,
        synth_overrides: None,
        analysis_evidence: None,
    }
}

fn interval(degree: i32) -> i32 {
    match degree {
        0 => 0,
        1 => 2,
        2 => 3,
        3 => 5,
        4 => 7,
        5 => 8,
        6 => 10,
        _ => 0,
    }
}

fn key_pitch(key: &str, octave: i32) -> i32 {
    let pc = match key.trim().to_ascii_lowercase().as_str() {
        "c" => 0,
        "c#" | "db" => 1,
        "d" => 2,
        "d#" | "eb" => 3,
        "e" => 4,
        "f" => 5,
        "f#" | "gb" => 6,
        "g" => 7,
        "g#" | "ab" => 8,
        "a" => 9,
        "a#" | "bb" => 10,
        "b" => 11,
        _ => 0,
    };
    12 * (octave + 1) + pc
}
