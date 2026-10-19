# Dragonoff 0.2 — headless club production engine

Dragonoff is an agent-controlled music composition and rendering engine. Its Rust MCP server edits musical projects, synthesizes and samples sounds, mixes independent tracks through stereo groups, and renders audio without a DAW UI. Native synthesis, sample playback, mixing, and mastering use the Rust engine. An optional offline Python backend renders VST3 instruments.

## Clone and build

```sh
git clone https://github.com/crivero-0b101100/dragonoff.git
cd dragonoff
cargo build --release
```

Dragonoff requires Rust 1.87 or newer. The native engine needs no audio device, DAW, or Python installation. The server uses MCP over standard input/output (stdio); clients should launch one process per MCP connection. It does not expose an HTTP endpoint.

To check a clone, run `cargo test` and `cargo clippy --all-targets -- -D warnings`.

Point the MCP client at the compiled binary. For clients that accept JSON server configuration, the entry looks like this (replace the absolute paths):

```json
{
  "mcpServers": {
    "dragonoff": {
      "command": "/absolute/path/to/dragonoff/target/release/dragonoff",
      "args": [],
      "env": {
        "DRAGONOFF_SESSION": "/absolute/path/to/session.dragonoff.json"
      }
    }
  }
}
```

`DRAGONOFF_SESSION` is optional. It points to the active project file for session continuity. Without it, create a project with `dragonoff.project_create` and pass the desired project path. MCP tools use the `dragonoff.*` namespace.

For development, `cargo run --release` starts the same stdio server. Requests are newline-delimited MCP JSON-RPC; `initialize`, `tools/list`, and `tools/call` are supported.

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"dragonoff.project_render","arguments":{"output":"/absolute/path/master.wav","stems_directory":"/absolute/path/stems"}}}
```

Projects are editable JSON. Existing projects still load, but the repaired gain staging, EQ, sampler and noise source change their sound. Remix old projects before treating them as final masters. New projects use schema version 2.

## Independent sound and mixing controls

`dragonoff.instrument_catalog` describes the native sources and editable patches. `dragonoff.sample_catalog` lists installed sample IDs and license provenance, or scans an external pack directory for WAV paths; pass the pack's license label along with the directory so the source remains clear. The server cannot verify that license claim. `dragonoff.vst_parameter_catalog` exposes a plugin's exact parameter IDs; `track_set.plugin_parameters` can override any normalized VST parameter per track, so the VST backend is not limited to its role starting patches. The `generate_arrangement` tool is an optional starter. Agents can instead create a project from scratch with `track_add`, write arbitrary notes with `pattern_import_json`, place clips, and design the form directly. The `club_drive` profile remains available as a reproducible starting point.

`dragonoff.patch_apply` deliberately replaces a sound with a complete preset. `dragonoff.track_set` merges partial sound/effect/drum changes into the current settings, preserving untouched settings. Unknown setting keys are rejected. Setting an effect highpass/lowpass or sample ID to `null` clears it. `dragonoff.track_clone` makes an independently mixed layer, with optional copied musical content.

A track's `source_type` selects `auto`, `synth` or `one_shot` independently of its semantic role. Drum roles no longer receive a hidden procedural pattern during rendering: `auto` renders only written note events, using the selected sample when present; `one_shot` explicitly triggers its sample. A `kick` can also use a phase-reset synthesizer with an editable pitch envelope. A role no longer inserts sub oscillators, changes synthesizer amplitude, or reduces gain when another oscillator layer is added. Oscillator layers sum at explicit levels. Unison uses an energy normalization; actual level still depends on phase, waveform, pitch and filters.

Synth controls include amplitude/pitch/filter envelopes, pitch reset, continuous-time cutoff/amplitude/pitch LFO, block-automated pitch-LFO depth, bandlimited saw/pulse/triangle sources, independent oscillator tuning, PWM and true two-operator FM ratio/index/decay. Unison voices are spread in stereo before independent left/right filtering. Tracks can use musical-time equal-power stereo orbit with a short interaural delay cue for headphone-oriented, 8D-like movement; this is not a full HRTF renderer. Orbit depth/rate and pitch-LFO depth are automatable, so movement can build or recede with an arrangement. A release starts at the envelope level at note-off. Sample playback retains stereo, uses cubic interpolation when reading at or below the source rate, and uses a 25-tap windowed-sinc lowpass during downsampling. PCM16/24/32 and float32 WAV sources are supported. Only zones needed by active, transposed sample notes are loaded.

`one_shot` tracks can trigger an installed sample or an external WAV on each explicit MIDI note in either renderer. Set `sample_id` to an installed ID or a unique external ID and set `sample_path` to an absolute path, or to a path relative to the engine's bundled asset directory. Use `sample_alt_id`/`sample_alt_path` for alternating hits. External files are read in place and are not copied into project archives; retain their source and license information with the project.

## EQ, dynamics and group routing

Tracks, buses and the master accept up to twelve additional parametric bands. Supported kinds are `bell`, `low_shelf`, `high_shelf`, `notch`, `highpass`, and `lowpass`. EQ uses double precision RBJ coefficients and states. A dynamic bell uses a linked band detector, threshold, ratio, maximum cut, attack and release. It reduces the selected frequency range instead of turning down the whole channel. The corrected shelf equations follow the [W3C Audio EQ Cookbook](https://www.w3.org/TR/audio-eq-cookbook/).

```json
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"dragonoff.track_set","arguments":{"track_id":"mid-bass","bus_id":"lowend","effects":{"eq":[{"kind":"highpass","frequency_hz":90},{"kind":"bell","frequency_hz":320,"q":1.1,"gain_db":-1,"dynamic_threshold_db":-24,"dynamic_ratio":2,"dynamic_range_db":3,"attack_ms":10,"release_ms":120}],"sidechain":true,"sidechain_source":"kick-body","sidechain_amount":0.4,"sidechain_release_ms":90}}}}
```

Create a bus with `dragonoff.bus_set`, then route a track with `bus_id`. `null` sends directly to the master. Every track has one destination; routed tracks are not also added to the master. Buses have their own EQ, dynamic EQ, gain, width, linked RMS compression and saturation. Time effects belong on independent tracks; unsupported bus time-effect settings fail explicitly. Buses are parallel subgroups; serial/nested group routing is not implemented.

Ducking follows the actual audio of a named source track, independent of track-processing order. It defaults to the first kick if no source is given. It does not pump on a grid when the source is silent. The detector uses the source before its inserts and fader. Compressors have soft knee, explicit makeup and optional detector highpass. Soft saturation is gain compensated and uses antiderivative antialiasing; hard/foldback modes remain deliberate unfiltered distortion colors.

A damped diffuse stereo room has editable decay, pre-delay, damping, filtering and return level through `dragonoff.reverb_set`. Chorus uses fractional delay reads and a wet/dry blend. Track automation controls gain, effect lowpass, reverb send, delay wet/feedback, chorus depth, pitch envelope/LFO depth and rate, and stereo-orbit depth/rate. Pitch-LFO rate is most phase-stable when kept fixed while depth is automated.

The `dragonoff.generate_arrangement` tool is an optional phrase-first starter. It writes explicit drum hits and melodic phrases, but is not the only composition path and does not run automatically when a project is created. For user-directed compositions, the agent can author every track, phrase, clip, sound patch, effect curve, and section through the MCP editing tools. Rendering plays only written events.

## Mastering and exports

The master signal path is high/low stereo management, corrective/tone and dynamic EQ, linked RMS glue compression, a 4x FIR-oversampled gain-compensated soft clipper, makeup gain, and a linked lookahead limiter with a smoothed attack and a 4x FIR reconstructed-peak detector and 0.2 dB safety guard. FIR and lookahead delays are compensated so the musical grid and exported duration stay aligned.

The detector is an engineering true-peak estimate, not a certified BS.1770 device. Independent export verification is still appropriate before commercial delivery. The engine does not perform automatic reference matching or claim professional results from a loudness number.

The final master is PCM24 with TPDF dither. `master_bypass: true` exports the premaster as float32. Optional stems are aligned float32 WAVs: each processed track, each processed bus, the shared room return, and the full premaster. Sum the bus stems and room return to reconstruct a fully routed premaster. Track stems come before their group processing and cannot reconstruct a nonlinear bus by simple summing. Completed WAVs replace their destination atomically; an aborted render does not replace a previously completed WAV.

Each render writes `<output>.mix.json`: track/bus/master peak, RMS, crest factor, DC offset, stereo correlation/side level, six band-center energy probes, compressor gain reduction, limiter average/maximum reduction, output format and latency information. Band probes are not a full spectral masking model.

`dragonoff.audio_measure` reads PCM16/24/32 or float32 stereo exports and reports signal properties such as approximate gated K-weighted loudness, estimated reconstructed peak, clipping and stereo energy. It does not score musical quality, enforce an instrument list, or compare against genre profiles. Musical decisions remain with the producer and should be judged by listening.

`dragonoff.track_analyze` reads a mono or multichannel PCM16/24/32 or float32 RIFF/WAVE file and estimates tempo, major/minor key, a 4/4 beat/bar grid, per-bar chord candidates and chord progression, low/mid/high spectral balance, frequency-band transient events, and four-bar relative-energy sections. These built-in estimates need no trained model. Bars assume the file starts on beat one. Chords and transient labels are estimates from the full mix, not stem-separated transcription or verified drum identities.

For a deeper optional analysis, set `include_ml: true` and `output_directory` in `dragonoff.track_analyze`. Install `python -m pip install -r tools/requirements-track-analysis.txt` in a dedicated Python environment and set `DRAGONOFF_ANALYSIS_PYTHON` to that environment's Python executable. The optional pass uses pretrained Demucs `htdemucs` to write drums, bass, other, and vocals WAV stems; Basic Pitch to write note-event CSV and MIDI files for the bass, other, and vocal stems; and beat-aligned chord and drum-event estimates. Schema 4 adds compressed per-note envelopes and harmonic profiles, bass pitch-bend trajectories, per-hit drum decay and low-frequency pitch trajectories, and event-level synth overrides that Dragonoff applies while rendering bass and drum notes. `synth_reconstruction_spec` describes functional synth roles rather than requiring real instrument identities, includes Dragonoff-format patterns, and estimates native synth layers and per-event oscillator, filter, pitch-envelope, noise, and amplitude-envelope controls. These values are evidence-based heuristic fits, not recovered original settings. Source separation can leak sounds between stems; note and drum-role confidence is not calibrated, chord margins are not probabilities, and the chord part remains a sustained triad per bar without source chord rhythm or voicing.

Use `dragonoff.track_reconstruct` with the generated `track_analysis_ml.json` to create a project whose parts all use Dragonoff's native synth. Render it with `dragonoff.project_render`, then call `dragonoff.track_reconstruction_compare` with the source and rendered WAVs. The comparison reports alignment for duration, tempo, key/chords, bar energy, spectral balance, loudness, and stereo width, with suggested adjustments for another automated pass. These signal comparisons guide iteration; they do not guarantee an exact source reconstruction.

## Validation

```sh
cargo test
cargo clippy --all-targets -- -D warnings
```

Regression coverage includes EQ response at multiple sample rates, selective dynamic EQ, gain staging, sidechain recovery, latency, release continuity, sampler behavior, group reconstruction, export alignment and MCP edits. The old template-generator regression was removed with the generator it described.

## License and Ardour relationship

GPL-2.0-or-later. Bundled samples retain their individual licenses and attributions in `assets/samples/manifest.json` and `assets/samples/ASSET_NOTICES.md`. Newly designed native patches are project data, with no newly downloaded recordings. Dragonoff does not embed Ardour's engine; it adopted the one-based region-relative bar/beat/tick MIDI JSON convention described in `THIRD_PARTY_NOTICES.md`.

## Optional offline VST3 instrument backend

`tools/plugin_render.py` renders instrument plugins through Spotify Pedalboard, without a UI or audio device. It preserves plugin voice state across bars and sends explicit note-offs. `tools/surge_render.py` hosts Surge XT for the optional plugin backend. Role patches are starting sounds; raw per-track `plugin_parameters` override them, and the plugin catalog exposes exact IDs to the agent. Installed and external one-shot WAVs are supported. `dragonoff.project_render` accepts `backend: "surge_xt"` and `plugin_path`; install `pedalboard`, `numpy`, `scipy`, and `soundfile` in the Python environment and set `DRAGONOFF_PYTHON` if `python3` does not resolve to it. The default `backend: "native"` uses the Rust renderer and does not require Python.

Install `pedalboard`, `numpy`, `soundfile`, and `mido` into a local virtual environment, then point the backend at a compatible, separately obtained VST3 plugin:

```sh
python tools/plugin_render.py /absolute/path/session.json /absolute/path/stems --plugin /absolute/path/Synth.vst3
```

The six-minute Last Light session uses TAL-NoiseMaker 5.0.6 with custom patches. TAL's installer was extracted locally; no plugin was installed into system folders and no Ardour executable was run. Plugin binaries are not redistributed with this project. The session uses 48 kHz processing, the plugin's oversampled filter, and a 192 kHz mastering limiter; its 128 kHz WAV is a delivery resample. Increasing the delivery sample rate does not change the composition or recover source detail.

Arrangement generation is an optional starting point; MCP clients can author every track, note, clip, sound, sample source, and section directly. Engineering measurements do not establish musical quality.

## End-to-end MCP smoke check

`tools/e2e_mcp_smoke.py` starts the compiled stdio server, discovers sample and VST controls, builds an agent-authored project, imports independent note patterns, places clips, and renders through both native and VST backends. It also checks that external sample paths render and a per-track plugin parameter change alters the exported audio. Run it with an installed Surge XT VST3 and a Python environment containing Pedalboard, NumPy, SciPy, SoundFile, and Mido:

```sh
cargo build
python tools/e2e_mcp_smoke.py \
  --binary target/debug/dragonoff \
  --plugin /absolute/path/Surge\ XT.vst3 \
  --python /absolute/path/to/python \
  --sample "$PWD/assets/samples/kicks/punch-04.wav" \
  --alternate-sample "$PWD/assets/samples/kicks/deep-02.wav"
```

The smoke check runs in a temporary directory and does not leave rendered files in the project.
