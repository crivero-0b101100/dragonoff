# Dragonoff agent guide

Dragonoff is an MCP stdio music engine. Build it with `cargo build --release`, then configure your MCP client to launch `target/release/dragonoff` from this checkout. See the README for a portable JSON configuration example. Each client connection should launch its own process; the stdio server is not an HTTP service.

## Working with a project

- Start with `dragonoff.project_create` for a new project or `dragonoff.project_get` for an existing one.
- Discover available instruments, samples, and presets with `dragonoff.instrument_catalog`, `dragonoff.sample_catalog`, and `dragonoff.composition_catalog`; use returned IDs and source paths.
- Write musical events explicitly with `dragonoff.pattern_import_json`, place them with `dragonoff.clip_place`, and shape form with `dragonoff.sections_set`. The renderer plays written events; it does not add hidden drum patterns.
- Use `dragonoff.track_set` for partial changes so untouched sound and effect settings stay intact. `dragonoff.patch_apply` replaces the full patch.
- Render with `dragonoff.project_render`, then listen to the audio. `dragonoff.audio_measure` reports technical signal measurements; it does not judge musical quality.
- Keep project files and exports outside source directories unless the user asks to add them to the repository. Preserve sample provenance and license details for external audio.

## Native and optional rendering

Native synthesis, sample playback, mixing, and mastering run in Rust and require no audio device. Surge XT VST3 rendering is optional; it needs a compatible plugin plus the Python dependencies documented in the README. Do not assume that a plugin or external sample pack is installed.
