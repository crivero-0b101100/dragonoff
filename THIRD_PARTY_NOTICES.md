# Third-party notices

## Ardour MIDI convention

The bulk MIDI JSON timing and repeated-bar interface is compatible with the region-relative convention implemented by Ardour's MCP HTTP surface, including one-based bars and beats, tick offsets, and repeat entries. The Rust parser in `src/midi.rs` is a small independent implementation of that interface and is not copied from Ardour source files.

Ardour is licensed under the GNU General Public License, version 2 or (at your option) any later version. Ardour source and license: [github.com/Ardour/ardour](https://github.com/Ardour/ardour). Dragonoff is distributed under GPL-2.0-or-later; see `LICENSE`.

## CC0 sample assets

The curated one-shot sample subset in `assets/samples/` comes from the `house-core` pack in [Octal Samples](https://github.com/octalmusic/octal-samples), pinned at commit `22d222ce36e290ba3792c776811aa9c69dcc8776`. The source pack is dedicated under CC0-1.0. See `assets/samples/LICENSE`, `assets/samples/ASSET_NOTICES.md`, and the per-file source/processing metadata in `assets/samples/manifest.json`.

## CC0 synthesized instrument zones

The `ghost-mult`, `saw-morph`, `supra-saw`, and `fuzz-bass` pitched zones in `assets/samples/instruments/sf2/` were rendered from John Oestmann's JOM-MECHSOUNDS SoundFont. The author states that all sounds are released as Public Domain / CC0 on the [MechSounds source page](https://johnoestmannmusic.com/mechsounds/). These derived files are 48 kHz mono PCM16, pitch-rooted and peak-matched for Dragonoff's built-in sample player. FluidSynth 2.6.1 was used to render the source presets; the engine does not require FluidSynth at runtime. Per-zone preset, source program, transformation, and SHA-256 metadata are in `assets/samples/manifest.json`.
