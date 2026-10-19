# Curated sample asset notices

The selected assets are from [Octal Samples](https://github.com/octalmusic/octal-samples) at commit `22d222ce36e290ba3792c776811aa9c69dcc8776`. The source project dedicates the pack under CC0-1.0; see `LICENSE`. Per-file origin and processing notes are retained in `manifest.json`.

| ID | Role | Character | Root | SHA-256 |
|---|---|---|---|---|
| `house.kick.punch-04` | kick | punchy, bright, hard | A#1 | `082b96236aa25596176dc7de53236481c2417e348888630a7ce1318df0c69f99` |
| `house.hat.closed-06` | hat_closed | dark, tight | — | `b9a5b709e9795b3686b851fd7fc2e82a4369d2c3f8e792045de81c6ca067b2fb` |
| `house.hat.closed-04` | hat_closed | short, clicky | — | `34539d73d6a026dbc07dfe4d4f4c97561fd1c9dbf4cae8580b18b137a10a0651` |
| `house.hat.open-04` | hat_open | crunchy, open | — | `f67f257a3958a200c57028a2c378efc78f36541674c9531b39ae7c8a90ba881b` |
| `house.snare.snap-01` | snare | snappy, short | — | `c94d564db184b1bea0e3ea3a6d544fe8fa1d934495f08e2a7b7fd51c2bcba960` |
| `house.perc.tom-01` | perc | low, round | — | `5c1d5c933008d648e22d634a2596065dd05ff6c83ce4590da36bd15c98d4b7b3` |
| `house.perc.tom-02` | perc | mid, round | — | `9fccf1f5f6d4a1abcbde3219f59cf62cb1238dfcc8fb63afcae2de56553a0763` |
| `house.perc.tick-02` | perc | woody, tight | — | `e9339ced3de90fb27e27700e0c536ca425a8bb3ae05d759cd242416b26d3be55` |
| `house.perc.shaker-02` | perc | driving, tight | — | `7ff28d7e238f3318f8a3ab6fbdfef8cedfa0ef4d6a6f45d17be4009d13dd94f6` |
| `house.ride.ping-01` | ride | metallic, bright | — | `c33c5da57c06f7ed12a0be3b228a0310377eaa16b3496a7005d37631b0d73240` |
| `house.kick.punch-02` | kick | punchy, solid | G1 | `a3a100ddcfc126bf75c282f54c3ac2a054ed09e6d0a393620a7226169b27b7f9` |
| `house.hat.closed-08` | hat_closed | crunchy, bright | — | `db9e6da61a706a5d9fa1191b2e2143e64a966c4a02f8eb4935e6aaf6f0bc8d68` |
| `house.hat.closed-05` | hat_closed | loose, airy | — | `60490494d934c80e9e75a33910f4e7be7a74f74b10ef614f53f167772268f071` |
| `house.hat.open-03` | hat_open | airy, long | — | `0efceadad9e7ef780510207f9a8b31cf5ff225af559539368ec624b1e67aa2c0` |
| `house.clap.classic-02` | clap | wide, roomy | — | `53d98144934f6b90f16f2bcfd00bef82522c392d6c7f23c86a5f1488cb9cddcd` |

## Added sampled instruments and low kick

- `instruments/piano/`: selected Meta Piano notes C3, G3, C4, F4, A4, D5, G5 and C6 at velocity layers 2 and 4. Source: [Meta-open-sound Meta Piano](https://github.com/facebookincubator/Meta-open-sound), CC0-1.0, commit `e2ff154845994fe4135becc4d5b857c19f172784`. The source OGG previews were converted to mono PCM16 WAV and trimmed to 3.4 seconds with a 160 ms fade to limit load and memory.
- `instruments/flute/`: ten mono note zones from C3 through C6. Source: [Wind Studio flute samples](https://huggingface.co/AEmotionStudio/windstudio-flute-samples), derived from VSCO 2 Community Edition LDFlute susvib, CC0-1.0. The 48 kHz 24-bit WAVs were converted to mono PCM16.
- `instruments/guitar/`: ten nylon-guitar note zones from E2 through B5, each with three velocity-selected articulations (soft fingerstyle, light pick, firm pick). Source: [Meta Open Sound Nylon Guitar](https://github.com/facebookincubator/Meta-open-sound/tree/main/systems/instruments/meta-nylon-guitar), CC0-1.0, source commit `e2ff154845994fe4135becc4d5b857c19f172784`. The 48 kHz mono OGG previews were converted to PCM16 WAV and normalized zone by zone to consistent peaks (maximum gain 12x, ceiling -0.9 dBFS); the sampler loads these 30 zones only for a guitar track.
- `instruments/steel-guitar/`: nine steel-string note zones from A2 through C6 with two velocity layers. Source: [Meta Open Sound Steel Guitar](https://github.com/facebookincubator/Meta-open-sound/tree/main/systems/instruments/meta-steel-guitar), CC0-1.0, source commit `e2ff154845994fe4135becc4d5b857c19f172784`. Mono OGG previews were converted to standard 48 kHz PCM16 WAV; only zones required by notes in an active steel-guitar track are loaded.
- `kicks/deep-02.wav`: D2 warm, round kick from [Octal Samples house-core](https://github.com/octalmusic/octal-samples), CC0-1.0, commit `22d222ce36e290ba3792c776811aa9c69dcc8776`.

The bundled player loads only the sampled-instrument zones required by active instrument roles. Piano uses two velocity layers and eight root notes; flute selects among ten recorded root notes; nylon guitar selects among ten root notes and three dynamic colors; steel guitar selects among nine notes and two dynamics. All transformed assets and hashes are recorded in `manifest.json`.

## Synth instrument zones

- `instruments/sf2/ghost-mult/`, `instruments/sf2/saw-morph/`, `instruments/sf2/supra-saw/`, and `instruments/sf2/fuzz-bass/`: four pitch-root zones per program rendered from the author's JOM-MECHSOUNDS SoundFont, including the TNL-SUPRASAW preset (bank 10, program 34). The source sound bank and WAV samples are dedicated to the public domain / CC0 by John Oestmann; the source page says all sounds are CC0 and project files are MIT. Source: [MechSounds](https://johnoestmannmusic.com/mechsounds/). The compact zones were rendered using FluidSynth 2.6.1 at 48 kHz, downmixed to mono PCM16, trimmed to 3.5 seconds, peak-matched, and given short edge fades. Runtime playback has no FluidSynth dependency. Every zone's preset, source bank/program, transformation, and SHA-256 are listed in `manifest.json`.
