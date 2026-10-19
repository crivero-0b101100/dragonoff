// Copyright (C) 2026 OpenAI
// This program is free software under the GNU GPL v2 or (at your option) any later version.

#![recursion_limit = "256"]

mod analysis;
mod audio;
mod composer;
mod dsp;
mod mcp;
mod midi;
mod model;
mod render;

fn main() {
    if let Err(e) = mcp::serve() {
        eprintln!("dragonoff: {e}");
        std::process::exit(1);
    }
}
