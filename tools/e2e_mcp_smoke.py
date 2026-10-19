"""Exercise the local MCP server from project creation through both renderers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import soundfile as sf


class McpProcess:
    def __init__(self, binary: Path, env: dict[str, str]):
        self.proc = subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, env=env,
        )
        self.next_id = 1

    def request(self, method: str, params: dict | None = None) -> dict:
        request_id = self.next_id
        self.next_id += 1
        self.proc.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": request_id, "method": method,
            "params": params or {},
        }) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("MCP server exited without a response: " + self.proc.stderr.read())
        response = json.loads(line)
        if response.get("id") != request_id:
            raise RuntimeError(f"unexpected MCP response: {response}")
        if "error" in response:
            raise RuntimeError(response["error"].get("message", str(response["error"])))
        return response.get("result", {})

    def call(self, name: str, arguments: dict) -> dict:
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            message = result.get("content", [{}])[0].get("text", "MCP tool failed")
            raise RuntimeError(f"{name}: {message}")
        return result.get("structuredContent", {})

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--alternate-sample", type=Path, required=True)
    args = parser.parse_args()

    env = os.environ.copy()
    env["DRAGONOFF_PYTHON"] = str(args.python)
    with tempfile.TemporaryDirectory(prefix="dragonoff-e2e-") as directory:
        temp = Path(directory)
        server = McpProcess(args.binary, env)
        try:
            server.request("initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "e2e-smoke", "version": "1"}})
            definitions = server.request("tools/list").get("tools", [])
            names = {item["name"] for item in definitions}
            for required in {"dragonoff.project_create", "dragonoff.track_add", "dragonoff.track_set",
                             "dragonoff.pattern_import_json", "dragonoff.clip_place", "dragonoff.project_render",
                             "dragonoff.sample_catalog", "dragonoff.vst_parameter_catalog"}:
                assert required in names, f"missing MCP tool {required}"

            project = temp / "agent-authored.dragonoff.json"
            server.call("dragonoff.project_create", {
                "path": str(project), "title": "MCP E2E", "bpm": 120,
                "key": "D", "scale": "major", "sample_rate": 48000, "seed": 12345,
            })
            samples = server.call("dragonoff.sample_catalog", {"type": "kick"})
            assert samples["samples"], "sample catalog returned no kicks"
            external_samples = server.call("dragonoff.sample_catalog", {
                "directory": str(args.sample.parent), "license": "CC0 1.0 (E2E fixture)",
            })
            assert any(item.get("path") == str(args.sample.resolve()) for item in external_samples["samples"])
            assert all(item.get("license") == "CC0 1.0 (E2E fixture)" for item in external_samples["samples"])
            catalog = server.call("dragonoff.vst_parameter_catalog", {
                "plugin_path": str(args.plugin), "prefix": "a_osc_1_",
            })
            parameter_ids = {parameter["id"] for parameter in catalog["parameters"]}
            assert {"a_osc_1_type", "a_osc_1_volume"} <= parameter_ids

            server.call("dragonoff.track_add", {"id": "kick-layer", "name": "Layered Kick", "role": "kick"})
            server.call("dragonoff.track_set", {
                "track_id": "kick-layer", "source_type": "one_shot",
                "sample_id": "e2e.cc0.kick.a", "sample_path": str(args.sample),
                "sample_alt_id": "e2e.cc0.kick.b", "sample_alt_path": str(args.alternate_sample),
                "gain_db": -4.0,
            })
            server.call("dragonoff.pattern_import_json", {
                "track_id": "kick-layer", "pattern_id": "kick-pattern", "bars": 2,
                "midi": {"events": [
                    {"bar": 1, "beat": 1.0, "pitch": 36, "length_beats": 0.2, "velocity": 110},
                    {"bar": 1, "beat": 3.0, "pitch": 36, "length_beats": 0.2, "velocity": 96},
                    {"bar": 2, "beat": 1.0, "pitch": 36, "length_beats": 0.2, "velocity": 102},
                    {"bar": 2, "beat": 3.0, "pitch": 36, "length_beats": 0.2, "velocity": 94},
                ]},
            })
            server.call("dragonoff.clip_place", {
                "track_id": "kick-layer", "clip_id": "kick-clip",
                "pattern_id": "kick-pattern", "start_bar": 1, "bars": 2,
            })

            server.call("dragonoff.track_add", {"id": "lead-layer", "name": "Custom Lead", "role": "lead"})
            server.call("dragonoff.track_set", {
                "track_id": "lead-layer", "source_type": "synth", "gain_db": -9.0,
                "plugin_parameters": {"a_osc_1_type": 0.005, "a_osc_1_volume": 0.82,
                                      "a_filter_1_cutoff": 0.78, "a_amp_eg_attack": 0.08},
                "effects": {"highpass_hz": 180.0, "reverb_send": 0.35, "chorus_depth": 0.24},
            })
            server.call("dragonoff.pattern_import_json", {
                "track_id": "lead-layer", "pattern_id": "lead-pattern", "bars": 2,
                "midi": {"events": [
                    {"bar": 1, "beat": 1.0, "pitch": 69, "length_beats": 0.7, "velocity": 92},
                    {"bar": 1, "beat": 2.25, "pitch": 74, "length_beats": 0.5, "velocity": 76},
                    {"bar": 1, "beat": 3.5, "pitch": 78, "length_beats": 0.8, "velocity": 88},
                    {"bar": 2, "beat": 1.5, "pitch": 76, "length_beats": 0.6, "velocity": 82},
                    {"bar": 2, "beat": 3.0, "pitch": 73, "length_beats": 0.9, "velocity": 94},
                ]},
            })
            server.call("dragonoff.clip_place", {
                "track_id": "lead-layer", "clip_id": "lead-clip",
                "pattern_id": "lead-pattern", "start_bar": 1, "bars": 2,
            })

            native = temp / "native.wav"
            native_report = server.call("dragonoff.project_render", {"output": str(native), "backend": "native"})
            assert native.is_file() and native_report.get("frames", 0) > 0
            native_info = sf.info(native)
            assert native_info.channels == 2 and abs(native_info.duration - 4.0) < 0.01

            surge = temp / "surge.wav"
            surge_report = server.call("dragonoff.project_render", {
                "output": str(surge), "backend": "surge_xt", "plugin_path": str(args.plugin),
            })
            assert surge.is_file() and surge_report.get("duration_seconds") == 4.0
            surge_info = sf.info(surge)
            assert surge_info.channels == 2 and abs(surge_info.duration - 4.0) < 0.01
            first_hash = digest(surge)

            server.call("dragonoff.track_set", {
                "track_id": "lead-layer",
                "plugin_parameters": {"a_osc_1_type": 0.27, "a_filter_1_cutoff": 0.18},
            })
            variant = temp / "surge-variant.wav"
            server.call("dragonoff.project_render", {
                "output": str(variant), "backend": "surge_xt", "plugin_path": str(args.plugin),
            })
            assert digest(variant) != first_hash, "raw plugin changes did not change rendered audio"
            saved = json.loads(project.read_text())
            lead = next(track for track in saved["tracks"] if track["id"] == "lead-layer")
            assert lead["plugin_parameters"]["a_osc_1_type"] == 0.27
            print(json.dumps({
                "status": "passed", "tools": len(names), "sample_catalog_kicks": len(samples["samples"]),
                "external_sample_catalog_entries": len(external_samples["samples"]),
                "vst_parameters_discovered": catalog["count"],
                "native_duration_seconds": native_info.duration,
                "surge_duration_seconds": surge_info.duration,
                "raw_patch_changes_audio": True, "external_samples_rendered": True,
            }, indent=2))
        finally:
            server.close()


if __name__ == "__main__":
    main()
