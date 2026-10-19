"""Emit stable VST3 parameter IDs and normalized values for an MCP caller."""
from __future__ import annotations

import argparse
import json

from pedalboard import load_plugin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plugin")
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()
    plugin = load_plugin(args.plugin)
    parameters = []
    for parameter_id, parameter in sorted(plugin.parameters.items()):
        if args.prefix and not parameter_id.startswith(args.prefix):
            continue
        item = {
            "id": parameter_id,
            "name": parameter.name,
            # Pedalboard exposes normalized reads through the plugin property,
            # while parameter.raw_value is a write-only setter. Keep the
            # catalog schema independent of that implementation detail.
            "current_value": str(getattr(plugin, parameter.python_name)),
            "normalized_range": [0.0, 1.0],
        }
        # Continuous plugin controls can expose thousands of stepped legal
        # values. Only return compact categorical choices in the discovery
        # response; numeric controls are writable at any normalized value.
        if parameter.valid_values and all(isinstance(value, str) for value in parameter.valid_values):
            item["choices"] = parameter.valid_values
        parameters.append(item)
    print(json.dumps({"plugin": args.plugin, "count": len(parameters), "parameters": parameters}))


if __name__ == "__main__":
    main()
