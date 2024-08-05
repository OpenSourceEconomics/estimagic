from pathlib import Path

import toml

file_path = Path(__file__).parent.parent.resolve() / "pyproject.toml"

with file_path.open("r") as f:
    config = toml.load(f)

config["project"]["name"] = "estimagic"

with file_path.open("w") as f:
    toml.dump(config, f)
