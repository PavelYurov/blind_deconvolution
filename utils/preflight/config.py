from pathlib import Path
try: import tomllib
except ModuleNotFoundError: import pip._vendor.tomli as tomllib


def load_pyproject(path: Path = Path("pyproject.toml")) -> dict:
    if not path.exists():
        raise RuntimeError("pyproject.toml not found")

    with path.open("rb") as f:
        return tomllib.load(f)


def resolve_profile_dependencies(cfg: dict, profile: str) -> list[str]:
    tool_cfg = cfg.get("tool", {}).get("preflight", {})
    profiles = tool_cfg.get("profiles", {})

    if not profiles:
        raise RuntimeError("No [tool.preflight.profiles] section found")

    if profile not in profiles:
        raise RuntimeError(f"Unknown profile: {profile}")

    ref = profiles[profile]["dependencies"]

    section = cfg
    for key in ref.split("."):
        section = section[key]

    if not isinstance(section, list):
        raise RuntimeError(
            f"Resolved dependencies for profile '{profile}' are not a list"
        )

    return section