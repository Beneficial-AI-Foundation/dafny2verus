from pathlib import Path
import yaml


def load_config() -> dict:
    """Load configuration from prompts.yml"""
    config_path = Path(".") / "config.yml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


_cfg = load_config()
cfg = _cfg.get("config", {})
system_prompt = _cfg.get("system", "")

ARTIFACTS = Path(cfg.get("artifacts_dir", "artifacts"))
ARTIFACTS.mkdir(parents=True, exist_ok=True)
