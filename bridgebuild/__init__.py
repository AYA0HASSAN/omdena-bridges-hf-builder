"""bridgebuild: build the fAIr Bridges training dataset (OSM/BD TOPO bridges over open imagery,
256 px z18 chips) in small batches and store it on the Hugging Face Hub."""
__version__ = "1.0.0"


def load_config(path, **overrides):
    import os
    import yaml
    cfg = yaml.safe_load(open(path))
    if os.environ.get("BB_REPO_ID"):
        cfg.setdefault("hub", {})["repo_id"] = os.environ["BB_REPO_ID"]
    if os.environ.get("BB_LOCAL_DIR"):
        cfg.setdefault("hub", {})["local_dir"] = os.environ["BB_LOCAL_DIR"]
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg
