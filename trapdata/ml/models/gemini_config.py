"""
Configuration for Gemini VLM classifier.

Supports multiple configuration discovery patterns:
1. Environment variables (highest priority)
2. Configuration file (XDG_CONFIG_HOME/ami/gemini_vlm.json)
3. Project data directory (uses importlib.resources)
"""

import json
import os
from pathlib import Path

def get_config_dir() -> Path:
    """Get config directory following XDG Base Directory spec."""
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home) / "ami"
    return Path.home() / ".config" / "ami"

def get_data_dir() -> Path:
    """Get data directory for AMI."""
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        return Path(data_home) / "ami"
    return Path.home() / ".local" / "share" / "ami"

def load_config_file(config_path: Path | None = None) -> dict:
    """Load configuration from JSON file."""
    if config_path is None:
        config_path = get_config_dir() / "gemini_vlm.json"
    
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}

def get_gemini_config() -> dict:
    """
    Get Gemini VLM configuration from environment variables or config file.
    
    Priority (highest to lowest):
    1. Environment variables (AMI_GEMINI_*)
    2. Config file (~/.config/ami/gemini_vlm.json)
    3. Raises ValueError if required config missing
    
    Required configuration:
    - api_key: OpenRouter API key
    - labels_path: Path to label_map.json file
    
    Optional configuration:
    - model: Gemini model to use (default: google/gemini-3-flash-preview)
    """
    config = load_config_file()
    
    # Environment variables override config file
    config_env = {
        "api_key": os.environ.get("AMI_OPENROUTER_API_KEY"),
        "model": os.environ.get("AMI_GEMINI_VLM_MODEL"),
        "labels_path": os.environ.get("AMI_GEMINI_VLM_LABELS"),
    }
    
    # Merge with env vars taking precedence
    for key, value in config_env.items():
        if value is not None:
            config[key] = value
    
    # Set defaults for optional fields
    if "model" not in config:
        config["model"] = "google/gemini-3-flash-preview"
    
    # Validate required fields
    if "api_key" not in config or not config["api_key"]:
        raise ValueError(
            "AMI_OPENROUTER_API_KEY must be set via environment variable or "
            f"config file at {get_config_dir() / 'gemini_vlm.json'}"
        )
    
    if "labels_path" not in config or not config["labels_path"]:
        raise ValueError(
            "AMI_GEMINI_VLM_LABELS must be set via environment variable or "
            f"config file at {get_config_dir() / 'gemini_vlm.json'}"
        )
    
    # Validate labels_path exists
    labels_path = Path(config["labels_path"])
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found at: {labels_path}\n"
            f"Set AMI_GEMINI_VLM_LABELS environment variable to correct path"
        )
    
    return config

def create_example_config():
    """Create an example configuration file."""
    example_config = {
        "api_key": "your-openrouter-api-key-here",
        "labels_path": "/path/to/label_map.json",
        "model": "google/gemini-3-flash-preview"
    }
    
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "gemini_vlm.json"
    
    if not config_file.exists():
        with open(config_file, 'w') as f:
            json.dump(example_config, f, indent=2)
        print(f"Created example config at: {config_file}")
    
    return config_file
