"""
Configuration Management for SmartRep AI Sandbox
Handles API keys, mode selection (simulation vs live), and app settings.

Setup for live mode:
  1. Copy .env.example to .env
  2. Fill in your Azure OpenAI credentials
  3. Set SMARTREP_MODE=live
  4. Run: streamlit run app.py
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Auto-load .env file if present (no extra install needed at runtime)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI connection settings."""
    api_key: str = ""
    azure_endpoint: str = ""
    api_version: str = "2024-12-01-preview"
    model: str = "gpt-4"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.azure_endpoint)


@dataclass
class AppConfig:
    """Top-level application configuration."""
    # Mode: "simulation" uses mock AI responses; "live" calls Azure OpenAI
    mode: str = "simulation"

    # Azure settings (only used when mode == "live")
    azure: AzureOpenAIConfig = field(default_factory=AzureOpenAIConfig)

    # De-identification settings
    use_spacy: bool = False          # If False, uses regex-only lightweight de-ID
    use_scispacy: bool = False       # Biomedical NER for medications

    # Data paths
    synthetic_data_path: str = "data/synthetic_messages.csv"
    saved_scenarios_dir: str = "saved_scenarios"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Build config from environment variables or .env file."""
        mode = os.getenv("SMARTREP_MODE", "simulation")
        cfg = cls(mode=mode)

        cfg.azure.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        cfg.azure.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        cfg.azure.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        cfg.azure.model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4")

        cfg.use_spacy = os.getenv("USE_SPACY", "false").lower() == "true"
        cfg.use_scispacy = os.getenv("USE_SCISPACY", "false").lower() == "true"

        return cfg
