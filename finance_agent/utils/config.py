"""
Central configuration loader.
Reads from .env (secrets) and config.yaml (settings).
"""

import os
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv
import yaml

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT_DIR / ".env"
CONFIG_FILE = ROOT_DIR / "config.yaml"

load_dotenv(ENV_FILE)


@dataclass
class LLMConfig:
    provider: str = "google"
    model: str = "gemini-3.5-flash"       # Primary — good quality, higher quota
    temperature: float = 0.2
    max_tokens: int = 8192
    # Fallback cascade: tried in order when primary hits 404/daily-quota
    model_cascade: list = field(default_factory=lambda: [
        "gemini-3.5-flash",       # Tier 1 — primary
        "gemini-3.1-flash-lite",  # Tier 2 — lighter, very high quota
        "gemini-3.6-flash",       # Tier 3 — last resort (20/day)
    ])


@dataclass
class DatabaseConfig:
    url: str = f"sqlite:///{ROOT_DIR / 'finance_agent.db'}"
    echo: bool = False


@dataclass
class DataConfig:
    market_data_dir: Path = ROOT_DIR / "finance_agent" / "data" / "market_data"
    fundamentals_dir: Path = ROOT_DIR / "finance_agent" / "data" / "fundamentals"
    earnings_dir: Path = ROOT_DIR / "finance_agent" / "data" / "earnings"
    macro_dir: Path = ROOT_DIR / "finance_agent" / "data" / "macro"
    news_dir: Path = ROOT_DIR / "finance_agent" / "data" / "news"
    documents_dir: Path = ROOT_DIR / "finance_agent" / "data" / "documents"


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    benchmark: str = "^GSPC"          # S&P 500 by default
    default_horizon_months: int = 12
    max_position_size_pct: float = 10.0   # % of portfolio
    max_sector_exposure_pct: float = 30.0


def _load_yaml_overrides(cfg: AppConfig) -> None:
    """Apply overrides from config.yaml if it exists."""
    if not CONFIG_FILE.exists():
        return
    with open(CONFIG_FILE, "r") as f:
        raw = yaml.safe_load(f) or {}

    llm = raw.get("llm", {})
    if llm.get("provider"):
        cfg.llm.provider = llm["provider"]
    if llm.get("model"):
        cfg.llm.model = llm["model"]
        # Primary model is always first in cascade
        if llm["model"] not in cfg.llm.model_cascade:
            cfg.llm.model_cascade.insert(0, llm["model"])
        else:
            # Ensure primary is first
            cfg.llm.model_cascade.remove(llm["model"])
            cfg.llm.model_cascade.insert(0, llm["model"])
    if llm.get("temperature") is not None:
        cfg.llm.temperature = float(llm["temperature"])

    app = raw.get("app", {})
    if app.get("benchmark"):
        cfg.benchmark = app["benchmark"]
    if app.get("default_horizon_months"):
        cfg.default_horizon_months = int(app["default_horizon_months"])


def load_config() -> AppConfig:
    cfg = AppConfig()
    _load_yaml_overrides(cfg)

    # Inject secrets from environment
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        import warnings
        warnings.warn("GEMINI_API_KEY not set in .env — LLM calls will fail.")

    return cfg


# Singleton — import this everywhere
settings: AppConfig = load_config()

__all__ = ["settings", "load_config", "AppConfig"]
