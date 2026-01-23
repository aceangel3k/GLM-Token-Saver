import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    enabled: bool
    name: str
    endpoint: str
    api_key: str = ""
    model: str
    context_length: int
    max_tokens: int
    timeout: int = 120


class RoutingConfig(BaseModel):
    strategy: str = "smart_routing"
    simple_task_threshold: int = 1000
    complexity_keywords: list[str] = Field(default_factory=list)


class SpeculativeDecodingConfig(BaseModel):
    enabled: bool = True
    draft_model: str = "local"
    verify_model: str = "cerebras"
    max_draft_tokens: int = 10
    min_confidence: float = 0.8


class CostTrackingConfig(BaseModel):
    budget_limit: float = 100.0
    alert_threshold: float = 80.0
    cerebras_cost_per_1k_tokens: float = 0.002


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


class LoggingConfig(BaseModel):
    file: str = "logs/api_server.log"
    max_bytes: int = 10485760
    backup_count: int = 5


class Config(BaseModel):
    models: Dict[str, ModelConfig]
    routing: RoutingConfig
    speculative_decoding: SpeculativeDecodingConfig
    cost_tracking: CostTrackingConfig
    server: ServerConfig
    logging: LoggingConfig


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)
    
    return Config(**config_data)


# Global config instance
config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance."""
    global config
    if config is None:
        config = load_config()
    return config
