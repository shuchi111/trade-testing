import tradingagents.default_config as default_config
from typing import Dict, Optional

_config: Optional[Dict] = None


def initialize_config():
    global _config
    if _config is None:
        _config = default_config.DEFAULT_CONFIG.copy()


def set_config(config: Dict):
    global _config
    if _config is None:
        _config = default_config.DEFAULT_CONFIG.copy()
    _config.update(config)


def get_config() -> Dict:
    if _config is None:
        initialize_config()
    return _config.copy()


initialize_config()
