import yaml
from pathlib import Path
from typing import Optional

class DreamBoothConfig:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        
        self.model = self.config['model']
        self.dataset = self.config['dataset']
        self.training = self.config['training']
        self.prior = self.config['prior_preservation']

    def _load_config(self) -> dict:
        path = Path(self.config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
            
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def __repr__(self):
        return f"DreamBoothConfig(path={self.config_path}, content={self.config})"


