import yaml
import os
from pathlib import Path

class Config:
    """
    Centralized configuration manager for the CineEmotion pipeline.
    Parses Kaggle-optimized hyperparameters from config.yaml.
    """
    def __init__(self, config_path: str = "config.yaml"):
        # Resolve the absolute path to ensure it works anywhere
        base_dir = Path(__file__).resolve().parent
        full_path = base_dir / config_path
        
        with open(full_path, "r") as f:
            self._config = yaml.safe_load(f)
            
        # Parse sections for easier attribute access
        self.directories = self._config.get("directories", {})
        self.hardware = self._config.get("hardware", {})
        self.module1 = self._config.get("module1_encoder", {})
        self.module2_3 = self._config.get("module2_3_narrative", {})
        self.module4 = self._config.get("module4_planner", {})
        
        # Initialize project paths dynamically
        self.data_dir = base_dir / self.directories.get("data_dir", "../data")
        self.checkpoint_dir = base_dir / self.directories.get("checkpoint_dir", "./checkpoints")
        self.log_dir = base_dir / self.directories.get("log_dir", "./logs")
        
        # Self-initialize required folders
        self._setup_dirs()

    def _setup_dirs(self):
        """Creates the necessary output directories if they don't exist."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

# Expose a singleton instance for importing directly
# e.g. from config import pipeline_config
pipeline_config = Config()
