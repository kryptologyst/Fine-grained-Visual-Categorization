"""Configuration management for fine-grained visual categorization."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from omegaconf import OmegaConf
import yaml
import os


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_name: str = "cub200"
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    batch_size: int = 32
    num_workers: int = 4
    image_size: int = 224
    crop_size: int = 224
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    augmentation: str = "standard"  # standard, strong, weak
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 200
    dropout: float = 0.1
    attention: bool = False
    attention_type: str = "cbam"  # cbam, se, eca
    loss_type: str = "cross_entropy"  # cross_entropy, focal, label_smoothing
    label_smoothing: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    mixed_precision: bool = True
    device: str = "auto"  # auto, cuda, mps, cpu


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    output_dir: str = "outputs"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "fine_grained_classification"
    resume: Optional[str] = None
    eval_only: bool = False
    debug: bool = False

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def save(self, path: str) -> None:
        """Save configuration."""
        self.to_yaml(path)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def load_config(config_path: str) -> Config:
    """Load configuration from file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        return Config.from_yaml(config_path)
    else:
        raise ValueError(f"Unsupported config format: {config_path}")


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to file."""
    config.save(config_path)
