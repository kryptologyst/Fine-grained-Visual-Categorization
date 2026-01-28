"""Main training script for fine-grained visual categorization."""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.utils.config import Config, get_default_config
from src.utils.utils import set_seed, get_device, create_logger, ensure_dir
from src.models.fine_grained_models import create_model
from src.data.dataset import create_dataloaders
from src.train.trainer import Trainer
from src.eval.evaluator import Evaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fine-grained Visual Categorization Training')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Path to dataset directory')
    parser.add_argument('--dataset', type=str, default='cub200',
                       help='Dataset name (cub200, cars, aircraft)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet50',
                       help='Model name (resnet50, resnet101, vit_base_patch16_224)')
    parser.add_argument('--num_classes', type=int, default=200,
                       help='Number of classes')
    parser.add_argument('--attention', type=str, default='cbam',
                       help='Attention mechanism (cbam, se, eca, none)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       help='Learning rate scheduler')
    parser.add_argument('--loss_type', type=str, default='cross_entropy',
                       help='Loss function type')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                       help='Mixup alpha parameter')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, mps, cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--experiment_name', type=str, default='fine_grained_classification',
                       help='Experiment name')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate the model')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(args.output_dir) / args.experiment_name
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    assets_dir = output_dir / "assets"
    
    ensure_dir(checkpoint_dir)
    ensure_dir(log_dir)
    ensure_dir(assets_dir)
    
    # Create logger
    logger = create_logger(
        name="fine_grained_training",
        log_file=str(log_dir / "training.log")
    )
    
    logger.info(f"Starting experiment: {args.experiment_name}")
    logger.info(f"Arguments: {args}")
    
    # Load or create config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = get_default_config()
        
        # Update config with command line arguments
        config.data.data_dir = args.data_dir
        config.data.dataset_name = args.dataset
        config.data.batch_size = args.batch_size
        config.data.num_workers = args.num_workers
        config.data.image_size = args.image_size
        config.data.mixup_alpha = args.mixup_alpha
        
        config.model.name = args.model
        config.model.num_classes = args.num_classes
        config.model.attention_type = args.attention
        config.model.pretrained = args.pretrained
        config.model.loss_type = args.loss_type
        
        config.training.epochs = args.epochs
        config.training.learning_rate = args.lr
        config.training.weight_decay = args.weight_decay
        config.training.scheduler = args.scheduler
        
        config.seed = args.seed
        config.output_dir = str(output_dir)
        config.log_dir = str(log_dir)
        config.checkpoint_dir = str(checkpoint_dir)
        config.experiment_name = args.experiment_name
        config.resume = args.resume
        config.eval_only = args.eval_only
    
    # Save config
    config.save(str(output_dir / "config.yaml"))
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, classes = create_dataloaders(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        image_size=config.data.image_size,
        crop_size=config.data.crop_size,
        augmentation=config.data.augmentation
    )
    
    logger.info(f"Number of classes: {len(classes)}")
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    logger.info(f"Creating model: {config.model.name}")
    model = create_model(
        model_name=config.model.name,
        num_classes=len(classes),
        attention_type=config.model.attention_type,
        dropout=config.model.dropout,
        pretrained=config.model.pretrained
    )
    
    # Log model info
    from src.utils.utils import get_model_size
    model_info = get_model_size(model)
    logger.info(f"Model parameters: {model_info['parameters_millions']:.2f}M")
    logger.info(f"Model size: {model_info['size_mb']:.2f}MB")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        logger=logger
    )
    
    if config.eval_only:
        # Load checkpoint
        if config.resume:
            logger.info(f"Loading checkpoint: {config.resume}")
            checkpoint = torch.load(config.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        # Evaluate
        logger.info("Evaluating model...")
        evaluator = Evaluator(model, device, classes)
        results = evaluator.evaluate(val_loader)
        
        logger.info(f"Evaluation Results:")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Precision: {results['precision']:.4f}")
        logger.info(f"Recall: {results['recall']:.4f}")
        logger.info(f"F1-Score: {results['f1_score']:.4f}")
        
        # Plot confusion matrix
        evaluator.plot_confusion_matrix(
            results['confusion_matrix'],
            save_path=str(assets_dir / "confusion_matrix.png")
        )
        
        # Plot per-class accuracy
        evaluator.plot_per_class_accuracy(
            results['per_class_accuracy'],
            save_path=str(assets_dir / "per_class_accuracy.png")
        )
        
    else:
        # Train model
        logger.info("Starting training...")
        trainer.train()
        
        # Final evaluation
        logger.info("Final evaluation...")
        evaluator = Evaluator(model, device, classes)
        results = evaluator.evaluate(val_loader)
        
        logger.info(f"Final Results:")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Precision: {results['precision']:.4f}")
        logger.info(f"Recall: {results['recall']:.4f}")
        logger.info(f"F1-Score: {results['f1_score']:.4f}")
        
        # Plot confusion matrix
        evaluator.plot_confusion_matrix(
            results['confusion_matrix'],
            save_path=str(assets_dir / "confusion_matrix.png")
        )
        
        # Plot per-class accuracy
        evaluator.plot_per_class_accuracy(
            results['per_class_accuracy'],
            save_path=str(assets_dir / "per_class_accuracy.png")
        )
    
    logger.info("Experiment completed!")


if __name__ == "__main__":
    main()
