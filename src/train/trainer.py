"""Training and evaluation modules for fine-grained visual categorization."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.utils import AverageMeter, ProgressMeter, get_lr
from ..data.dataset import mixup_data, mixup_criterion, cutmix_data


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        confidence = 1. - self.smoothing
        logprobs = nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """Get loss function based on type."""
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    elif loss_type == "label_smoothing":
        return LabelSmoothingCrossEntropy(**kwargs)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def get_optimizer(model: nn.Module, optimizer_name: str, lr: float, weight_decay: float, **kwargs) -> optim.Optimizer:
    """Get optimizer based on name."""
    if optimizer_name.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, **kwargs)
    elif optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(optimizer: optim.Optimizer, scheduler_name: str, **kwargs) -> Optional[Any]:
    """Get learning rate scheduler based on name."""
    if scheduler_name.lower() == "cosine":
        return CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_name.lower() == "step":
        return StepLR(optimizer, **kwargs)
    elif scheduler_name.lower() == "plateau":
        return ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler_name.lower() == "none":
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


class Trainer:
    """Trainer class for fine-grained visual categorization."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Any,
        device: torch.device,
        logger: Any
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Initialize loss function
        self.criterion = get_loss_function(
            config.model.loss_type,
            label_smoothing=config.model.label_smoothing
        )
        
        # Initialize optimizer
        self.optimizer = get_optimizer(
            self.model,
            "adamw",
            config.training.learning_rate,
            config.training.weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = get_scheduler(
            self.optimizer,
            config.training.scheduler,
            T_max=config.training.epochs
        )
        
        # Initialize mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.training.mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.best_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(self.current_epoch)
        )
        
        end = time.time()
        
        for i, (images, targets) in enumerate(self.train_loader):
            # Measure data loading time
            data_time.update(time.time() - end)
            
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Apply mixup/cutmix if enabled
            if hasattr(self.config.data, 'mixup_alpha') and self.config.data.mixup_alpha > 0:
                images, targets_a, targets_b, lam = mixup_data(images, targets, self.config.data.mixup_alpha)
            
            # Compute output
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    if hasattr(self.config.data, 'mixup_alpha') and self.config.data.mixup_alpha > 0:
                        loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                    else:
                        loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                if hasattr(self.config.data, 'mixup_alpha') and self.config.data.mixup_alpha > 0:
                    loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = self.criterion(outputs, targets)
            
            # Compute accuracy
            if hasattr(self.config.data, 'mixup_alpha') and self.config.data.mixup_alpha > 0:
                acc1, acc5 = self._accuracy(outputs, targets_a, topk=(1, 5))
            else:
                acc1, acc5 = self._accuracy(outputs, targets, topk=(1, 5))
            
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            # Compute gradient and do optimizer step
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.config.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
                self.optimizer.step()
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % 100 == 0:
                progress.display(i)
        
        return losses.avg, top1.avg
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(self.val_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: '
        )
        
        end = time.time()
        
        with torch.no_grad():
            for i, (images, targets) in enumerate(self.val_loader):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Compute output
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                # Measure accuracy
                acc1, acc5 = self._accuracy(outputs, targets, topk=(1, 5))
                
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                
                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                if i % 100 == 0:
                    progress.display(i)
        
        return losses.avg, top1.avg
    
    def _accuracy(self, output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
        """Computes the accuracy over the k top predictions for the specified values of k."""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
    
    def train(self) -> None:
        """Train the model."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Save checkpoint if best
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                checkpoint_path = f"{self.config.checkpoint_dir}/best_model.pth"
                self._save_checkpoint(checkpoint_path)
                self.logger.info(f"New best model saved with accuracy: {val_acc:.2f}%")
            
            # Save last checkpoint
            checkpoint_path = f"{self.config.checkpoint_dir}/last_model.pth"
            self._save_checkpoint(checkpoint_path)
        
        self.logger.info(f"Training completed. Best accuracy: {self.best_acc:.2f}%")
    
    def _save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)


class Evaluator:
    """Evaluator class for fine-grained visual categorization."""
    
    def __init__(self, model: nn.Module, device: torch.device, classes: List[str]):
        self.model = model
        self.device = device
        self.classes = classes
        self.model.eval()
    
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Evaluate the model on a dataset."""
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc="Evaluating"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Compute metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'per_class_accuracy': per_class_acc,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None) -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_per_class_accuracy(self, per_class_acc: np.ndarray, save_path: Optional[str] = None) -> None:
        """Plot per-class accuracy."""
        plt.figure(figsize=(15, 8))
        plt.bar(range(len(self.classes)), per_class_acc)
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.xticks(range(len(self.classes)), self.classes, rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
