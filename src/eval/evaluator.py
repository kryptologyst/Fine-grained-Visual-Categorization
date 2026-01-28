"""Evaluation module for fine-grained visual categorization."""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator:
    """Evaluator class for fine-grained visual categorization."""
    
    def __init__(self, model: torch.nn.Module, device: torch.device, classes: List[str]):
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
    
    def get_top_k_accuracy(self, dataloader: torch.utils.data.DataLoader, k: int = 5) -> float:
        """Compute top-k accuracy."""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc=f"Computing Top-{k} Accuracy"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                _, pred = outputs.topk(k, dim=1)
                
                correct += pred.eq(targets.view(-1, 1).expand_as(pred)).sum().item()
                total += targets.size(0)
        
        return correct / total
    
    def analyze_predictions(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Analyze prediction patterns."""
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc="Analyzing predictions"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Analyze confidence
        max_probs = np.max(all_probabilities, axis=1)
        confidence_stats = {
            'mean_confidence': np.mean(max_probs),
            'std_confidence': np.std(max_probs),
            'min_confidence': np.min(max_probs),
            'max_confidence': np.max(max_probs)
        }
        
        # Analyze per-class performance
        per_class_stats = {}
        for i, class_name in enumerate(self.classes):
            class_mask = all_targets == i
            if np.sum(class_mask) > 0:
                class_predictions = all_predictions[class_mask]
                class_targets = all_targets[class_mask]
                class_probs = all_probabilities[class_mask]
                
                accuracy = np.mean(class_predictions == class_targets)
                confidence = np.mean(np.max(class_probs, axis=1))
                
                per_class_stats[class_name] = {
                    'accuracy': accuracy,
                    'confidence': confidence,
                    'samples': np.sum(class_mask)
                }
        
        return {
            'confidence_stats': confidence_stats,
            'per_class_stats': per_class_stats,
            'overall_accuracy': accuracy_score(all_targets, all_predictions)
        }
