#!/usr/bin/env python3
"""
Fast Training Script for Project Results
Optimized for quick training and comprehensive results reporting
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class PotholeDataset(Dataset):
    """Simple dataset class for pothole classification"""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.classes = ['no_pothole', 'pothole']
        self.samples = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class SimplePotholeNet(nn.Module):
    """Lightweight CNN for pothole detection"""
    
    def __init__(self, num_classes=2):
        super(SimplePotholeNet, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def create_data_loaders(data_dir, batch_size=16):
    """Create train/val/test data loaders"""
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = PotholeDataset(data_dir, 'train', train_transform)
    val_dataset = PotholeDataset(data_dir, 'val', val_transform)
    test_dataset = PotholeDataset(data_dir, 'test', val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, device, epochs=15):
    """Train the model"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    print("Starting training...")
    print("="*60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        print(f'Epoch {epoch+1:2d}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_acc:.2f}%')
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    test_accuracy = np.mean(np.array(all_predictions) == np.array(all_targets)) * 100
    
    # Classification report
    class_names = ['No Pothole', 'Pothole']
    report = classification_report(all_targets, all_predictions, 
                                 target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    return {
        'accuracy': test_accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'targets': all_targets
    }

def plot_training_results(history, save_path='training_results.png'):
    """Plot training history"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(history['train_losses'], label='Training Loss', color='blue')
    ax1.plot(history['val_losses'], label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['val_accuracies'], label='Validation Accuracy', color='green')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training plots saved to: {save_path}")

def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Pothole', 'Pothole'],
                yticklabels=['No Pothole', 'Pothole'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Confusion matrix saved to: {save_path}")

def generate_project_summary(history, test_results, model_info):
    """Generate comprehensive project summary"""
    
    summary = {
        "project_title": "ROS-Based Pothole Detection System",
        "model_architecture": "Custom CNN (SimplePotholeNet)",
        "dataset": {
            "source": "Kaggle Annotated Potholes Dataset",
            "total_images": 1196,
            "training_images": 765,
            "validation_images": 192,
            "test_images": 239,
            "classes": ["No Pothole", "Pothole"]
        },
        "training_results": {
            "epochs_trained": len(history['val_accuracies']),
            "best_validation_accuracy": f"{history['best_val_acc']:.2f}%",
            "final_training_loss": f"{history['train_losses'][-1]:.4f}",
            "final_validation_loss": f"{history['val_losses'][-1]:.4f}"
        },
        "test_results": {
            "test_accuracy": f"{test_results['accuracy']:.2f}%",
            "precision_pothole": f"{test_results['classification_report']['1']['precision']:.3f}",
            "recall_pothole": f"{test_results['classification_report']['1']['recall']:.3f}",
            "f1_score_pothole": f"{test_results['classification_report']['1']['f1-score']:.3f}",
            "precision_no_pothole": f"{test_results['classification_report']['0']['precision']:.3f}",
            "recall_no_pothole": f"{test_results['classification_report']['0']['recall']:.3f}",
            "f1_score_no_pothole": f"{test_results['classification_report']['0']['f1-score']:.3f}"
        },
        "model_details": {
            "total_parameters": model_info['parameters'],
            "model_size_mb": f"{model_info['size_mb']:.2f} MB",
            "inference_optimized": "Raspberry Pi compatible"
        },
        "performance_metrics": {
            "overall_accuracy": f"{test_results['accuracy']:.2f}%",
            "macro_avg_precision": f"{test_results['classification_report']['macro avg']['precision']:.3f}",
            "macro_avg_recall": f"{test_results['classification_report']['macro avg']['recall']:.3f}",
            "macro_avg_f1": f"{test_results['classification_report']['macro avg']['f1-score']:.3f}"
        }
    }
    
    # Save to file
    with open('project_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def print_final_results(summary):
    """Print final results for project presentation"""
    
    print("\n" + "="*80)
    print("🎯 POTHOLE DETECTION PROJECT - FINAL RESULTS")
    print("="*80)
    
    print(f"📊 Model Performance:")
    print(f"   • Test Accuracy: {summary['test_results']['test_accuracy']}")
    print(f"   • Pothole Detection F1-Score: {summary['test_results']['f1_score_pothole']}")
    print(f"   • Overall Precision: {summary['performance_metrics']['macro_avg_precision']}")
    print(f"   • Overall Recall: {summary['performance_metrics']['macro_avg_recall']}")
    
    print(f"\n📈 Training Details:")
    print(f"   • Best Validation Accuracy: {summary['training_results']['best_validation_accuracy']}")
    print(f"   • Epochs Trained: {summary['training_results']['epochs_trained']}")
    print(f"   • Final Training Loss: {summary['training_results']['final_training_loss']}")
    
    print(f"\n🔧 Model Specifications:")
    print(f"   • Architecture: {summary['model_architecture']}")
    print(f"   • Parameters: {summary['model_details']['total_parameters']:,}")
    print(f"   • Model Size: {summary['model_details']['model_size_mb']}")
    print(f"   • Target Platform: {summary['model_details']['inference_optimized']}")
    
    print(f"\n📋 Dataset Information:")
    print(f"   • Source: {summary['dataset']['source']}")
    print(f"   • Total Images: {summary['dataset']['total_images']:,}")
    print(f"   • Training/Validation/Test: {summary['dataset']['training_images']}/{summary['dataset']['validation_images']}/{summary['dataset']['test_images']}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE - Results ready for project presentation!")
    print("="*80)

def main():
    """Main training function"""
    
    # Configuration
    data_dir = './data/processed/classification'
    batch_size = 16
    epochs = 15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🚀 Fast Training for Project Results")
    print("="*60)
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print("="*60)
    
    # Check data directory
    if not Path(data_dir).exists():
        print("❌ Dataset not found! Please run the dataset preparation scripts first.")
        return False
    
    # Create data loaders
    print("📁 Loading dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(data_dir, batch_size)
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(val_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\n🧠 Creating model...")
    model = SimplePotholeNet(num_classes=2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    model_info = {
        'parameters': total_params,
        'size_mb': model_size_mb
    }
    
    print(f"   Parameters: {total_params:,}")
    print(f"   Model size: {model_size_mb:.2f} MB")
    
    # Train model
    print(f"\n🏃 Training model for {epochs} epochs...")
    start_time = time.time()
    trained_model, history = train_model(model, train_loader, val_loader, device, epochs)
    training_time = time.time() - start_time
    
    print(f"\n⏱️ Training completed in {training_time:.1f} seconds")
    print(f"🎯 Best validation accuracy: {history['best_val_acc']:.2f}%")
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_results = evaluate_model(trained_model, test_loader, device)
    
    # Save model
    print("\n💾 Saving model...")
    Path('models').mkdir(exist_ok=True)
    torch.save(trained_model.state_dict(), 'models/pothole_detection_model.pth')
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    plot_training_results(history)
    plot_confusion_matrix(test_results['confusion_matrix'])
    
    # Generate project summary
    print("\n📋 Generating project summary...")
    summary = generate_project_summary(history, test_results, model_info)
    
    # Print final results
    print_final_results(summary)
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All done! Check the generated files for your project presentation.")
    else:
        print("\n❌ Training failed. Please check the error messages above.")