"""
Complete Pipeline: Pretraining + Federated Learning
Trains a pretrained model first (optional), then runs federated learning
"""

import os
import sys
import time
import argparse
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tempfile import TemporaryDirectory

from model import get_model
from data_utils import load_datasets


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, 
                num_epochs=25, device='cuda'):
    """
    Train model with validation
    
    Args:
        model: PyTorch model
        dataloaders: Dictionary with 'train' and 'val' DataLoaders
        dataset_sizes: Dictionary with dataset sizes
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Device to train on
    
    Returns:
        model: Trained model with best weights loaded
    """
    since = time.time()
    
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
                
                running_loss = 0.0
                running_corrects = 0
                
                # Iterate over data
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        # Backward pass and optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train':
                    scheduler.step()
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # Deep copy the model if it's the best validation accuracy
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
            
            print()
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')
        
        # Load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    
    return model


def run_pretraining(args):
    """Run transfer learning pretraining"""
    print("=" * 80)
    print("STEP 1: TRANSFER LEARNING PRETRAINING")
    print("=" * 80)
    
    # Configuration
    data_dir = args.data_dir
    checkpoint_path = args.checkpoint
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    num_workers = args.workers
    
    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    # Load data
    print("\nLoading datasets...")
    image_datasets, class_names = load_datasets(data_dir=data_dir)
    
    dataloaders = {
        'train': DataLoader(
            image_datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        ),
        'val': DataLoader(
            image_datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    }
    
    dataset_sizes = {
        'train': len(image_datasets['train']),
        'val': len(image_datasets['val'])
    }
    
    print(f"Classes: {class_names}")
    print(f"Train samples: {dataset_sizes['train']}")
    print(f"Val samples: {dataset_sizes['val']}")
    
    # Create model with frozen backbone (transfer learning)
    print("\nCreating model...")
    model = get_model(
        num_classes=2,
        pretrained=True,  # Use ImageNet pretrained weights
        freeze_backbone=True  # Freeze all layers except FC
    )
    model = model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer - only optimize the final FC layer
    optimizer_conv = optim.SGD(
        model.fc.parameters(),
        lr=learning_rate,
        momentum=0.9
    )
    
    # Learning rate scheduler
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_conv,
        step_size=7,
        gamma=0.1
    )
    
    # Train model
    print("\nStarting training...")
    model = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer_conv,
        scheduler=exp_lr_scheduler,
        num_epochs=num_epochs,
        device=device
    )
    
    # Save final model
    print(f"\nSaving model to {checkpoint_path}...")
    torch.save(model.state_dict(), checkpoint_path)
    print("✓ Pretraining complete!")
    print(f"✓ Model saved to: {checkpoint_path}")


def run_federated_learning(args):
    """Run federated learning"""
    print("\n" + "=" * 80)
    print("STEP 2: FEDERATED LEARNING")
    print("=" * 80)
    
    # Check if pretrained model exists
    if not args.skip_pretrain and not os.path.exists(args.checkpoint):
        print(f"⚠ Warning: Pretrained model not found at {args.checkpoint}")
        print("  Federated learning will use ImageNet pretrained weights as fallback")
    
    # Run federated learning
    print("\nStarting federated learning simulation...")
    print(f"Configuration: {args.num_clients} clients, {args.num_rounds} rounds")
    
    try:
        # Run flwr command
        result = subprocess.run(
            ['flwr', 'run', '.', 'local-simulation'],
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        print("\n✓ Federated learning completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Federated learning failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("\n✗ Error: 'flwr' command not found. Make sure Flower is installed.")
        print("  Try: conda activate fed && pip install -e .")
        return False


def main():
    """Main pipeline function"""
    parser = argparse.ArgumentParser(
        description='Complete Pipeline: Pretraining + Federated Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
Examples:
  # Run full pipeline (pretrain + FL)
  python run_pipeline.py
  
  # Skip pretraining, run FL only
  python run_pipeline.py --skip-pretrain
  
  # Custom pretraining epochs
  python run_pipeline.py --epochs 10
        """
    )
    
    # Pretraining arguments
    parser.add_argument('--skip-pretrain', action='store_true',
                        help='Skip pretraining step and go directly to federated learning')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of pretraining epochs (default: 25)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for pretraining (default: 4)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for pretraining (default: 0.001)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--data-dir', type=str, default='./hymenoptera_data',
                        help='Path to hymenoptera_data directory (default: ./hymenoptera_data)')
    parser.add_argument('--checkpoint', type=str, default='./pretrained_model.pt',
                        help='Path to save/load pretrained model (default: ./pretrained_model.pt)')
    
    # Federated learning arguments (for info display)
    parser.add_argument('--num-clients', type=int, default=5,
                        help='Number of federated learning clients (from pyproject.toml)')
    parser.add_argument('--num-rounds', type=int, default=3,
                        help='Number of federated learning rounds (from pyproject.toml)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("FEDERATED LEARNING PIPELINE")
    print("=" * 80)
    print(f"Pretraining: {'SKIPPED' if args.skip_pretrain else 'ENABLED'}")
    print(f"Federated Learning: {args.num_clients} clients, {args.num_rounds} rounds")
    print("=" * 80)
    
    # Step 1: Pretraining (optional)
    if not args.skip_pretrain:
        run_pretraining(args)
    else:
        print("\n⏭ Skipping pretraining step")
        if os.path.exists(args.checkpoint):
            print(f"✓ Using existing pretrained model: {args.checkpoint}")
        else:
            print(f"⚠ No pretrained model found at {args.checkpoint}")
            print("  Federated learning will use ImageNet pretrained weights as fallback")
    
    # Step 2: Federated Learning
    success = run_federated_learning(args)
    
    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    if args.skip_pretrain:
        print("✓ Pretraining: Skipped")
    else:
        print(f"✓ Pretraining: Completed (model saved to {args.checkpoint})")
    
    if success:
        print("✓ Federated Learning: Completed successfully")
        print(f"✓ Final model saved to: final_model.pt")
    else:
        print("✗ Federated Learning: Failed")
    
    print("=" * 80)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
