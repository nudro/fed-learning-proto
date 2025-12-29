"""
Data loading and partitioning utilities using Flower Datasets
Following: https://flower.ai/docs/datasets/how-to-use-with-local-data.html
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datasets import load_dataset as hf_load_dataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner


DATA_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def load_datasets(data_dir='./hymenoptera_data'):
    """Load train and validation datasets using torchvision ImageFolder"""
    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x),
            DATA_TRANSFORMS[x]
        )
        for x in ['train', 'val']
    }
    class_names = image_datasets['train'].classes
    return image_datasets, class_names


# Global partitioner instances - will be initialized on first use
_train_partitioner = None
_val_partitioner = None
_train_dataset = None
_val_dataset = None


def _initialize_partitioners(data_dir='./hymenoptera_data', num_clients=2, alpha=0.5):
    """Initialize partitioners for train and val splits"""
    global _train_partitioner, _val_partitioner, _train_dataset, _val_dataset
    
    if _train_partitioner is None or _val_partitioner is None:
        # Ensure absolute path
        data_dir = os.path.abspath(data_dir)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Load dataset using imagefolder format
        # Structure: hymenoptera_data/train/ants/..., hymenoptera_data/train/bees/...
        dataset_dict = hf_load_dataset("imagefolder", data_dir=data_dir)
        
        # Store datasets
        _train_dataset = dataset_dict["train"]
        _val_dataset = dataset_dict["validation"]
        
        # Create partitioner for train split (heterogeneous using Dirichlet)
        _train_partitioner = DirichletPartitioner(
            num_partitions=num_clients,
            partition_by="label",
            alpha=alpha,
            seed=42
        )
        # Assign dataset to partitioner (as per Flower Datasets docs)
        _train_partitioner.dataset = _train_dataset
        
        # For validation, we use IID partitioner
        _val_partitioner = IidPartitioner(num_partitions=num_clients)
        # Assign dataset to partitioner
        _val_partitioner.dataset = _val_dataset
    
    return _train_partitioner, _val_partitioner


def get_client_dataloader(partition_id, data_dir='./hymenoptera_data', 
                         num_clients=2, batch_size=4, alpha=0.5, num_workers=0):
    """Get dataloader for a specific client partition
    
    Args:
        partition_id: Integer partition ID from context.node_config["partition-id"]
        data_dir: Path to hymenoptera_data directory
        num_clients: Number of clients/partitions
        batch_size: Batch size for DataLoader
        alpha: Dirichlet alpha parameter for heterogeneous partitioning
        num_workers: Number of worker processes for DataLoader
    
    Returns:
        train_loader, val_loader, class_names
    """
    # Ensure partition_id is an integer
    partition_id = int(partition_id)
    
    # Initialize partitioners
    train_partitioner, val_partitioner = _initialize_partitioners(
        data_dir=data_dir,
        num_clients=num_clients,
        alpha=alpha
    )
    
    # Load the specific partition for this client
    train_partition = train_partitioner.load_partition(partition_id=partition_id)
    val_partition = val_partitioner.load_partition(partition_id=partition_id)
    
    # Create custom PyTorch Dataset wrapper that applies transforms on-the-fly
    class TransformedImageDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            item = self.dataset[idx]
            image = item["image"]
            label = item["label"]
            # Apply transform to image
            if self.transform:
                image = self.transform(image.convert("RGB"))
            return image, label
    
    train_dataset = TransformedImageDataset(train_partition, DATA_TRANSFORMS['train'])
    val_dataset = TransformedImageDataset(val_partition, DATA_TRANSFORMS['val'])
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    # Get class names from the dataset
    class_names = train_partition.features["label"].names if hasattr(train_partition.features["label"], "names") else ["ants", "bees"]
    
    return train_loader, val_loader, class_names
