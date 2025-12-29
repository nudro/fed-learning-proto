# Federated Learning with Flower and PyTorch

This project implements federated learning using Flower framework and PyTorch, following the [Flower tutorial format](https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html).

## Project Structure

- `model.py` - ResNet18 model definition
- `task.py` - Training and test functions
- `client.py` - Flower ClientApp implementation
- `server.py` - Flower ServerApp implementation
- `data_utils.py` - Data loading and partitioning utilities
- `run_pipeline.py` - Complete pipeline script (pretraining + federated learning)
- `pyproject.toml` - Flower configuration

## Setup

1. Install dependencies:
```bash
pip install -e .
```

2. Ensure you have the pretrained model (`pretrained_model.pt`) or it will be created automatically.

## Running

### Option 1: Full Pipeline (Pretraining + Federated Learning)
```bash
python run_pipeline.py
```

### Option 2: Skip Pretraining (Use Existing Model)
```bash
python run_pipeline.py --skip-pretrain
```

### Option 3: Direct Federated Learning
```bash
flwr run . local-simulation
```

This will execute the simulation with 5 clients and 3 rounds as configured in `pyproject.toml`.

## Configuration

Edit `pyproject.toml` to adjust:
- Number of clients (`num-supernodes`)
- Training parameters (`num-server-rounds`, `local-epochs`, `lr`)
- Batch size (`batch-size`)

## Features

- **Transfer Learning**: ResNet18 pretrained on ImageNet, fine-tuned for ants/bees classification
- **Heterogeneous Data Distribution**: Uses Dirichlet partitioner (alpha=0.5) for non-IID data splits
- **Flower Datasets Integration**: Leverages `flwr_datasets` for efficient data partitioning
- **Message API**: Implements Flower's Message API following quickstart-pytorch template pattern
- **Complete Pipeline**: End-to-end workflow from pretraining to federated learning

## Pipeline Script

The `run_pipeline.py` script provides a complete workflow:

1. **Pretraining Phase** (optional): Trains ResNet18 with frozen backbone on full dataset
2. **Federated Learning Phase**: Runs federated learning simulation with multiple clients

### Pipeline Options

```bash
# Full pipeline with custom epochs
python run_pipeline.py --epochs 10

# Skip pretraining
python run_pipeline.py --skip-pretrain

# Custom learning rate
python run_pipeline.py --lr 0.0005
```

## Model Architecture

- **Base Model**: ResNet18 (ImageNet pretrained)
- **Task**: Binary classification (ants vs bees)
- **Transfer Learning**: Frozen backbone during pretraining, unfrozen during federated learning

## Data

The project uses the hymenoptera dataset with:
- Training set: 245 images (124 ants, 121 bees)
- Validation set: 153 images
- Heterogeneous partitioning across clients using Dirichlet distribution