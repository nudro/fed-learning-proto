# Federated Learning Prototype

A federated learning implementation using CNN transfer learning with heterogeneous data distribution across multiple devices.

## Overview

This project implements federated learning for image classification using a ResNet18-based CNN fine-tuned on the hymenoptera dataset (ants and bees). The system supports:

- **Real device deployment**: Server on DGX Spark, clients on NVIDIA Orin Nano
- **Heterogeneous data distribution**: Non-IID data splits across devices
- **Multiple aggregation strategies**: Configurable federated aggregation algorithms

## Training Approach: Centralized Pretraining vs. Independent Transfer Learning

**Option A: Centralized Pretraining Followed by Federated Learning (Implemented Approach)**

This project employs a two-phase training strategy where transfer learning is first performed centrally on the complete hymenoptera dataset to establish a robust initialization point. The pretraining phase leverages all available data (245 training samples) to train the final fully-connected layer while keeping the ImageNet-pretrained backbone frozen, resulting in a model that has learned general features for distinguishing ants from bees. All clients then begin federated learning from this shared, well-initialized model, allowing them to adapt the entire network (with all layers unfrozen) to their local heterogeneous data distributions. This approach provides superior initialization compared to training from ImageNet alone, ensures all clients share a common feature representation space that facilitates meaningful aggregation, handles data imbalance across clients more effectively, and enables faster convergence since clients can focus on adaptation rather than learning from scratch. The server aggregates these client-specific adaptations, combining knowledge learned across different data distributions into a more robust global model.

**Option B: Independent Transfer Learning per Client (Alternative Approach)**

An alternative approach would involve each client independently performing transfer learning from ImageNet pretrained weights using only their local data subset, followed by federated aggregation of these independently trained models. While this approach is more truly distributed from the start and may be necessary under strict privacy constraints where centralized pretraining is impossible, it suffers from several critical limitations. Each client would have insufficient data (approximately 50 samples per client) to effectively learn the task-specific features, resulting in poor initialization that aggregation cannot fully remedy. Additionally, clients would develop divergent feature representations since they learn from different subsets without a shared starting point, making aggregation less meaningful as the server attempts to average incompatible feature spaces. This approach essentially wastes the benefit of having all data available for initialization and is only recommended when data cannot be centralized or when each client has a sufficiently large dataset (hundreds or thousands of samples) to support effective independent transfer learning.

## Phase 1: Simulations on Local Clients 

Default of 5 clients, using 245 images in the training set: 124 ants, 121 bees, with an average of 49 images per client. We use `alpha=0.5` for the heterogeneity parameter which is moderately non-IID, whereby each client gets an imbalanced class distribution: e.g. Client k: ~70% ants, ~30% bees; Client k+1: ~ 30% ants, ~ 70% bees, etc. It works by using a Dirichlet distribution to split each class across clients (lower `alpha` more extreme imbalance, higher more balanced and closer to IID)

**Running the Simulation**: The `run_pipeline.py` script serves as a convenience wrapper that checks for the pretrained model and runs pretraining if needed, then executes `flwr run .` which reads the configuration from `pyproject.toml`. The script itself doesn't directly parse the TOML file; instead, it delegates to Flower's `flwr run .` command, which automatically reads `pyproject.toml` in the current directory, loads the configuration from `[tool.flwr.app]` and `[tool.flwr.app.config]` sections, and starts the server and clients accordingly.

## Phase 2: Orin Nano Physical Edge Client 

## Phase 3: Mix of Local Simulations and Nano Edge 

## Phase 4: Increase Client Dataset using Generated Images 

## Phase 5: Object Detection