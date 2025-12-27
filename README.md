# Federated Learning Prototype

A federated learning implementation using CNN transfer learning with heterogeneous data distribution across multiple devices.

## Overview

This project implements federated learning for image classification using a ResNet18-based CNN fine-tuned on the hymenoptera dataset (ants and bees). The system supports:

- **Real device deployment**: Server on DGX Spark, clients on NVIDIA Orin Nano
- **Heterogeneous data distribution**: Non-IID data splits across devices
- **Multiple aggregation strategies**: Configurable federated aggregation algorithms

## Research: Federated Learning Aggregation Strategies

### 1. FedAvg (Federated Averaging)

**Algorithm**: Weighted average of client model updates
- **Formula**: $w_{global} = \frac{\sum_{i=1}^{n} n_i \cdot w_i}{\sum_{i=1}^{n} n_i}$
- Where $n_i$ is the number of samples on client $i$ and $w_i$ is the model weights from client $i$

**Characteristics**:
- Simple and widely adopted
- Works well with IID (independent and identically distributed) data
- Standard baseline for federated learning
- Assumes all clients participate equally

**Use Case**: Baseline approach, good starting point for homogeneous data distributions

**References**:
- McMahan, B., et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS.

---

### 2. FedProx (Federated Proximal)

**Algorithm**: Adds a proximal term to local loss function to prevent client drift
- **Local Loss**: $L_{local} = L_{original} + \frac{\mu}{2} \|w - w_{global}\|^2$
- Where $\mu$ is the proximal parameter controlling regularization strength

**Characteristics**:
- Better convergence with heterogeneous (non-IID) data
- Reduces client drift by keeping local updates close to global model
- More stable training with varying client data distributions
- Requires tuning of $\mu$ parameter

**Use Case**: Non-IID data distributions, heterogeneous client capabilities

**References**:
- Li, T., et al. (2020). "Federated Optimization in Heterogeneous Networks." MLSys.

---

### 3. FedNova (Normalized Averaging)

**Algorithm**: Normalizes local updates by the number of local training steps
- **Formula**: $w_{global} = \frac{\sum_{i=1}^{n} \tau_i \cdot w_i}{\sum_{i=1}^{n} \tau_i}$
- Where $\tau_i$ is the number of local epochs/steps for client $i$

**Characteristics**:
- Handles clients with different numbers of local training epochs
- More fair aggregation when clients train for varying durations
- Better convergence when client compute capabilities differ
- Accounts for training effort differences

**Use Case**: Clients with varying compute capabilities, different local epoch counts

**References**:
- Wang, J., et al. (2020). "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization." NeurIPS.

---

### 4. FedOpt (Federated Optimization)

**Algorithm**: Uses adaptive optimizers (Adam, AdamW) on server instead of simple averaging
- **Server Update**: Uses Adam/AdamW optimizer to update global model
- Combines client updates using adaptive learning rates

**Characteristics**:
- Faster convergence compared to FedAvg
- Better handling of non-convex optimization landscapes
- More compute-intensive on server side
- Requires tuning optimizer hyperparameters

**Use Case**: Faster convergence needed, non-convex loss landscapes

**References**:
- Reddi, S., et al. (2021). "Adaptive Federated Optimization." ICLR.

---

### 5. Scaffold (Stochastic Controlled Average)

**Algorithm**: Maintains control variates to correct for client drift
- **Control Variates**: $c_i$ per client, $c$ global
- **Client Update**: $w_i^{t+1} = w_i^t - \eta (g_i^t + c - c_i)$
- **Server Update**: Aggregates with control variate corrections

**Characteristics**:
- Excellent for highly non-IID data
- Reduces variance in client updates
- Requires maintaining additional state (control variates)
- More complex implementation

**Use Case**: Highly heterogeneous data, significant client drift issues

**References**:
- Karimireddy, S. P., et al. (2020). "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning." ICML.

---

### 6. FedAvgM (FedAvg with Momentum)

**Algorithm**: Adds momentum to server-side aggregation
- **Momentum Update**: $v_t = \beta v_{t-1} + (1-\beta) \Delta w_t$
- **Model Update**: $w_{t+1} = w_t + v_t$

**Characteristics**:
- Simple extension of FedAvg
- Can improve convergence speed
- Smooths out update variations
- Minimal additional complexity

**Use Case**: Quick improvement over FedAvg, smoother convergence

---

## Comparison Matrix

| Strategy | Non-IID Performance | Complexity | Convergence Speed | Server Compute |
|----------|-------------------|------------|-------------------|----------------|
| FedAvg   | Low               | Low        | Medium            | Low            |
| FedProx  | High              | Medium     | Medium            | Low            |
| FedNova  | Medium            | Low        | Medium            | Low            |
| FedOpt   | Medium            | Medium     | High              | Medium         |
| Scaffold | Very High         | High       | High              | Low            |
| FedAvgM  | Low               | Low        | Medium-High       | Low            |

## Recommended Strategy Selection

### For This Project (Heterogeneous Hymenoptera Data):

1. **Start with FedAvg**: Establish baseline performance
2. **Try FedProx**: If convergence is slow or accuracy is low (likely with non-IID data)
3. **Consider FedNova**: If clients train for different numbers of epochs
4. **Advanced: Scaffold**: If highly heterogeneous data causes significant issues

## Implementation Notes

- All strategies can be implemented using Flower framework
- Custom strategies can be created by subclassing `flwr.server.strategy.Strategy`
- Override `aggregate_fit()` method for custom aggregation logic
- Monitor convergence metrics to select optimal strategy

## Loss Functions

The loss function used during local training is separate from aggregation:

- **Cross-Entropy Loss**: Standard for classification tasks
- **Focal Loss**: Handles class imbalance (if needed)
- **Label Smoothing**: Regularization technique

## Communication Protocol

Each federated round follows this protocol:

1. **Server Selection**: Server selects participating clients
2. **Model Broadcast**: Server sends global model to selected clients
3. **Local Training**: Each client trains on local data (using chosen loss function)
4. **Update Upload**: Clients send weight updates to server
5. **Aggregation**: Server aggregates updates using chosen strategy
6. **Model Update**: Server updates global model
7. **Repeat**: Process repeats for specified number of rounds

## References

- McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. AISTATS.
- Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. MLSys.
- Wang, J., Liu, Q., Liang, H., Joshi, G., & Poor, H. V. (2020). Tackling the objective inconsistency problem in heterogeneous federated optimization. NeurIPS.
- Reddi, S., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konečný, J., ... & McMahan, B. (2021). Adaptive federated optimization. ICLR.
- Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S., Stich, S., & Suresh, A. T. (2020). SCAFFOLD: Stochastic controlled averaging for federated learning. ICML.

## Project Structure

```
fed-learning-prototype/
├── README.md                 # This file
├── server.py                 # Federated learning server (DGX Spark)
├── client.py                 # Federated learning client (Orin Nano)
├── model.py                  # CNN model architecture (ResNet18)
├── data_utils.py             # Data loading and heterogeneous partitioning
├── strategies.py             # Custom aggregation strategies
├── requirements.txt          # Python dependencies
└── config.yaml               # Configuration file
```

## Setup

(To be added)

## Usage

(To be added)

## Results

(To be added)
