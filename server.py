"""
Flower ServerApp - Following quickstart-pytorch template pattern
"""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from model import get_model
from task import test_fn
from data_utils import load_datasets

# Create ServerApp
app = ServerApp()


def load_centralized_dataset():
    """Load validation set and return dataloader."""
    from torch.utils.data import DataLoader
    from torchvision import transforms
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load validation dataset
    image_datasets, _ = load_datasets(data_dir='./hymenoptera_data')
    val_dataset = image_datasets['val']
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return val_loader


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config.get("fraction-evaluate", 1.0)
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config.get("lr", 0.001)

    # Load global model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global_model = get_model(num_classes=2, pretrained=False, freeze_backbone=False)
    
    # Try to load pretrained model
    try:
        global_model.load_state_dict(torch.load("./pretrained_model.pt", map_location=device))
    except FileNotFoundError:
        # Fallback to ImageNet pretrained weights
        global_model = get_model(num_classes=2, pretrained=True, freeze_backbone=False)
    
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = get_model(num_classes=2, pretrained=False, freeze_backbone=False)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire validation set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the validation set
    test_loss, test_acc = test_fn(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": float(test_acc), "loss": float(test_loss)})
