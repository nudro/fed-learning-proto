"""
Flower ClientApp - Following quickstart-pytorch template pattern
"""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from model import get_model
from task import train_fn, test_fn
from data_utils import get_client_dataloader

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = get_model(num_classes=2, pretrained=False, freeze_backbone=False)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config.get("batch-size", 4))
    
    trainloader, valloader, _ = get_client_dataloader(
        partition_id=partition_id,
        data_dir="./hymenoptera_data",
        num_clients=num_partitions,
        batch_size=batch_size,
        alpha=0.5,
        num_workers=0
    )

    # Call the training function
    train_loss, train_acc = train_fn(
        model,
        trainloader,
        device,
        int(context.run_config.get("local-epochs", 1)),
        float(msg.content["config"].get("lr", 0.001)),
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = get_model(num_classes=2, pretrained=False, freeze_backbone=False)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config.get("batch-size", 4))
    
    trainloader, valloader, _ = get_client_dataloader(
        partition_id=partition_id,
        data_dir="./hymenoptera_data",
        num_clients=num_partitions,
        batch_size=batch_size,
        alpha=0.5,
        num_workers=0
    )

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": float(eval_loss),
        "eval_acc": float(eval_acc),
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
