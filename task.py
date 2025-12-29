"""
Training and test functions for the model
Following Flower tutorial format
"""

import torch
import torch.nn as nn


def train_fn(model, train_loader, device, epochs, lr):
    """Train the model on the training set"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Calculate training metrics
    model.eval()
    train_loss = 0.0
    train_corrects = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)
    
    train_loss = train_loss / total
    train_acc = train_corrects.double() / total
    
    return train_loss, train_acc


def test_fn(model, test_loader, device):
    """Test the model on the test set"""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    test_loss = 0.0
    test_corrects = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)
    
    test_loss = test_loss / total
    test_acc = test_corrects.double() / total
    
    return test_loss, test_acc

