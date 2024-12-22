---
title: "A Simple Pytorch Trainpipeline"
date: 2024-06-30T01:52:00+08:00
lastmod: 2024-06-30T02:44:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
    - pytorch
    - trainpipeline
categories:
    - deeplearning
tags:
    - python
    - pytorch
description: How to build a simple Pytorch trainpipeline.
summary: How to build a simple Pytorch trainpipeline.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

## 1. Introduction

In general, you will need these things to train a model:

- A Model
- A Dataset
- A Dataloader
- A Loss Function (Criterion)
- An Optimizer


## 2. Model

We will build a simple model for demonstration. The model takes a tensor of shape `(batch_size, 10)` as input and outputs a tensor of shape `(batch_size, 2)`.

```python {linenos=true}
# @file simple_model.py
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

if __name__ == "__main__":
    model = SimpleModel()
    x = torch.randn(4, 10)  # Shape: (4, 10)
    y = model(x)
    print(y.shape)  # Shape: (4, 2)
```

You can run the script to check how the model works:

```bash {linenos=true}
python simple_model.py
```

## 3. Dataset

We will build a simple dataset for demonstration. The dataset generates random data and labels.

```python {linenos=true}
# @file simple_dataset.py
import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(10)  # Shape: (10,); Element type: float32
        y = torch.randint(0, 2, (1,))  # Shape: (1,); Element type: int64
        return x, y

if __name__ == "__main__":
    dataset = SimpleDataset(4)
    x, y = dataset[0]
    print(x.shape, y.shape)  # Shape: (10,), (1,)
```

You can run the script to check how the dataset works:

```bash {linenos=true}
python simple_dataset.py
```

## 4. Dataloader

As long as the dataset is built, creating a dataloader is quite easy. 

A dataloader will provide `batch_size` samples in each iteration. For example:

```python {linenos=true}
# @file temp.py
from torch.utils.data import DataLoader
from simple_dataset import SimpleDataset

dataset = SimpleDataset(100)
# Get a sample, shape: (10,), (1,)
sample_x, sample_y = dataset[0] 

# Suppose batch_size is 16, the dataloader will provide 16 samples in each iteration
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
for i, (x, y) in enumerate(dataloader):
    print(x.shape, y.shape)  # Shape: (16, 10), (16, 1)
    break
```

You can run the script to check how the dataloader works:

```bash {linenos=true}
python temp.py
```

## 5. Loss Function

Different tasks require different loss functions. For example, a 2-class classification task can use `nn.CrossEntropyLoss`, while a regression task can use `nn.MSELoss`.

In our case, we will use `nn.CrossEntropyLoss`. 

## 6. Optimizer

We will use `torch.optim.SGD` as the optimizer. `torch.optim.Adam` is also a good choice. This is a hyperparameter that you can tune.

## 7. Trainpipeline

Now we can build the trainpipeline. The trainpipeline will train the model on the dataset.

```python {linenos=true}
# @file trainpipeline.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# This is the model we built
from simple_model import SimpleModel
# This is the dataset we built
from simple_dataset import SimpleDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.01


def train():
    # Create a model and move it to DEVICE
    model = SimpleModel().to(DEVICE)

    # Create train dataset and dataloader
    train_dataset = SimpleDataset(1000)
    val_dataset = SimpleDataset(100)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # Create a loss function and an optimizer; The optimizer will update the model's parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()  # Set the model to training mode
        for i, (x, y) in enumerate(train_dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y.squeeze())
            loss.backward()
            optimizer.step()

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            total_loss = 0
            total_correct = 0
            total_samples = 0
            for i, (x, y) in enumerate(val_dataloader):
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pred = model(x)
                loss = criterion(y_pred, y.squeeze())
                total_loss += loss.item()
                total_correct += (y_pred.argmax(dim=1) == y.squeeze()).sum().item()
                total_samples += y.size(0)

            print(f"Epoch: {epoch}, Loss: {total_loss / total_samples}, Accuracy: {total_correct / total_samples}")

if __name__ == "__main__":
    train()
```

You can run the script to check how the trainpipeline works:

```bash {linenos=true}
python trainpipeline.py
```