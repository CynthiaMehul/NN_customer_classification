# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model


## DESIGN STEPS

### STEP 1
Load and preprocess the dataset (handle missing values, encode categorical features, scale numeric data).

### STEP 2
Split the dataset into training and testing sets, convert to tensors, and create DataLoader objects.

### STEP 3
Build the neural network model, train it with CrossEntropyLoss and Adam optimizer, then evaluate with confusion matrix and classification report.

## PROGRAM
### Name: Cynthia Mehul J
### Register Number: 212223240020

```python
class NeuralNetwork(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc1 = torch.nn.Linear(size, 16)
        self.fc2 = torch.nn.Linear(16, 8)
        self.fc3 = torch.nn.Linear(8, 4)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
```python
# Initialize the Model, Loss Function, and Optimizer
cynthia_brain=NeuralNetwork(x_train.shape[1])
loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(cynthia_brain.parameters(),lr=0.001)
```
```python
def train_model(model, train_loader, loss_fn, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in train_loader:
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
```

## Dataset Information

<img width="1451" height="638" alt="image" src="https://github.com/user-attachments/assets/cb844952-5aa5-440f-8864-09197b6e52c3" />

## OUTPUT

### Confusion Matrix

<img width="704" height="585" alt="image" src="https://github.com/user-attachments/assets/f3dafd8d-ece4-4ccd-a204-216d716b8627" />

### Classification Report

<img width="551" height="430" alt="image" src="https://github.com/user-attachments/assets/9b0adb78-f332-41b5-b873-a131324107c0" />

### New Sample Data Prediction

<img width="362" height="102" alt="image" src="https://github.com/user-attachments/assets/c217a669-3ed0-4900-92b0-86f283b6693c" />

## RESULT
The neural network model was successfully built and trained to handle classification tasks.
