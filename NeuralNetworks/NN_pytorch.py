import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim, activation_function):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_units))
        
        if activation_function == 'tanh':
            nn.init.xavier_uniform_(layers[0].weight)
        elif activation_function == 'relu':
            nn.init.kaiming_uniform_(layers[0].weight, nonlinearity='relu')
        
        self.activation = nn.Tanh() if activation_function == 'tanh' else nn.ReLU()
        
        for _ in range(1, depth - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            if activation_function == 'tanh':
                nn.init.xavier_uniform_(layers[-1].weight)
            elif activation_function == 'relu':
                nn.init.kaiming_uniform_(layers[-1].weight, nonlinearity='relu')
            layers.append(self.activation)
        
        layers.append(nn.Linear(hidden_units, output_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_model(X_train, y_train, model, criterion, optimizer, epochs=100):
    train_errors = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        
        loss.backward()
        optimizer.step()
        
        train_errors.append(loss.item())
        
    return train_errors


def calculate_error(model, X, y, criterion):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        error = criterion(outputs.squeeze(), y).item()
    return error

train_data = pd.read_csv("NeuralNetworks/data/bank-note/train.csv", header=None)
test_data = pd.read_csv("NeuralNetworks/data/bank-note/test.csv", header=None)

X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

widths = [5, 10, 25, 50, 100]
depths = [3, 5, 9]
learning_rate = 1e-3
epochs = 100

results = []

for activation in ['tanh', 'relu']:
    for hidden_units in widths:
        for depth in depths:
            print(f"\nTraining with activation: {activation}, hidden units: {hidden_units}, depth: {depth}")
            
            model = MLP(input_dim=X_train.shape[1], hidden_units=hidden_units, 
                        output_dim=1, activation_function=activation)
            
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            train_errors = train_model(X_train_tensor, y_train_tensor, model, criterion, optimizer, epochs)
            
            train_error = calculate_error(model, X_train_tensor, y_train_tensor, criterion)
            val_error = calculate_error(model, X_val_tensor, y_val_tensor, criterion)
            
            results.append({
                'activation': activation,
                'hidden_units': hidden_units,
                'depth': depth,
                'train_error': train_error,
                'val_error': val_error
            })
            
            print(f"Training Error: {train_error:.4f} | Validation Error: {val_error:.4f}")

print("\nSummary of Results:")
for result in results:
    print(f"Activation: {result['activation']}, Hidden Units: {result['hidden_units']}, Depth: {result['depth']}, "
          f"Training Error: {result['train_error']:.4f}, Validation Error: {result['val_error']:.4f}")
