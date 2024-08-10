import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Functions to calculate technical indicators
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(data, window=20, num_of_std=2):
    rolling_mean = data.rolling(window).mean()
    rolling_std = data.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return upper_band, lower_band

# Load dataset
data = pd.read_csv('/Users/mingfanxie/Desktop/blockhouse/xnas-itch-20230703.tbbo.csv')

# Handle missing values
data = data.dropna()

# Generate labels
data['label'] = 0
data.loc[data['price'].diff() > 0, 'label'] = 1
data.loc[data['price'].diff() < 0, 'label'] = 2

# Feature engineering
data['Moving_Avg'] = data['price'].rolling(window=5).mean()
data['Volatility'] = data['price'].rolling(window=5).std()
data['RSI'] = calculate_rsi(data['price'])
data['MACD'], data['Signal_Line'] = calculate_macd(data['price'])
data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data['price'])

# Handle infinite and missing values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(0, inplace=True)

# Features and labels
features = ['price', 'Moving_Avg', 'Volatility', 'RSI', 'MACD', 'Signal_Line', 'Upper_Band', 'Lower_Band']
X = data[features].values
y = data['label'].values

# Data standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Check for zero standard deviation columns
for i in range(X_scaled.shape[1]):
    if np.std(X_scaled[:, i]) == 0:
        print(f"Feature {features[i]} has zero standard deviation.")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert data to Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the model
class HybridModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, num_classes):
        super(HybridModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim * 2, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        x, _ = self.lstm(x.unsqueeze(1))  # Add an additional dimension to match LSTM input
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])  # Take the output of the last time step
        return x

# Initialize model, loss function, and optimizer
input_dim = X_train.shape[1]
hidden_dim = 64
num_layers = 2
num_heads = 4
num_classes = len(np.unique(y))
model = HybridModel(input_dim, hidden_dim, num_layers, num_heads, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

# Train the model
num_epochs = 100
best_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    scheduler.step(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        # Save model weights
        torch.save(model.state_dict(), 'best_hybrid_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break

# Load best model weights
model.load_state_dict(torch.load('best_hybrid_model.pth'))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    accuracy = correct / total
print(f'Accuracy: {accuracy:.4f}')

# Model evaluation
from sklearn.metrics import classification_report, confusion_matrix

y_test_np = y_test_tensor.cpu().numpy()
y_pred_np = []
model.eval()
with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs[0].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred_np.extend(predicted.cpu().numpy())

# Print classification report and confusion matrix
print("Classification Report:\n", classification_report(y_test_np, y_pred_np))
print("Confusion Matrix:\n", confusion_matrix(y_test_np, y_pred_np))
