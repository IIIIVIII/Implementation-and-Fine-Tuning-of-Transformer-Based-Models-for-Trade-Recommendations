import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD
def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Load the dataset
data = pd.read_csv('xnas-itch-20230703.tbbo.csv')  # Use relative path

# Display the first few rows of the dataset
print(data.head())

# Display basic information about the dataset
print(data.info())

# Handle missing values
data = data.dropna()

# Feature engineering
data['Moving_Avg'] = data['price'].rolling(window=5).mean()
data['Volatility'] = data['price'].rolling(window=5).std()
data['RSI'] = calculate_rsi(data['price'])
data['MACD'], data['Signal_Line'] = calculate_macd(data['price'])
data = data.dropna()

# Create a 'label' column for buy/sell/hold signals
data['label'] = np.where(data['price'].shift(-1) > data['price'], 'buy',
                         np.where(data['price'].shift(-1) < data['price'], 'sell', 'hold'))

label_mapping = {'buy': 1, 'sell': 0, 'hold': 2}
data['label'] = data['label'].map(label_mapping)

# Normalize features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['price', 'Moving_Avg', 'Volatility', 'RSI', 'MACD', 'Signal_Line']])

# Prepare the dataset for training
X = scaled_data
y = data['label'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Parameters
input_dim = 6  # Adjusted for new features
hidden_dim = 64
num_layers = 2
num_classes = 3
num_epochs = 50
batch_size = 128
learning_rate = 0.001

# Adjust the data to match the new input dimension
X_train_padded = np.expand_dims(X_train, axis=1)
X_test_padded = np.expand_dims(X_test, axis=1)

# Convert to PyTorch tensors
X_train_padded = torch.tensor(X_train_padded, dtype=torch.float32)
X_test_padded = torch.tensor(X_test_padded, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Initialize the model, loss function and optimizer
model = LSTMModel(input_dim, hidden_dim, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
best_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i in range(0, len(X_train_padded), batch_size):
        X_batch = X_train_padded[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / (len(X_train_padded) / batch_size)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break

# Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_padded)
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f'Accuracy: {accuracy:.4f}')
