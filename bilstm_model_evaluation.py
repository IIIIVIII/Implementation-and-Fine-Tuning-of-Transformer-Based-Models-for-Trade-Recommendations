import torch
import torch.nn as nn
import pandas as pd
import numpy as np  # 确保导入 numpy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

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

# Load the dataset
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

# Convert data to Tensors
X_test_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y, dtype=torch.long)

# Initialize model
input_dim = X_test_tensor.shape[1]
hidden_dim = 64
num_layers = 2
num_heads = 4
num_classes = len(np.unique(y))
model = HybridModel(input_dim, hidden_dim, num_layers, num_heads, num_classes).to(device)

# Load best model weights, ignoring mismatched keys
checkpoint = torch.load('best_hybrid_model.pth')
model.load_state_dict(checkpoint, strict=False)

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor.to(device))
    _, predicted = torch.max(outputs, 1)

# Convert to numpy arrays for evaluation
y_test_np = y_test_tensor.cpu().numpy()
y_pred_np = predicted.cpu().numpy()

# Print classification report and confusion matrix
print("Classification Report:\n", classification_report(y_test_np, y_pred_np))
print("Confusion Matrix:\n", confusion_matrix(y_test_np, y_pred_np))
