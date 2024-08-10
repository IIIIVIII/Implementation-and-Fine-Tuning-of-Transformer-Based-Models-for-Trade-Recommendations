import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define BiLSTM model
class BiLSTMModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(BiLSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_dim * 2, num_classes)  # Bidirectional

    def forward(self, x):
        x, _ = self.lstm(x.unsqueeze(1))  # Add a dimension to match LSTM input
        x = self.fc(x[:, -1, :])  # Take output from the last time step
        return x

# Load the model
input_dim = 8  # Number of features
hidden_dim = 64
num_layers = 2
num_classes = 3  # Assuming 3 classes: Buy, Sell, Hold

model = BiLSTMModel(input_dim, hidden_dim, num_layers, num_classes).to(device)

# Load the trained BiLSTM model weights
model.load_state_dict(torch.load('best_hybrid_model.pth'))
model.eval()

# Load and preprocess the data
data = pd.read_csv('/Users/mingfanxie/Desktop/blockhouse/xnas-itch-20230703.tbbo.csv')

# Feature engineering
features = ['price', 'volume', 'symbol']  # Replace with actual features
X = data[features].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_test_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

# Get predictions
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

# Assuming 'label' column exists in your data for the true labels
y_test = data['label'].values

# Evaluation
print("Classification Report:\n", classification_report(y_test, predicted.cpu().numpy()))
print("Confusion Matrix:\n", confusion_matrix(y_test, predicted.cpu().numpy()))
