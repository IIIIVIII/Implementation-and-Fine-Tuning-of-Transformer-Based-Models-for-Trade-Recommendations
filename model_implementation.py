import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('xnas-itch-20230703.tbbo.csv')

# Handle missing values
data = data.dropna()

# Feature engineering
data['Moving_Avg'] = data['price'].rolling(window=5).mean()
data['Volatility'] = data['price'].rolling(window=5).std()
data = data.dropna()

# Create a 'label' column for buy/sell/hold signals
data['label'] = np.where(data['price'].shift(-1) > data['price'], 'buy',
                         np.where(data['price'].shift(-1) < data['price'], 'sell', 'hold'))

label_mapping = {'buy': 1, 'sell': 0, 'hold': 2}
data['label'] = data['label'].map(label_mapping)

# Normalize features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['price', 'Moving_Avg', 'Volatility']])

# Prepare the dataset for training
X = scaled_data
y = data['label'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, src):
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)  # Pooling
        output = self.fc(output)
        return output

# Set input_dim to a larger value that is divisible by num_heads
input_dim = 8  # New input dimension
num_heads = 2
num_layers = 4  # Increase number of layers
num_classes = 3  # Buy, Sell, Hold

# Adjust the data to match the new input dimension
X_train_padded = np.pad(X_train, ((0, 0), (0, input_dim - X_train.shape[1])), 'constant')
X_test_padded = np.pad(X_test, ((0, 0), (0, input_dim - X_test.shape[1])), 'constant')

# Add a sequence dimension
X_train_padded = np.expand_dims(X_train_padded, axis=1)
X_test_padded = np.expand_dims(X_test_padded, axis=1)

model = TransformerModel(input_dim, num_heads, num_layers, num_classes)

# Training the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Reduce learning rate

num_epochs = 50  # Increase number of epochs
batch_size = 128  # Increase batch size

# Early stopping variables
best_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i in range(0, len(X_train_padded), batch_size):
        X_batch = torch.tensor(X_train_padded[i:i+batch_size], dtype=torch.float32)
        y_batch = torch.tensor(y_train[i:i+batch_size], dtype=torch.long)

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
    test_outputs = model(torch.tensor(X_test_padded, dtype=torch.float32))
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = (predicted == torch.tensor(y_test)).sum().item() / len(y_test)
    print(f'Accuracy: {accuracy:.4f}')
