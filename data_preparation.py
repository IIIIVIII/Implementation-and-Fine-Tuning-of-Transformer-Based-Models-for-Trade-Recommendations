import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('xnas-itch-20230703.tbbo.csv')

# Display the first few rows of the dataset
print(data.head())

# Display basic information about the dataset
print(data.info())

# Handle missing values
data = data.dropna()

# Feature engineering: create additional features if necessary
# For example, moving averages or price volatility
data['Moving_Avg'] = data['price'].rolling(window=5).mean()
data['Volatility'] = data['price'].rolling(window=5).std()

# Drop rows with NaN values created by rolling window calculations
data = data.dropna()

# Create a 'label' column for buy/sell/hold signals
# This is just an example; you can modify the logic as needed
data['label'] = np.where(data['price'].shift(-1) > data['price'], 'buy',
                         np.where(data['price'].shift(-1) < data['price'], 'sell', 'hold'))

# Convert 'label' to numeric values for machine learning
label_mapping = {'buy': 1, 'sell': -1, 'hold': 0}
data['label'] = data['label'].map(label_mapping)

# Normalize features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['price', 'Moving_Avg', 'Volatility']])

# Prepare the dataset for training
X = scaled_data
y = data['label'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
