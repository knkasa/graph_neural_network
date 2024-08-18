import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from spektral.layers import GCNConv
from spektral.data import Dataset, Graph
from spektral.data.loaders import SingleLoader
from spektral.transforms import AdjToSpTensor

# Create a simple dataset for fraud detection
class FraudDetectionDataset(Dataset):
    def read(self):
        # Node features (e.g., transaction amount, transaction type)
        x = np.array([
            [1000, 0],  # Transaction 0
            [1500, 1],  # Transaction 1
            [200, 0],   # Transaction 2
            [1200, 1],  # Transaction 3
            [250, 0],   # Transaction 4
            [3000, 1],  # Transaction 5
        ], dtype=np.float32)

        # Adjacency matrix (connections based on shared account numbers, etc.)
        a = np.array([
            [0, 1, 0, 0, 0, 0],  # Transaction 0
            [1, 0, 1, 0, 0, 0],  # Transaction 1
            [0, 1, 0, 1, 0, 0],  # Transaction 2
            [0, 0, 1, 0, 1, 1],  # Transaction 3
            [0, 0, 0, 1, 0, 0],  # Transaction 4
            [0, 0, 0, 1, 0, 0],  # Transaction 5
        ], dtype=np.float32)

        # Labels (fraudulent: 1, not fraudulent: 0)
        y = np.array([
            [1],  # Label for Transaction 0
            [0],  # Label for Transaction 1
            [0],  # Label for Transaction 2
            [1],  # Label for Transaction 3
            [0],  # Label for Transaction 4
            [1],  # Label for Transaction 5
        ], dtype=np.float32)

        return [Graph(x=x, a=a, y=y)]

# Create the dataset
dataset = FraudDetectionDataset(transforms=AdjToSpTensor())
loader = SingleLoader(dataset)

# Define the GNN model
class GNNModel(Model):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCNConv(16, activation='relu')
        self.gcn2 = GCNConv(1, activation='sigmoid')

    def call(self, inputs):
        x, a = inputs
        x = self.gcn1([x, a])
        x = self.gcn2([x, a])
        return x

# Create the model
model = GNNModel()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
for batch in loader:
    model.fit(batch[0], batch[1], epochs=200, batch_size=1, verbose=1)

# Predict using the model
predictions = model.predict(loader.load())
print(predictions)
