# https://dss.i.u-tokyo.ac.jp/blog/gnn%E3%81%A7%E3%81%A7%E3%81%8D%E3%82%8B%E3%81%93%E3%81%A8%EF%BC%9A%E3%82%B0%E3%83%A9%E3%83%95%E3%83%8B%E3%83%A5%E3%83%BC%E3%83%A9%E3%83%AB%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF/
# Pytorch example. Convolutional graph network.  https://pub.towardsai.net/graph-neural-networks-unlocking-the-power-of-relationships-in-predictions-04dd74daa742

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from spektral.layers import GCNConv
from spektral.data import Dataset, Graph
from spektral.data.loaders import SingleLoader
from spektral.transforms import AdjToSpTensor
import scipy.sparse as sp
import pdb

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
        
        # Convert adjacency matrix to sparse format before creating the graph
        a_sparse = sp.csr_matrix(a)
        
        # Labels (fraudulent: 1, not fraudulent: 0)
        y = np.array([
            [1],  # Label for Transaction 0
            [0],  # Label for Transaction 1
            [0],  # Label for Transaction 2
            [1],  # Label for Transaction 3
            [0],  # Label for Transaction 4
            [1],  # Label for Transaction 5
        ], dtype=np.float32)
        
        return [Graph(x=x, a=a_sparse, y=y)]

# Create the dataset
dataset = FraudDetectionDataset()
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

# Create input layers
X_in = Input(shape=(2,))  # 2 features per node
A_in = Input(shape=(None,), sparse=True)

# Build the model
x = X_in
a = A_in
model = GNNModel()
output = model([x, a])
model = Model(inputs=[X_in, A_in], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

# Train the model
history = model.fit(
    loader.load(),
    steps_per_epoch=loader.steps_per_epoch,
    epochs=20,
    verbose=1
    )

predictions = model.predict(loader.load(),steps=loader.steps_per_epoch) # model.predict(loader.load(), steps=1) will also work. steps means the number of batchs to split the data during prediction.

# Print predictions
print("\nPredictions:")
for i, pred in enumerate(predictions):
    print(f"Transaction {i}: {'Fraudulent' if pred[0] > 0.5 else 'Not Fraudulent'} (Score: {pred[0]:.3f})")
