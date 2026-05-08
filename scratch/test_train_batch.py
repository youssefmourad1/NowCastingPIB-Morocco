import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

X = np.random.rand(25, 4, 10).astype(np.float32)
y = np.random.rand(25, 1).astype(np.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4, 10)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse")

for epoch in range(5):
    loss = model.train_on_batch(X, y)
    print(f"Epoch {epoch} loss: {loss}")

print("SUCCESS")
