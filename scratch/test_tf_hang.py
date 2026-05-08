import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import tensorflow as tf

print("TF imported")
# tf.config.set_visible_devices([], 'GPU')
# try: tf.config.set_visible_devices([], 'MPS')
# except: pass

X = np.random.rand(25, 4, 10).astype(np.float32)
y = np.random.rand(25, 1).astype(np.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4, 10)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

print("Starting fit...")
model.fit(X, y, epochs=5, batch_size=16, validation_split=0.2, verbose=1)
print("Fit done.")
