import numpy as np
import concurrent.futures

def train_in_process():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    import tensorflow as tf
    X = np.random.rand(25, 4, 10).astype(np.float32)
    y = np.random.rand(25, 1).astype(np.float32)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4, 10)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=1, batch_size=16, verbose=0)
    return "SUCCESS"

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(train_in_process)
        print("Result:", future.result())
