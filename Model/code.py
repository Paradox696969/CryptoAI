import tensorflow as tf
import keras_tuner as kt
import pandas as pd

dfd = pd.read_csv("./CryptoModelData/CryptoDaily.csv")
df_train = dfd.copy()

df = dfd.pop("Open")
df = df.pop("Close")
df = df.pop("High")
df = df.pop("Low")

df_labels = df.copy()

def model_builder(hp):
    model = tf.keras.Sequential()
    model.add(layers.Embedding(input_dim=1000, output_dim=64))

    hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
    model.add(layers.SimpleRNN(units=hp_units, activation="softmax"))
    model.add(layers.Dense(10))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy']
    )

    return model

tuner = kt.BayesianOptimization(
    model_builder,
    objective='val_accuracy',
    max_trials=10
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")
