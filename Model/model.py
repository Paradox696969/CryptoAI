# imports + adding dlls ------------------------------------------------------->
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.add_dll_directory("C:/Users/joyje/Downloads/cuda/bin")
import tensorflow as tf
import keras_tuner as kt
import pandas as pd
import numpy as np
import pydot

tf.keras.backend.clear_session()

# loads training data and separates into training and labels ------------------>
df = pd.read_csv("./CryptoModelData/CryptoDaily.csv", low_memory=False)

df = df.loc[::-1]

df_features = df.copy().drop(["Close", "Low", "High", "Volume BTC", "Symbol", "CryptoCurrency", "Platform"], axis=1)

df_labels = df_features.pop("Open")

inputs = {}

for name, column in df_features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float64

    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

numeric_inputs = {name:input for name, input in inputs.items()
if input.dtype==tf.float64}



x = tf.keras.layers.Concatenate()(list(numeric_inputs.values()))
norm = tf.keras.layers.Normalization()
norm.adapt(np.array(df[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

preprocessed_inputs = [all_numeric_inputs]

for name, input in inputs.items():
    if input.dtype == tf.float64:
        continue

    lookup = tf.keras.layers.StringLookup(vocabulary=np.unique(df_features[name]))
    one_hot = tf.keras.layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

    print(lookup.vocabulary_size())

    x = lookup(input)
    x = one_hot(x)

    preprocessed_inputs.append(x)

preprocessed_inputs_cat = tf.keras.layers.Concatenate()(preprocessed_inputs)
df_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

df_features_dict = {name:np.array(value) for name, value in df_features.items()}
features_dict = {name:values[:1] for name, values in df_features_dict.items()}
df_preprocessing(features_dict)

# ModelBuilder function:creates a model compiled with an optimizer and certain -
# types of layers ------------------------------------------------------------->
def model_builder(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=1024, output_dim=128))
    # hp_units0 = hp.Int("units", min_value=64, max_value=1024, step=32)
    # model.add(tf.keras.layers.Dense(units=hp_units0))

    hp_units = hp.Int("units", min_value=64, max_value=1024, step=16)
    model.add(tf.keras.layers.LSTM(units=hp_units))
    # model.add(tf.keras.layers.Dense(units=hp_units1, activation="relu"))
    model.add(tf.keras.layers.Dense(1))

    preprocessed_inputs = df_preprocessing(inputs)
    result = model(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=['accuracy']
    )

    return model

# Tuner Definition:Bayesian Optimization Tuner is effective as Crypto is -
# stochastic in nature -------------------------------------------------------->
tuner = kt.BayesianOptimization(
    model_builder,
    objective="val_accuracy",
    max_trials=50
)

# callback for stopping early ------------------------------------------------->
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

# callback for saving weights ------------------------------------------------->
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./CryptoModelData/SavedModels/", save_weights_only=True, verbose=1)

tuner.search(df_features_dict, df_labels, epochs=15, validation_split=0.2, callbacks=[stop_early], batch_size=32768)



# Tuner searches and finds the best Hyper Parameters -------------------------->
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

