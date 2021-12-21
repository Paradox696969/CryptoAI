# imports + adding dlls ------------------------------------------------------->
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.add_dll_directory("C:/Users/joyje/Downloads/cuda/bin")
import tensorflow as tf
import keras_tuner as kt
import pandas as pd
import numpy as np
import pydot
from sklearn.preprocessing import RobustScaler
import shutil


tf.keras.backend.clear_session()

# callback for stopping early ------------------------------------------------->
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# callback for saving weights ------------------------------------------------->

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


for filename in os.listdir("./CryptoModelData/Minutely/"):
    file_split = filename.split("_")
    # loads training data and separates into training and labels ------------------>
    df = pd.read_csv(f"./CryptoModelData/Minutely/{filename}", low_memory=False)

    df = df.copy().drop(["Close", "Low", "High", "Volume", "Symbol", "CryptoCurrency", "Platform", "Date"], axis=1)
    print(df)
    train_size = int(len(df) * 0.8)
    test_size = len(df) - train_size

    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

    print(len(train), len(test))

    feature_cols = ["Unix Timestamp"]
    feature_transformer = RobustScaler()
    feature_transformer = feature_transformer.fit(train[feature_cols].to_numpy())
    train.loc[:, feature_cols] = feature_transformer.transform(
        train[feature_cols].to_numpy()
    )
    test.loc[:, feature_cols] = feature_transformer.transform(
        test[feature_cols].to_numpy()
    )
    value_transformer = RobustScaler()
    value_transformer = value_transformer.fit(train[['Open']])
    train['Open'] = value_transformer.transform(train[['Open']])
    test['Open'] = value_transformer.transform(test[['Open']])

    time_steps = 1

    train_features, train_labels = create_dataset(train, train.Open, time_steps)
    test_features, test_labels = create_dataset(test, test.Open, time_steps)
    print(train_features.shape, train_labels.shape)

    # ModelBuilder function:creates a model compiled with an optimizer and certain -
    # types of layers ------------------------------------------------------------->
    def model_builder(hp):
        model = tf.keras.Sequential()

        hp_units = hp.Int("units", min_value=64, max_value=512, step=32)
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=hp_units, input_shape=(train_features.shape[1], train_features.shape[1]))))
        model.add(tf.keras.layers.Dense(1))

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

        return model

    # Tuner Definition:Bayesian Optimization Tuner is effective as Crypto is -
    # stochastic in nature -------------------------------------------------------->
    tuner = kt.BayesianOptimization(
        model_builder,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=7
    )


    tuner.search(train_features, train_labels, epochs=5, validation_split=0.2, callbacks=[stop_early], batch_size=960, shuffle=False)

    # Tuner searches and finds the best Hyper Parameters -------------------------->
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"Optimal Units: {best_hps.get('units')}\nOptimal LR: {best_hps.get('learning_rate')}\n")

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_features, train_labels, epochs=30, validation_split=0.2, batch_size=968, shuffle=False)

    val_loss_per_epoch = history.history['val_loss']
    best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1

    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f"./CryptoModelData/SavedModels/{file_split[0]}/{file_split[1]}/Model_Weights", save_weights_only=True, verbose=1)

    hypermodel.fit(train_features, train_labels, epochs=best_epoch, validation_split=0.2, callbacks=[cp_callback], batch_size=968, shuffle=False)

    eval_result = hypermodel.evaluate(test_features, test_labels)
    print("[test loss, test accuracy]:", eval_result)

    data = {
    'Platform': [file_split[0]],
    'Symbol': [file_split[1]],
    'Units': [best_hps.get('units')],
    'LR': [best_hps.get('learning_rate')],
    'Test_Loss': [eval_result[0]],
    'Test_Acc': [eval_result[1]],
    'Val_Loss': [min(history.history['val_loss'])]
    }
    rf = pd.DataFrame(data)
    print(rf)


    rf.to_csv('./CryptoModelData/Model_results/Results.csv', mode='a', index=False, header=False)

    shutil.rmtree("./untitled_project", ignore_errors=True)
