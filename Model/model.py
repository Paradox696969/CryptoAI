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
import matplotlib.pyplot as plt
import matplotlib

# All code between sets of :: were taken from Venelin Valkov ------------------>

tf.keras.backend.clear_session()

# define matplotlib font parameters ------------------------------------------->
font = {'family' : 'normal',
    'weight' : 'bold',
    'size'   : 24}

matplotlib.rc('font', **font)

# callback for stopping early ------------------------------------------------->
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# callback for saving weights ------------------------------------------------->
# ::

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# ::

# iterating through all of the files in the data folder ----------------------->
for filename in os.listdir("./CryptoModelData/Minutely/"):
    file_split = filename.split("_")

    # loads data files to a pandas dataframe ---------------------------------->
    df = pd.read_csv(f"./CryptoModelData/Minutely/{filename}", low_memory=False)

    # removing all unnecessary columns (LHC values may be used in the future) ->
    df = df.copy().drop(["Close", "Low", "High", "Volume", "Symbol", "CryptoCurrency", "Platform", "Date"], axis=1)
    print(df)

    # finding the ratio of lines needed for an 8:2 split in train:test data --->
    train_size = int(len(df) * 0.8)
    test_size = len(df) - train_size

    # splitting data into training and testing data
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

    # ::

    # defining what columns are meant to be for training (features) ----------->
    feature_cols = ["Unix Timestamp"]

    # scaling the data to make it even easier to use by the model ------------->
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

    # creates datasets of sequences which better prep our data ---------------->
    train_features, train_labels = create_dataset(train, train.Open, time_steps)
    test_features, test_labels = create_dataset(test, test.Open, time_steps)

    # ::

    # ModelBuilder function:creates a model compiled with optimizer and certain -
    # types of layers --------------------------------------------------------->
    def model_builder(hp):
        model = tf.keras.Sequential()

        hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=hp_units, input_shape=(train_features.shape[1], train_features.shape[2]))))
        model.add(tf.keras.layers.Dense(1))

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

        return model

    # Tuner Definition:Bayesian Optimization Tuner is effective as Crypto is -
    # stochastic in nature ---------------------------------------------------->
    tuner = kt.BayesianOptimization(
        model_builder,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=15
    )

    # Tuner searches and finds the best Hyper Parameters ---------------------->
    tuner.search(train_features, train_labels, epochs=10, validation_split=0.2, callbacks=[stop_early], batch_size=968, shuffle=False)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # print the optimal neurons in the LSTM layer as well as LR --------------->
    print(f"Optimal Units: {best_hps.get('units')}\nOptimal LR: {best_hps.get('learning_rate')}\n")

    # train the model again to find the best epoch out of 35 epochs ----------->
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_features, train_labels, epochs=35, validation_split=0.2, batch_size=968, shuffle=False)

    # find and print the best epochs ------------------------------------------>
    val_loss_per_epoch = history.history['val_loss']
    best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1

    print(f"Best epoch: {best_epoch}")

    # callback for stopping early --------------------------------------------->
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f"./CryptoModelData/SavedModels/{file_split[0]}/{file_split[1][:-4]}/Model_Weights", save_weights_only=True, verbose=1)

    # Buildand train the model again ------------------------------------------>
    hypermodel = tuner.hypermodel.build(best_hps)

    # evaluate the model and save its predictions ----------------------------->
    hypermodel.fit(train_features, train_labels, epochs=best_epoch, validation_split=0.2, callbacks=[cp_callback], batch_size=968, shuffle=False)

    eval_result, result_for_plot = hypermodel.evaluate(test_features, test_labels), hypermodel.predict(test_features)
    print("[test loss, test accuracy]:", eval_result)

    # create a dataframe to save current model's results to a file ------------>
    data = {
    'Platform': [file_split[0]],
    'Symbol': [file_split[1][:-4]],
    'Units': [best_hps.get('units')],
    'LR': [best_hps.get('learning_rate')],
    'Test_Loss': [eval_result[0]],
    'Test_Acc': [eval_result[1]],
    'Val_Loss': [min(history.history['val_loss'])],
    'Train_f_shape_0': [train_features.shape[1]],
    'Train_f_shape_1': [train_features.shape[2]]
    }
    rf = pd.DataFrame(data)
    print(rf)

    # save current model's results to a file ---------------------------------->
    rf.to_csv('./CryptoModelData/Model_results/Results_Final.csv', mode='a', index=False, header=False)

    # clear plot -------------------------------------------------------------->
    plt.clf()

    # save current model's losses --------------------------------------------->
    os.mkdir(f"./CryptoModelData/SavedModels/{file_split[0]}/{file_split[1][:-4]}/Plots/")
    plt.figure(figsize=(21, 13), dpi=200)
    plt.plot(history.history['loss'], label="Train")
    plt.plot(history.history['val_loss'], label="Validation")
    plt.legend()
    with open(f"./CryptoModelData/SavedModels/{file_split[0]}/{file_split[1][:-4]}/Plots/lossGraph.png", "wb") as f:
        plt.savefig(f)

    # clear plot -------------------------------------------------------------->
    plt.clf()

    # save current model's test values vs predictions to graph
    plt.figure(figsize=(55, 21), dpi=200)
    plt.plot(value_transformer.inverse_transform(test_labels.reshape(1, -1)).flatten(), label="True Value", marker=".")
    plt.plot(value_transformer.inverse_transform(result_for_plot.reshape(1, -1)).flatten(), label="Prediction", marker=".")
    plt.legend()
    with open(f"./CryptoModelData/SavedModels/{file_split[0]}/{file_split[1][:-4]}/Plots/Pred-True.png", "wb") as f:
        plt.savefig(f)

    # delete the directory where tuner trials are stored for next model ------->
    shutil.rmtree("./untitled_project", ignore_errors=True)
