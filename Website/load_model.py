# imports + adding dlls ------------------------------------------------------->
import os
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
import requests
import pandas as pd
import numpy as np
import datetime

tf.config.run_functions_eagerly(True)

def create_dataset(X, time_steps=1):
    Xs = []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
    return np.array(Xs)

def ModelPredict(file, predict):
    predictList = []

    try:
        predict[0], predict[1] = float(predict[0]), float(predict[1])
        for x in range(int(predict[0]), int(predict[1]), 60000):
            predictList.append(x)
    except:
        predict[0], predict[1] = predict[0].split(":"), predict[1].split(":")
        predict[0], predict[1] = datetime.timestamp(datetime.datetime(predict[0][0], predict[0][1], predict[0][2], predict[0][3], predict[0][4])), datetime.timestamp(datetime.datetime(predict[1][0], predict[1][1], predict[1][2], predict[1][3], predict[1][4]))
        for x in range(int(predict[0]), int(predict[1]), 60000):
            predictList.append(x)



    collectedData = pd.DataFrame(requests.get(f"https://cryptoai.paradoxa.repl.co/downloads/Binance_{file[1]}.csv").json())
    length = len(predictList)

    predictDict = {
        "Unix Timestamp": predictList,
        "Open": list(collectedData["Open"].tail(length))
    }

    predictdf  = pd.DataFrame(predictDict)


    traindf = pd.read_csv(f"./CryptoAI/CryptoModelData/Data/Binance_{file[1]}_train.csv")
    train_size = int(len(traindf) * 0.8)
    train = traindf.iloc[0:train_size]

    feature_cols = ["Unix Timestamp"]
    feature_transformer = RobustScaler()
    feature_transformer = feature_transformer.fit(train[feature_cols].to_numpy())
    value_transformer = RobustScaler()
    value_transformer = value_transformer.fit(train[['Open']])

    predictdf.loc[:, feature_cols] = feature_transformer.transform(
        predictdf[feature_cols].to_numpy()
    )

    predictdf['Open'] = value_transformer.transform(predictdf[['Open']])

    time_steps = 1
    print(predictdf)

    predictFeatures = create_dataset(predictdf, time_steps)

    resultsDf = pd.read_csv("./CryptoAI/CryptoModelData/Model_results/Results_Final.csv")
    tempdf = resultsDf.loc[resultsDf["Platform"] == file[0]]
    ModelDf = tempdf.loc[tempdf["Symbol"] == file[1]]
    optimal_values = (int(ModelDf["Units"]), float(ModelDf["LR"]), int(ModelDf["Train_f_shape_0"]), int(ModelDf["Train_f_shape_1"]))


    def model_loader(optima):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=optima[0], input_shape=(optima[2], optima[3]))))
        model.add(tf.keras.layers.Dense(1))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=optima[1]),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
            run_eagerly=True
        )

        return model

    model = model_loader(optimal_values)

    model.load_weights(f"./CryptoAI/CryptoModelData/SavedModels/{file[0]}/{file[1]}/Model_Weights")

    return value_transformer.inverse_transform(model.predict(predictFeatures).reshape(1, -1)).flatten()
