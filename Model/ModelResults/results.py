# imports
import pandas as pd
import numpy as np
import statistics as stats

# read results .csv file
df = pd.read_csv("./Results_Final.csv")

data = {
    "Property": [],
    "Minimum": [],
    "Maximum": [],
    "Mean": [],
    "Median": [],
    "Mode": [],
    "Mode_Amount": [],
    "Range": [],
    "TotalItems": []
}

# define which columns we need
cols = ["Units", "LR", "Test_Loss", "Test_Acc", "Val_Loss"]

# iterate through columns
for col in cols:
    # create list of current column
    listOfData = list(df[col])

    # perform statistical operations on the lists of data
    data["Property"].append(col)
    data["Minimum"].append(min(listOfData))
    data["Maximum"].append(max(listOfData))
    data["Mean"].append(int(np.mean(listOfData)) if col != "LR" else np.mean(listOfData))
    data["Median"].append(np.median(listOfData))
    data["Mode"].append(stats.mode(listOfData))
    data["Mode_Amount"].append(listOfData.count(stats.mode(listOfData)))
    data["Range"].append(max(listOfData) - min(listOfData))
    data["TotalItems"].append(len(listOfData))

# create dataframe of the data
overallResults = pd.DataFrame(data)

# save data to .csv file
overallResults.to_csv("./OverallResults.csv", mode="w", header=True, index=False)
