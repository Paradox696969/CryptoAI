# imports
import pandas as pd
import os
import sys

# defines variables that will be used in the directory loop
directory = "./CryptoData"
cdef dict f = {}
cdef dict times = {
                   "12-AM": "00:00:00", "01-AM": "01:00:00", "02-AM": "02:00:00",
                   "03-AM": "03:00:00", "04-AM": "04:00:00", "05-AM": "05:00:00",
                   "06-AM": "06:00:00", "07-AM": "07:00:00", "08-AM": "08:00:00",
                   "09-AM": "09:00:00", "10-AM": "10:00:00", "11-AM": "11:00:00",
                   "12-PM": "12:00:00", "01-PM": "13:00:00", "02-PM": "14:00:00",
                   "03-PM": "15:00:00", "04-PM": "16:00:00", "05-PM": "17:00:00",
                   "06-PM": "18:00:00", "07-PM": "19:00:00", "08-PM": "20:00:00",
                   "09-PM": "21:00:00", "10-PM": "22:00:00", "11-PM": "23:00:00"
                  }
cdef list temp_list = []
cdef list cols = []

# Colours for error detection ------------------------------------------------->
suggest = '\033[93m'
error = '\033[91m'
endc = '\033[0m'

# loops through all .csv files in the CryptoData directory -------------------->
for filename in os.listdir(directory):
    temp_list = filename.split("_")

    f["platform"] = temp_list[0]
    f["crypto"] = temp_list[1]

    # Checking Time Frame - used for checking which file data is appended to -->
    if temp_list[2].endswith(".csv"):
        temp_list[2] = temp_list[2][:-4]

    if temp_list[2] in ["d", "day", "daily"]:
        f["time_frame"] = "Daily"
    elif temp_list[2] in ["hour", "h", "1h", "1hr"]:
        f["time_frame"] = "Hourly"
    else:
        f["time_frame"] = "Minutely"

        if temp_list[2].lower() not in ['minutely', 'min', 'm', '1min', '1minute', 'minute']:
            f["minute_year"] = temp_list[2]
        else:
            f["minute_year"] = False

    if f["time_frame"] != "Minutely":
        continue

    # reads current .csv file ------------------------------------------------->
    df = pd.read_csv(f"{directory}/{filename}", low_memory=False)

    # creates a columns list by listing the column headers in the loaded file ->
    cols = list(df)

    # Warning for if first line of file is not removed ------------------------>
    if len(cols) <= 3:
        print(f"{error}Error: Header link unremoved - Data may become un-usable{endc}\n\n{suggest}Use remove.py to re-prep data{endc}\n\n{error}Terminating...{endc}")
        sys.exit()

    # Iterates through the .csv file and removes columns that are not found in -
    # every file -------------------------------------------------------------->
    for col in cols:
        if col in ["Volume USD", "Volume USDT", "Volume HUSD", "Volume EUSD", "Volume DUSD", "Volume AUSD", "Volume ESUSD", "Volume ATUSD", "tradecount"] and col != "Volume USDC":
            df = df.drop(col, axis=1)

    # Error Handling for .csv headers ----------------------------------------->
    # Replacing 12-hour time format with 24 hour time format ------------------>
    if f["time_frame"] in ["hourly", "daily"]:
        try:
            for key in times.keys():
                df["date"] = df["date"].str.replace(key, times[key])
        except:
            try:
                for key in times.keys():
                    df["Date"] = df["Date"].str.replace(key, times[key])
            except:
                pass


    # Adding in the Platform and Crypto columns for when the data is placed in -
    # a singular file --------------------------------------------------------->
    df["Platform"] = f["platform"]

    try:
        print(df["CryptoCurrency"])
    except:
        if f["crypto"][-5:] in ["Volume ESUSD", "Volume ATUSD"]:
            df["CryptoCurrency"] = f["crypto"][:-5]
        elif f["crypto"][-4:] in ["Volume USDT", "Volume HUSD", "Volume EUSD", "Volume DUSD", "Volume AUSD"]:
            df["CryptoCurrency"] = f["crypto"][:-4]
        else:
            df["CryptoCurrency"] = f["crypto"][:-3]

        print(df["CryptoCurrency"])

    print(filename, f)

    # Writing new data to singular file ---------------------------------------->

    if f["time_frame"] == "Minutely":
        try:
            try:
                file_to_write = open(f"./Model/CryptoModelData/{f['time_frame']}/{f['platform']}_{f['crypto']}.csv", "w")
                file_to_write.write("Unix Timestamp,Date,Symbol,Open,High,Low,Close,Volume,Platform,CryptoCurrency\n")
                file_to_write.close()
                df.to_csv(f"./Model/CryptoModelData/{f['time_frame']}/{f['platform']}_{f['crypto']}.csv", index=False, mode="a", header=None)
            except FileExistsError:
                pass
        except:
            try:
                os.mkdir(f"./Model/CryptoModelData/{f['time_frame']}/")
                file_to_write = open(f"./Model/CryptoModelData/{f['time_frame']}/{f['platform']}_{f['crypto']}.csv", "w")
                file_to_write.write("Unix Timestamp,Date,Symbol,Open,High,Low,Close,Volume,Platform,CryptoCurrency\n")
                file_to_write.close()
                df.to_csv(f"./Model/CryptoModelData/{f['time_frame']}/{f['platform']}_{f['crypto']}.csv", index=False, mode="a", header=None)
            except FileExistsError:
                pass

        df = pd.read_csv(f"./Model/CryptoModelData/{f['time_frame']}/{f['platform']}_{f['crypto']}.csv")

        if df["Unix Timestamp"][0] > df["Unix Timestamp"][1]:
            if not f["minute_year"]:
                df = df.loc[::-1]
            else:
                if f["minute_year"] == "2021":
                    df = df.loc[::-1]
            df.to_csv(f"./Model/CryptoModelData/{f['time_frame']}/{f['platform']}_{f['crypto']}.csv", index=False)
