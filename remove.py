import os

directory = "./CryptoData"

for filename in os.listdir(directory):
    if not filename.startswith("Kraken") and not filename.startswith("Coinbase"):
        f = open(f"{directory}/{filename}", "r")
        save = f.read().splitlines(True)
        f.close()
        f = open(f"{directory}/{filename}", "w")
        f.writelines(save[1:])
