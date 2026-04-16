import pandas as pd

def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    store = pd.read_csv("data/store.csv")

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("Store shape:", store.shape)

    return train, test, store


def merge_data(train, store):
    df = pd.merge(train, store, on="Store", how="left")
    df["Date"] = pd.to_datetime(df["Date"])
    print("Merged data shape:", df.shape)
    return df


if __name__ == "__main__":
    train, test, store = load_data()
    df = merge_data(train, store)

    print(df.head())