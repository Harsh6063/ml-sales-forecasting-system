# features.py

import pandas as pd


def build_features(df):
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(by=["Store", "Date"])


    df["day"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    df["day_of_week"] = df["Date"].dt.weekday
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["StateHoliday"] = df["StateHoliday"].apply(lambda x: 0 if x == '0' else 1)

   
    df["lag_1"] = df.groupby("Store")["Sales"].shift(1)
    df["lag_7"] = df.groupby("Store")["Sales"].shift(7)
    df["lag_14"] = df.groupby("Store")["Sales"].shift(14)


    df["rolling_mean_7"] = df.groupby("Store")["Sales"].shift(1).rolling(7).mean()
    df["rolling_std_7"] = df.groupby("Store")["Sales"].shift(1).rolling(7).std()

   
    df = df.dropna()

   
    if "StoreType" in df.columns:
        df["StoreType"] = df["StoreType"].astype("category").cat.codes

    if "Assortment" in df.columns:
        df["Assortment"] = df["Assortment"].astype("category").cat.codes

   
    features = [
        "Store",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
        "day",
        "month",
        "year",
        "day_of_week",
        "week_of_year",
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_std_7",
    ]

    # Keep only available columns
    features = [col for col in features if col in df.columns]

    X = df[features]
    y = df["Sales"]

    print("Feature shape:", X.shape)

    return X, y


if __name__ == "__main__":
    # Quick test
    from ingest import load_data, merge_data

    train, test, store = load_data()
    df = merge_data(train, store)
    df.to_csv("data/merged_data.csv", index=False)  

    X, y = build_features(df)
    X.to_csv("data/features.csv", index=False)
    y.to_csv("data/targets.csv", index=False)
    print(X.head())
    print(y.head())