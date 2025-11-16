import numpy as np
import pandas as pd
from copy import deepcopy

from sklearn.neighbors import BallTree

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import shap

class BicikeljDataset(Dataset):
    def __init__(self, df, station, training=True):

        if training:
            X_df, y_df, feature_count = BicikeljDataset._construct_training_dataframes(df, station)
            self.feature_count = feature_count
            self.X = torch.tensor(X_df.values, dtype=torch.float32)
            self.y = torch.tensor(y_df.values, dtype=torch.float32)

        else:
            X_df, feature_count = self._construct_testing_dataframes(df, station)
            self.X = torch.tensor(X_df.values, dtype=torch.float32)
            self.y = None
    
    @staticmethod
    def _construct_training_dataframes(df, station_name, max_lag=24):
        y_df = df[station_name]
        
        X_df = pd.DataFrame()
        for lag in range(1, max_lag + 1):
            X_df[f"{station_name}_t-{lag}"] = y_df.shift(lag)

        for neighbor in meta_df.loc[station_name]["neighbors"]:
            X_df[f"{neighbor}_t-1"] = df[neighbor].shift(1)
        
        X_df[temporal_columns] = df[temporal_columns]
        X_df[weather_columns] = df[weather_columns]

        y_df = pd.concat([y_df.shift(-i) for i in range(0, 1)], axis=1)
        y_df.columns = [f"t+i_{i}" for i in range(0, 1)]

        valid = X_df.notna().all(axis=1) & y_df.notna().all(axis=1)
        X_df = X_df[valid]
        y_df = y_df[valid]

        return X_df, y_df, X_df.shape[1]
    
    @staticmethod
    def _construct_testing_dataframes(df, station_name, max_lag=24):
        y_df = df[station_name]

        X_df = pd.DataFrame()

        for lag in range(1, max_lag + 1):
            X_df[f"{station_name}_t-{lag}"] = y_df.shift(lag)

        for neighbor in meta_df.loc[station_name]["neighbors"]:
            X_df[f"{neighbor}_t-1"] = df[neighbor].shift(1)

        X_df[temporal_columns] = df[temporal_columns]
        X_df[weather_columns] = df[weather_columns]

        X_df = X_df[X_df.notna().all(axis=1)]
        
        return X_df, X_df.shape[1]

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class BicikeljDataloader:
    def __init__(self, df, shuffle):
        self.feature_count = 0
        self.datasets = {}
        self.loaders = {}
        for station in station_names:
            dataset = BicikeljDataset(df, station)
            self.datasets[station] = dataset
            self.feature_count = dataset.feature_count
            
            loader = DataLoader(dataset, batch_size=256, shuffle=shuffle)
            self.loaders[station] = loader
            
    def __getitem__(self, station):
        return self.loaders[station]
    

class BicikeljMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

def load_csv(name : str):
    return pd.read_csv(f"data/{name}.csv")

station_names = np.array(['LIDL BEŽIGRAD',
 'ŠMARTINSKI PARK',
 'SAVSKO NASELJE 1-ŠMARTINSKA CESTA',
 'ČRNUČE',
 'VILHARJEVA CESTA',
 'MASARYKOVA DDC',
 'POGAČARJEV TRG-TRŽNICA',
 'CANKARJEVA UL.-NAMA',
 'ANTONOV TRG',
 'PRUŠNIKOVA',
 'TEHNOLOŠKI PARK',
 'KOSEŠKI BAJER',
 'TIVOLI',
 'TRŽNICA MOSTE',
 'GRUDNOVO NABREŽJE-KARLOVŠKA C.',
 'LIDL-LITIJSKA CESTA',
 'ŠPORTNI CENTER STOŽICE',
 'ŠPICA',
 'ROŠKA - STRELIŠKA',
 'BAVARSKI DVOR',
 'STARA CERKEV',
 'SITULA',
 'ILIRSKA ULICA',
 'LIDL - RUDNIK',
 'KOPALIŠČE KOLEZIJA',
 'POVŠETOVA - KAJUHOVA',
 'DUNAJSKA C.-PS MERCATOR',
 'CITYPARK',
 'KOPRSKA ULICA',
 'LIDL - VOJKOVA CESTA',
 'POLJANSKA-POTOČNIKOVA',
 'POVŠETOVA-GRABLOVIČEVA',
 'PARK NAVJE-ŽELEZNA CESTA',
 'ZALOG',
 'CESTA NA ROŽNIK',
 'HOFER-KAJUHOVA',
 'DUNAJSKA C.-PS PETROL',
 'STUDENEC',
 'PARKIRIŠČE NUK 2-FF',
 'BRATOVŠEVA PLOŠČAD',
 'KONGRESNI TRG-ŠUBIČEVA ULICA',
 'BS4-STOŽICE',
 'GERBIČEVA - ŠPORTNI PARK SVOBODA',
 'ŽIVALSKI VRT',
 'VOKA - SLOVENČEVA',
 'BTC CITY/DVORANA A',
 'TRNOVO',
 'P+R BARJE',
 'ROŽNA DOLINA-ŠKRABČEVA UL.',
 'KINO ŠIŠKA',
 'BRODARJEV TRG',
 'ZALOŠKA C.-GRABLOVIČEVA C.',
 'DOLENJSKA C. - STRELIŠČE',
 'ŠTEPANJSKO NASELJE 1-JAKČEVA ULICA',
 'SOSESKA NOVO BRDO',
 'TRŽNICA KOSEZE',
 'ALEJA - CELOVŠKA CESTA',
 'MERCATOR CENTER ŠIŠKA',
 'GH ŠENTPETER-NJEGOŠEVA C.',
 'HOFER - POLJE',
 'VIŠKO POLJE',
 'BONIFACIJA',
 'P + R DOLGI MOST',
 'DRAVLJE',
 'POLJE',
 'SUPERNOVA LJUBLJANA - RUDNIK',
 'SREDNJA FRIZERSKA ŠOLA',
 'TRG OF-KOLODVORSKA UL.',
 'TRG MDB',
 'TRŽAŠKA C.-ILIRIJA',
 'PREŠERNOV TRG-PETKOVŠKOVO NABREŽJE',
 'MERCATOR MARKET - CELOVŠKA C. 163',
 'SAVSKO NASELJE 2-LINHARTOVA CESTA',
 'BREG',
 'BTC CITY ATLANTIS',
 'IKEA',
 'MIKLOŠIČEV PARK',
 'BARJANSKA C.-CENTER STAREJŠIH TRNOVO',
 'LEK - VEROVŠKOVA',
 'AMBROŽEV TRG',
 'VOJKOVA - GASILSKA BRIGADA',
 'RAKOVNIK',
 'PREGLOV TRG',
 'PLEČNIKOV STADION'])
temporal_columns = np.array(["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend", "is_holiday"])
weather_columns = np.array(['temperature', 'precipitation', 'wind_speed'])

meta_df = load_csv("bicikelj_metadata")
meta_df.set_index("name", inplace=True)
coords = np.radians(meta_df[["latitude", "longitude"]].values)
tree = BallTree(coords, metric="haversine")
k = 3
_, idx = tree.query(coords, k=k)

neighbors = []
for i, name in enumerate(meta_df.index):
    nearest_indices = idx[i][1:]
    nearest_stations = meta_df.index[nearest_indices].tolist()
    neighbors.append(nearest_stations)

meta_df["neighbors"] = neighbors

holiday_df = load_csv("holiday_data")
holiday_df["timestamp"] = pd.to_datetime(holiday_df["timestamp"], format="%Y-%m-%dT%H:%MZ")
holidays = np.array(holiday_df["timestamp"].dt.date)

weather_df = load_csv("weather_data")
weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], format="%Y-%m-%dT%H:%MZ")
weather_df.set_index("timestamp", inplace=True)
weather_columns = weather_df.columns.to_numpy()

station_means = load_csv("station_means").set_index("Unnamed: 0")["0"]
station_stds = load_csv("station_stds").set_index("Unnamed: 0")["0"]

weather_means = load_csv("weather_means").set_index("Unnamed: 0")["0"]
weather_stds = load_csv("weather_stds").set_index("Unnamed: 0")["0"]

def prepare_dataframe(df):
    df["hour_sin"] =  np.sin(df.index.hour *    (2 * np.pi / 24))
    df["hour_cos"] =  np.cos(df.index.hour *    (2 * np.pi / 24))
    df["dow_sin"] =   np.sin(df.index.weekday * (2 * np.pi / 7 ))
    df["dow_cos"] =   np.cos(df.index.weekday * (2 * np.pi / 7 ))
    df["month_sin"] = np.sin(df.index.month *   (2 * np.pi / 12))
    df["month_cos"] = np.cos(df.index.month *   (2 * np.pi / 12))

    df["is_weekend"] = (df.index.weekday == 5) | (df.index.weekday == 6)
    df["is_weekend"] = df["is_weekend"].astype(float)

    df["is_holiday"] = np.isin(df.index.date, holidays)
    df["is_holiday"] = df["is_holiday"].astype(float)
    
    df = df.join(weather_df)
    return df

def normalize(df):
    df[station_names] = (df[station_names] - station_means) / station_stds
    df[weather_columns] = (df[weather_columns] - weather_means) / weather_stds
    return df

if __name__ == "__main__":

    seed = 1337

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    models = {}
    for i, station in enumerate(station_names):
        path = f"models/{i}.pth"
        model = BicikeljMLP(37)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        models[station] = model

    testing_df = load_csv("bicikelj_test")
    testing_df["timestamp"] = pd.to_datetime(testing_df["timestamp"], format="%Y-%m-%dT%H:%MZ")
    testing_df.set_index("timestamp", inplace=True)

    testing_df_prepared = prepare_dataframe(testing_df)
    testing_df_normalized = normalize(testing_df_prepared)

    missing_timestamps = testing_df.index[testing_df[station].isna()].tolist()

    submission_df = pd.DataFrame(index=missing_timestamps, columns=station_names)

    for i, timestamp in enumerate(missing_timestamps):
        print(f"Predicting {missing_timestamps[i].strftime(format="%Y-%m-%d %Hh")} ({i + 1}/{len(missing_timestamps)})")
        
        model.eval()
        with torch.no_grad():
                
            for station, model in models.items():

                sub_df = testing_df_normalized.loc[:timestamp - pd.Timedelta(hours=1)].copy()

                dataset = BicikeljDataset(sub_df, station, training=False)
                X_test = dataset.X
                
                X_df = X_test[-1].unsqueeze(0)

                prediction_normalized = model(X_df).item()
                prediction_count = prediction_normalized * station_stds[station] + station_means[station]

                testing_df_normalized.loc[timestamp, station] = prediction_normalized
                submission_df.loc[timestamp, station] = prediction_count

    submission_df_1 = submission_df.reset_index()
    submission_df_1 = submission_df_1.rename(columns={"index": "timestamp"})
    submission_df_1["timestamp"] = pd.to_datetime(submission_df_1["timestamp"])
    submission_df_1["timestamp"] = submission_df_1["timestamp"].dt.strftime("%Y-%m-%dT%H:%MZ")

    submission_df_1.to_csv("data/final_predictions.csv", index=False)