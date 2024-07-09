#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple, List


def read_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    return df


def get_df_shape(df: pd.DataFrame) -> Tuple[int, int]:
    return df.shape


def extract_features_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    features = df[["Lag1", "Lag2"]]
    label = df["Direction"]
    return features, label


def data_split(
    features: pd.DataFrame, label: pd.Series, test_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, y_test = train_test_split(
        features, label, test_size=test_size
    )
    return x_train, y_train, x_test, y_test


def knn_test_score(
    n_neighbors: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    knn = KNeighborsClassifier(n_neighbors)
    fit = knn.fit(x_train, y_train)
    return knn.score(x_test, y_test)


def knn_evaluate_with_neighbours(
    n_neighbors_min: int,
    n_neighbors_max: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> List[float]:
    # Note neighbours_min, neighbours_max are inclusive
    vals = []
    for i in range(n_neighbors_min, n_neighbors_max + 1):
        vals.append(knn_test_score(i, x_train, y_train, x_test, y_test))
    return vals


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df = read_data("Smarket2.csv")
    # assert on df
    shape = get_df_shape(df)
    # assert on shape
    features, label = extract_features_label(df)
    x_train, y_train, x_test, y_test = data_split(features, label, 0.33)
    print(knn_test_score(1, x_train, y_train, x_test, y_test))
    acc = knn_evaluate_with_neighbours(1, 10, x_train, y_train, x_test, y_test)
    plt.plot(range(1, 11), acc)
    plt.show()
