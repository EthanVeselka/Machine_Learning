import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, log_loss
import matplotlib.pyplot as plt
import random
from typeguard import typechecked
from typing import Tuple, List, Optional, Any, Callable, Dict, Union


random.seed(42)
np.random.seed(42)


@typechecked
def read_classification_data(file_path: str) -> Tuple[np.array, np.array]:
    data = pd.read_csv(file_path, header=None)
    first_row = np.array(data.iloc[0]).reshape(-1, 1)
    second_row = np.array(data.iloc[1]).reshape(-1, 1)

    return first_row, second_row


@typechecked
def sigmoid(s: np.array) -> np.array:
    return 1 / (1 + np.exp(-s))


@typechecked
def cost_function(w: float, b: float, X: np.array, y: np.array) -> float:
    yhat = sigmoid(b + X * w)
    loss = -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
    return loss.sum()


@typechecked
def cross_entropy_optimizer(
    w: float, b: float, X: np.array, y: np.array, num_iterations: int, alpha: float
) -> (float, float, list):
    w0 = w
    b0 = b
    delta = []
    for i in range(num_iterations):
        loss = cost_function(w0, b0, X, y)
        yhat = sigmoid(b0 + X * w0)
        grad = ((yhat - y) * X).mean(axis=0)
        w0 -= alpha * grad[0]
        grad = (yhat - y).mean(axis=0)
        b0 -= alpha * grad[0]
        delta.append(loss)
    return w0, b0, delta


@typechecked
def read_sat_image_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    return data


@typechecked
def remove_nan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    return df


@typechecked
def normalize_data(
    Xtrain: pd.DataFrame, Xtest: pd.DataFrame
) -> (pd.DataFrame, pd.DataFrame):
    scaler = StandardScaler()
    Xtrain_norm = pd.DataFrame(scaler.fit_transform(Xtrain))
    Xtest_norm = pd.DataFrame(scaler.fit_transform(Xtest))
    return Xtrain_norm, Xtest_norm


@typechecked
def labels_to_binary(y: pd.DataFrame) -> pd.DataFrame:
    y.replace({1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1}, inplace=True)
    return y


@typechecked
def cross_validate_c_vals(
    X: pd.DataFrame, y: pd.DataFrame, n_folds: int, c_vals: np.array, d_vals: np.array
) -> (np.array, np.array):
    ERRAVGdc = np.zeros((len(c_vals), len(d_vals)))
    ERRSTDdc = np.zeros((len(c_vals), len(d_vals)))
    for c in range(len(c_vals)):
        for d in range(len(d_vals)):
            mean_errors = []
            skf = StratifiedKFold(n_folds)
            for k, (train_index, test_index) in enumerate(skf.split(X, y)):
                x_train, y_train = X.iloc[train_index], y.iloc[train_index]
                x_test, y_test = X.iloc[test_index], y.iloc[test_index]
                svc = SVC(C=c_vals[c], kernel="poly", degree=d_vals[d])
                y_train = np.array(y_train).reshape(-1)
                y_test = np.array(y_test).reshape(-1)
                svc = svc.fit(x_train, y_train)
                pred = svc.predict(x_test)
                mean_errors.append(mean_absolute_error(y_test, pred))
            mean = np.array(mean_errors).mean()
            std = np.array(mean_errors).std()
            ERRAVGdc[c][d] = mean
            ERRSTDdc[c][d] = std

    return ERRAVGdc, ERRSTDdc


@typechecked
def plot_cross_val_err_vs_c(
    ERRAVGdc: np.array, ERRSTDdc: np.array, c_vals: np.array, d_vals: np.array
) -> None:
    plt.figure()
    for d in range(len(d_vals)):
        row = ERRAVGdc[:, d].reshape(
            5,
        )
        plt.plot(c_vals, row, label=d + 1)

    plt.xscale("log")
    plt.legend()
    plt.title("Mean Error For (c, d) Pairs")
    plt.xlabel("C values")
    plt.ylabel("Mean Error at Degree d")
    plt.savefig("./images/Q3plot1")


@typechecked
def evaluate_c_d_pairs(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    n_folds: int,
    c_vals: np.array,
    d_vals: np.array,
) -> (np.array, np.array, np.array, np.array):
    ERRAVGdcTEST = np.zeros(len(d_vals))
    SuppVect = np.zeros(len(d_vals))
    vmd = np.zeros(len(d_vals))
    MarginT = np.zeros(len(d_vals))
    for i in range(len(d_vals)):
        y_train0 = np.array(y_train).reshape(-1)
        y_test0 = np.array(y_test).reshape(-1)
        svc = SVC(C=c_vals[i], kernel="poly", degree=d_vals[i])
        svc = svc.fit(X_train, y_train0)
        pred = svc.predict(X_test)
        ERRAVGdcTEST[i] = mean_absolute_error(y_test0, pred)
        SuppVect[i] = len(svc.support_)
        alphas = np.abs(svc.dual_coef_)
        vmd[i] = (alphas == c_vals[i]).sum()
        MarginT[i] = 1 / np.linalg.norm(np.dot(svc.dual_coef_, svc.support_vectors_))

    return ERRAVGdcTEST, SuppVect, vmd, MarginT


@typechecked
def plot_test_errors(ERRAVGdcTEST: np.array, d_vals: np.array) -> None:

    plt.figure()
    plt.plot(d_vals, ERRAVGdcTEST)
    plt.title("Mean Error For Optimal (c, d) Pairs")
    plt.xlabel("D-values")
    plt.ylabel("Mean Error")
    plt.savefig("./images/Q3plot2")


@typechecked
def plot_avg_support_vec(SuppVect: np.array, d_vals: np.array) -> None:
    plt.figure()
    plt.plot(d_vals, SuppVect)
    plt.title("Average Number of Support Vectors vs. D-values")
    plt.xlabel("D-values")
    plt.ylabel("Number of Support Vectors")
    plt.savefig("./images/Q3plot3")


@typechecked
def plot_avg_violating_support_vec(vmd: np.array, d_vals: np.array) -> None:
    plt.figure()
    plt.plot(d_vals, vmd)
    plt.title("Average Number of Support Vectors Violating Margin vs. D-values")
    plt.xlabel("D-values")
    plt.ylabel("Number of Violating Support Vectors")
    plt.savefig("./images/Q3plot4")


@typechecked
def plot_avg_hyperplane_margins(MarginT: np.array, d_vals: np.array) -> None:
    plt.figure()
    plt.yscale("log")
    plt.plot(d_vals, MarginT)
    plt.title("Average Hyperplane Margin vs. D-values")
    plt.xlabel("D-values")
    plt.ylabel("Hyperplane Margin")
    plt.savefig("./images/Q3plot5")


if __name__ == "__main__":
    classification_data_2d_path = "2d_classification_data_entropy.csv"
    x, y = read_classification_data(classification_data_2d_path)

    w = 1
    b = 1
    num_iterations = 300
    w, b, costs = cross_entropy_optimizer(w, b, x, y, num_iterations, 0.1)
    print("Weignt W: ", w)
    print("Bias b: ", b)
    plt.plot(range(num_iterations), costs)
    plt.savefig("./images/Q1plot")

    sat_image_Training_path = "satimageTraining.csv"
    sat_image_Test_path = "satimageTest.csv"

    train_df = read_sat_image_data(sat_image_Training_path)  # Training set
    test_df = read_sat_image_data(sat_image_Test_path)  # Testing set

    train_df_nan_removed = remove_nan(train_df)
    test_df_nan_removed = remove_nan(test_df)

    ytrain = train_df_nan_removed[["Class"]]
    Xtrain = train_df_nan_removed.drop(["Class"], axis=1)

    ytest = test_df_nan_removed[["Class"]]
    Xtest = test_df_nan_removed.drop(["Class"], axis=1)

    Xtrain_norm, Xtest_norm = normalize_data(Xtrain, Xtest)

    ytrain_bin_label = labels_to_binary(ytrain)
    ytest_bin_label = labels_to_binary(ytest)

    c_vals = np.power(float(10), range(-2, 2 + 1))
    n_folds = 5
    d_vals = np.array([1, 2, 3, 4])

    ERRAVGdc, ERRSTDdc = cross_validate_c_vals(
        Xtrain_norm, ytrain_bin_label, n_folds, c_vals, d_vals
    )

    plot_cross_val_err_vs_c(ERRAVGdc, ERRSTDdc, c_vals, d_vals)

    d_vals = [1, 2, 3, 4]
    n_folds = 5
    new_c_vals = [10, 100, 100, 100]

    ERRAVGdcTEST, SuppVect, vmd, MarginT = evaluate_c_d_pairs(
        Xtrain_norm,
        ytrain_bin_label,
        Xtest_norm,
        ytest_bin_label,
        n_folds,
        new_c_vals,
        d_vals,
    )
    plot_test_errors(ERRAVGdcTEST, d_vals)
    plot_avg_support_vec(SuppVect, d_vals)
    plot_avg_violating_support_vec(vmd, d_vals)
    plot_avg_hyperplane_margins(MarginT, d_vals)
