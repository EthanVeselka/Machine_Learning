#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple, List, Optional, Any, Callable, Dict, Union
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import roc_auc_score
import random
from typeguard import typechecked

random.seed(42)
np.random.seed(42)


@typechecked
def read_data(filename: str) -> pd.DataFrame:
    df1 = pd.read_csv(filename)
    return df1


@typechecked
def data_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    data = df.dropna()
    data = data.drop("Player", axis=1)  # drops player names from data set
    label = data["NewLeague"]  # selects label
    features = data.drop(
        "NewLeague", axis=1
    )  # selects all the other columns (features)

    nonnum = features.select_dtypes(exclude=["int64", "float64"])
    num = features.select_dtypes(include=["int64", "float64"])
    dummy_nonnum = pd.get_dummies(nonnum)

    t = {"A": 0, "N": 1}
    final_label = label.replace(to_replace=t)
    final_features = pd.concat([dummy_nonnum, num], axis=1)
    return final_features, final_label


@typechecked
def data_split(
    features: pd.DataFrame, label: pd.Series, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x_train, x_test, y_train, y_test = train_test_split(
        features, label, test_size=test_size
    )
    return x_train, x_test, y_train, y_test


@typechecked
def train_ridge_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    max_iter: int = int(1e8),
) -> Dict[float, float]:
    n = int(1e3)
    aucs = {"ridge": []}
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]

    ridge_model = Ridge()
    for i in range(n):
        vals = []
        for alpha in lambda_vals:
            ridge_model = Ridge(alpha, max_iter=max_iter)
            ridge_model.fit(x_train, y_train)  # fit
            predictions = ridge_model.predict(x_test)  # predict
            roc = roc_auc_score(y_test, predictions)  # score

            vals.append(roc)  # add score to list of scores for alpha
        aucs["ridge"].append(vals)  # add list of values for iteration to dictionary

    print("ridge mean AUCs:")
    ridge_mean_auc = {}
    array = np.array(aucs["ridge"])
    mean_array = array.sum(axis=0) / n
    for i, alpha in enumerate(lambda_vals):
        ridge_mean_auc[alpha] = mean_array[i]
        print("lambda:", lambda_vals[i], "AUC:", ridge_mean_auc[alpha])
    return ridge_mean_auc


@typechecked
def train_lasso(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    max_iter=int(1e8),
) -> Dict[float, float]:
    n = int(1e3)
    aucs = {"lasso": []}
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]

    lasso_model = Lasso()
    for i in range(n):
        vals = []
        for alpha in lambda_vals:
            lasso_model = Lasso(alpha, max_iter=max_iter)
            lasso_model.fit(x_train, y_train)  # fit
            predictions = lasso_model.predict(x_test)  # predict
            roc = roc_auc_score(y_test, predictions)  # score

            vals.append(roc)  # add score to list of scores for alpha
        aucs["lasso"].append(vals)  # add list of values for iteration to dictionary
        # nparray = np.array(vals)           #convert into np array
        # aucs["lasso"] = np.mean(nparray)   #store average roc score for the alpha in auc dictionary  -----CAN BE CHANGED-----

    print("lasso mean AUCs:")
    lasso_mean_auc = {}
    array = np.array(aucs["lasso"])
    mean_array = array.sum(axis=0) / n
    for i, alpha in enumerate(lambda_vals):
        lasso_mean_auc[alpha] = mean_array[i]
        print("lambda:", lambda_vals[i], "AUC:", lasso_mean_auc[alpha])
    return lasso_mean_auc


@typechecked
def ridge_coefficients(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    optimal_alpha: float,
    max_iter=int(1e8),
) -> Tuple[Ridge, np.ndarray]:

    ridge_model = Ridge(optimal_alpha, max_iter=max_iter)
    ridge_model.fit(x_train, y_train)
    return ridge_model, ridge_model.coef_


@typechecked
def lasso_coefficients(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    optimal_alpha: float,
    max_iter=int(1e8),
) -> Tuple[Lasso, np.ndarray]:

    lasso_model = Lasso(optimal_alpha, max_iter=max_iter)
    lasso_model.fit(x_train, y_train)
    return lasso_model, lasso_model.coef_


@typechecked
def ridge_area_under_curve(model_R, x_test: pd.DataFrame, y_test: pd.Series) -> float:

    predictions = model_R.predict(x_test)
    auc = roc_auc_score(y_test, predictions)
    return auc


@typechecked
def lasso_area_under_curve(model_L, x_test: pd.DataFrame, y_test: pd.Series) -> float:

    predictions = model_L.predict(x_test)
    auc = roc_auc_score(y_test, predictions)
    return auc


class Node:
    @typechecked
    def __init__(
        self,
        split_val: float,
        data: Any = None,
        left: Any = None,
        right: Any = None,
    ) -> None:
        if left is not None:
            assert isinstance(left, Node)

        if right is not None:
            assert isinstance(right, Node)

        self.left = left
        self.right = right
        self.split_val = split_val
        self.data = data


class TreeRegressor:
    @typechecked
    def __init__(self, data: np.ndarray, max_depth: int) -> None:
        self.data = data
        self.max_depth = max_depth  # maximum depth

    @typechecked
    def build_tree(self) -> Node:

        node = self.get_best_split(self.data)
        self.split(node, 1)
        return node

    @typechecked
    def mean_squared_error(
        self, left_split: np.ndarray, right_split: np.ndarray
    ) -> float:

        ly = left_split[:, -1]
        lym = np.mean(ly)
        sum1 = ((lym - ly) ** 2).sum()
        ry = right_split[:, -1]
        rym = np.mean(ry)
        sum2 = ((rym - ry) ** 2).sum()

        return (sum1 + sum2) / (left_split.shape[0] + right_split.shape[0])

    @typechecked
    def split(self, node: Node, depth: int) -> None:
        if depth == self.max_depth:
            node.left = Node(node.data["left_pred"])
            node.right = Node(node.data["right_pred"])
        elif node.data["left"].shape[0] == 1 & node.data["right"].shape[0] == 1:
            node.left = Node(node.data["left_pred"])
            node.right = Node(node.data["right_pred"])
        elif node.data["left"].shape[0] == 1 & node.data["right"].shape[0] != 1:
            node.left = Node(node.data["left_pred"])
            node.right = self.get_best_split(node.data["right"])
            self.split(node.right, depth + 1)
        elif node.data["left"].shape[0] != 1 & node.data["right"].shape[0] == 1:
            node.left = self.get_best_split(node.data["left"])
            node.right = Node(node.data["right_pred"])
            self.split(node.left, depth + 1)
        elif node.data["left"].shape[0] != 1 & node.data["right"].shape[0] != 1:
            node.left = self.get_best_split(node.data["left"])
            node.right = self.get_best_split(node.data["right"])
            self.split(node.left, depth + 1)
            self.split(node.right, depth + 1)

    @typechecked
    def get_best_split(self, data: np.ndarray) -> Node:
        mse_min = {"index": 0, "value": np.inf, "mse": np.inf}
        for i in range(data.shape[1] - 1):
            val = (data[1:, i] + data[:-1, i]) / 2
            for j in range(val.shape[0]):
                left, right = self.one_step_split(i, val[j], data)
                mse = self.mean_squared_error(left, right)
                if mse <= mse_min["mse"]:
                    mse_min["index"] = i
                    mse_min["value"] = val[j]
                    mse_min["mse"] = mse

        left, right = self.one_step_split(mse_min["index"], mse_min["value"], data)
        # print("Min mse: ", mse_min['mse'])
        # print("Split val: ", mse_min['value'])
        kvp = {
            "index": mse_min["index"],
            "left": left,
            "right": right,
            "left_pred": left[:, -1].mean(),
            "right_pred": right[:, -1].mean(),
        }  # stores kvp: threshold value as key, index of column the node split on as value
        node = Node(mse_min["value"], kvp)
        return node

    @typechecked
    def one_step_split(
        self, index: int, value: float, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        right = []
        left = []

        for i in range(data.shape[0]):
            if data[i, index] < value:
                left.append(data[i, :])  # append row to list
            else:
                right.append(data[i, :])
        left = np.array(left)
        right = np.array(right)
        return left, right


@typechecked
def compare_node_with_threshold(node: Node, row: np.ndarray) -> bool:

    index = node.data["index"]
    if node.split_val >= row[index]:
        return True
    else:
        return False


@typechecked
def predict(
    node: Node, row: np.ndarray, comparator: Callable[[Node, np.ndarray], bool]
) -> float:
    if node.left == None and node.right == None:  # return val at end of tree
        return node.split_val
    elif comparator(node, row):
        return predict(
            node.left, row, comparator
        )  # if split value is > row value, go right
    elif not comparator(node, row):
        return predict(
            node.right, row, comparator
        )  # else if split val is < row val, go left


class TreeClassifier(TreeRegressor):
    def build_tree(self):
        node = self.get_best_split(self.data)
        self.split(node, 1)
        return node

    @typechecked
    def gini_index(
        self,
        left_split: np.ndarray,
        right_split: np.ndarray,
        classes: List[float],
    ) -> float:

        ly = left_split[:, -1]
        lp = ((ly == classes[0]).sum()) / len(ly)  # prob of class[0] in left labels
        ldiff = 1 - lp  # prob difference from 1
        sum1 = 2 * lp * ldiff * len(ly)  # weight index by number of samples
        ry = right_split[:, -1]
        rp = ((ry == classes[0]).sum()) / len(ry)  # prob of class[0] in right labels
        rdiff = 1 - rp  # prob difference from 1
        sum2 = 2 * rp * rdiff * len(ry)  # weight index by number of samples

        return sum1 + sum2

    @typechecked
    def get_best_split(self, data: np.ndarray) -> Node:

        classes = list(set(row[-1] for row in data))
        gi_min = {"index": 0, "value": np.inf, "gi": np.inf}
        for i in range(data.shape[1] - 1):
            val = (data[1:, i] + data[:-1, i]) / 2
            for j in range(val.shape[0]):
                left, right = self.one_step_split(i, val[j], data)
                gi = self.gini_index(left, right, classes)
                if gi <= gi_min["gi"]:
                    gi_min["index"] = i
                    gi_min["value"] = val[j]
                    gi_min["gi"] = gi

        left, right = self.one_step_split(gi_min["index"], gi_min["value"], data)

        # stores kvp: threshold value as key, index of column the node split on as value
        class1_left = (left[:, -1] == classes[0]).sum()
        class2_left = len(left[:, -1]) - class1_left
        class1_right = (right[:, -1] == classes[0]).sum()
        class2_right = len(right[:, -1]) - class1_right
        maxclassleft = classes[0] if class1_left > class2_left else (classes[0] * -1)
        maxclassright = classes[0] if class1_right > class2_right else (classes[0] * -1)
        right_pred = maxclassright
        left_pred = maxclassleft

        kvp = {
            "index": gi_min["index"],
            "left": left,
            "right": right,
            "left_pred": left_pred,
            "right_pred": right_pred,
        }
        # pred =  max count between classes for each split

        node = Node(gi_min["value"], kvp)
        return node


if __name__ == "__main__":
    filename = "Hitters.csv"
    df = read_data(filename)
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    max_iter = 1e8
    final_features, final_label = data_preprocess(df)
    x_train, x_test, y_train, y_test = data_split(final_features, final_label, 0.2)
    ridge_mean_acu = train_ridge_regression(x_train, y_train, x_test, y_test)
    lasso_mean_acu = train_lasso(x_train, y_train, x_test, y_test)
    model_R, ridge_coeff = ridge_coefficients(x_train, y_train, 10)
    model_L, lasso_coeff = lasso_coefficients(x_train, y_train, 0.1)

    ridge_auc = ridge_area_under_curve(model_R, x_test, y_test)
    ridge_pred = model_R.predict(x_test)
    ridge_fpr, ridge_tpr, ridge_threshold = roc_curve(y_test, ridge_pred)
    plt.figure(1)
    plt.plot(ridge_fpr, ridge_tpr, label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title("Ridge Regression ROC Curve")
    plt.savefig("./images/Q1plot1")

    lasso_auc = lasso_area_under_curve(model_L, x_test, y_test)
    lasso_pred = model_L.predict(x_test)
    lasso_fpr, lasso_tpr, lasso_threshold = roc_curve(y_test, lasso_pred)
    plt.figure(2)
    plt.plot(lasso_fpr, lasso_tpr, label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title("Lasso Regression ROC Curve")
    plt.savefig("./images/Q1plot2")

    csvname = "noisy_sin_subsample_2.csv"
    data_regress = np.loadtxt(csvname, delimiter=",")
    data_regress = np.array([[x, y] for x, y in zip(*data_regress)])
    plt.figure()
    plt.scatter(data_regress[:, 0], data_regress[:, 1])
    plt.xlabel("Features, x")
    plt.ylabel("Target values, y")
    plt.title("Features vs. Target Values")
    plt.savefig("./images/Q2plot1")

    mse_depths = []
    for depth in range(1, 5):
        regressor = TreeRegressor(data_regress, depth)
        tree = regressor.build_tree()
        mse = 0.0
        for data_point in data_regress:
            mse += (
                data_point[1] - predict(tree, data_point, compare_node_with_threshold)
            ) ** 2
        mse_depths.append(mse / len(data_regress))

    plt.figure()
    plt.plot(range(1, 5), mse_depths)
    plt.xlabel("Depth")
    plt.ylabel("MSE")
    plt.title("MSE vs. Depth")
    plt.savefig("./images/Q2plot2")

    csvname = "new_circle_data.csv"
    data_class = np.loadtxt(csvname, delimiter=",")
    data_class = np.array([[x1, x2, y] for x1, x2, y in zip(*data_class)])
    plt.figure()
    plt.scatter(data_class[:, 0], data_class[:, 1], c=-data_class[:, 2], cmap="bwr")
    plt.xlabel("Features, x1")
    plt.ylabel("Features, x2")
    plt.show()

    accuracy_depths = []
    for depth in range(1, 8):
        classifier = TreeClassifier(data_class, depth)
        tree = classifier.build_tree()
        correct = 0.0
        for data_point in data_class:
            correct += float(
                data_point[2] == predict(tree, data_point, compare_node_with_threshold)
            )
        accuracy_depths.append(correct / len(data_class))

    plt.figure()
    plt.plot(accuracy_depths)
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Depth")
    plt.savefig("./images/Q2plot3")
