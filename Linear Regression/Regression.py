import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score
from typing import Tuple, List
from scipy import stats


# Download and read the data.
def read_train_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    return df


def read_test_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    return df


def prepare_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    cleandf_train = df_train.dropna()
    cleandf_test = df_test.dropna()
    train_data = cleandf_train.iloc[:, 0]
    train_label = cleandf_train.iloc[:, 1]
    test_data = cleandf_test.iloc[:, 0]
    test_label = cleandf_test.iloc[:, 1]
    return train_data, train_label, test_data, test_label


# Implement LinearRegression class
class LinearRegression_local:
    def __init__(self, learning_rate=0.00001, iterations=30):
        self.learning_rate = learning_rate
        self.iterations = iterations

    # Function for model training
    def fit(self, X, Y):
        # weight initialization
        self.weights = np.zeros(2)

        # gradient descent learning
        for i, e in enumerate(range(self.iterations)):
            self.update_weights(X, Y)

        return self

    # Helper function to update weights in gradient descent
    def update_weights(self, X, Y):
        # predict on data and calculate gradients
        grad0 = (-2 / len(X)) * (Y - self.weights[0] - self.weights[1] * X).sum()
        grad1 = (-2 / len(X)) * (
            (Y * X) - (self.weights[0] * X) - (self.weights[1] * X * X)
        ).sum()

        # update weights
        self.weights[0] -= self.learning_rate * grad0
        self.weights[1] -= self.learning_rate * grad1

    # Hypothetical function  h( x )
    def predict(self, X) -> np.array:
        exp = self.weights[0] + self.weights[1] * X
        return exp


# Build model
def build_model(train_x: np.array, train_y: np.array):
    model = LinearRegression_local()
    model.iterations = 1000
    model.learning_rate = 0.0001
    model.fit(train_x, train_y)
    return model


# Make predictions with test set
def pred_func(model, X_test):
    predictions = model.predict(X_test)
    return predictions


# Calculate and print the mean square error of the prediction
def MSE(y_test, pred):
    sum = 0
    for i in range(y_test.size):
        sum += (y_test[i] - pred[i]) ** 2

    return sum / y_test.size


# Download and read the data.
def read_training_data(filename: str) -> tuple:
    df1 = pd.read_csv(filename)
    df2 = df1.head(10)
    return df1, df2, df1.shape


# Prepare your input data and labels
def data_clean(df_train: pd.DataFrame) -> tuple:
    nan_count = df_train.isna().sum().sum()
    s = pd.Series(np.nan, list(range(nan_count)))
    df = df_train.dropna()
    print("-------------------------")
    print("Total missing values: ", nan_count)
    print("-------------------------")
    return s, df


def feature_extract(df_train: pd.DataFrame) -> tuple:
    label = df_train["NewLeague"]
    features = df_train.drop("NewLeague", axis=1)
    return features, label


def data_preprocess(features: pd.DataFrame) -> pd.DataFrame:
    nonnum = features.select_dtypes(exclude=["int64", "float64"])
    num = features.select_dtypes(include=["int64", "float64"])
    dummy_nonnum = pd.get_dummies(nonnum)
    new_df = pd.concat([dummy_nonnum, num], axis=1)
    return new_df


def label_transform(labels: pd.Series) -> pd.Series:
    t = {"A": 0, "N": 1}
    labels = labels.replace(to_replace=t)
    return labels


def data_split(
    features: pd.DataFrame, label: pd.Series, random_state=42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    x_train, x_test, y_train, y_test = train_test_split(
        features, label, test_size=0.2, random_state=random_state
    )
    return x_train, x_test, y_train, y_test


def train_linear_regression(x_train: np.ndarray, y_train: np.ndarray):

    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def train_logistic_regression(
    x_train: np.ndarray, y_train: np.ndarray, max_iter=1000000
):
    model = LogisticRegression(max_iter=max_iter)
    model.fit(x_train, y_train)
    return model


def models_coefficients(linear_model, logistic_model) -> Tuple[np.ndarray, np.ndarray]:
    lincoef = linear_model.coef_
    logcoef = logistic_model.coef_
    return lincoef, logcoef


def linear_pred_and_area_under_curve(
    linear_model, x_test: np.ndarray, y_test: np.ndarray
) -> Tuple[np.array, np.array, np.array, np.array, float]:
    linear_reg_pred = linear_model.predict(x_test)
    linear_reg_fpr, linear_reg_tpr, linear_threshold = metrics.roc_curve(
        y_test, linear_reg_pred
    )
    linear_reg_area_under_curve = metrics.roc_auc_score(y_test, linear_reg_pred)
    return (
        linear_reg_pred,
        linear_reg_fpr,
        linear_reg_tpr,
        linear_threshold,
        linear_reg_area_under_curve,
    )


def logistic_pred_and_area_under_curve(
    logistic_model, x_test: np.ndarray, y_test: np.ndarray
) -> Tuple[np.array, np.array, np.array, np.array, float]:
    log_reg_pred = logistic_model.predict_proba(x_test)[:, 1]
    log_reg_fpr, log_reg_tpr, log_threshold = metrics.roc_curve(y_test, log_reg_pred)
    log_reg_area_under_curve = metrics.roc_auc_score(y_test, log_reg_pred)
    return (
        log_reg_pred,
        log_reg_fpr,
        log_reg_tpr,
        log_threshold,
        log_reg_area_under_curve,
    )


def optimal_thresholds(
    linear_threshold: np.ndarray,
    linear_reg_fpr: np.ndarray,
    linear_reg_tpr: np.ndarray,
    log_threshold: np.ndarray,
    log_reg_fpr: np.ndarray,
    log_reg_tpr: np.ndarray,
) -> Tuple[float, float]:
    linear_threshold = linear_threshold[(linear_reg_tpr - linear_reg_fpr).argmax()]
    log_threshold = log_threshold[(log_reg_tpr - log_reg_fpr).argmax()]
    return linear_threshold, log_threshold


def stratified_k_fold_cross_validation(
    num_of_folds: int, features: pd.DataFrame, label: pd.Series
):

    skf = StratifiedKFold(num_of_folds)
    return skf


def train_test_folds(
    skf, num_of_folds: int, X, y
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    lin_model = LinearRegression()
    log_model = LogisticRegression(max_iter=100000008)
    auc_log = []
    auc_linear = []
    features_count = []
    f1_dict = {"log_reg": [], "linear_reg": []}

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print("Fold: ", i)
        x_train, y_train = X.iloc[train_index], y.iloc[train_index]
        x_test, y_test = X.iloc[test_index], y.iloc[test_index]
        features_count.append(x_test.shape[1])

        lin_model = lin_model.fit(x_train, y_train)
        lin_pred = lin_model.predict(x_test).round()
        # print(lin_pred)
        # print(y_test.to_numpy())
        # print(x_train, x_test)
        # print(y_test.to_numpy() - lin_pred)

        auc_linear.append(metrics.roc_auc_score(y_test, lin_pred))
        f1_dict["linear_reg"].append(metrics.f1_score(y_test, lin_pred))

        log_model = log_model.fit(x_train, y_train)
        log_pred = log_model.predict_proba(x_test)[:, 1].round()
        auc_log.append(metrics.roc_auc_score(y_test, log_pred))
        f1_dict["log_reg"].append(metrics.f1_score(y_test, log_pred))

        # log_y_pred, log_reg_fpr, log_reg_tpr, log_reg_area_under_curve, log_threshold = logistic_pred_and_area_under_curve(logistic_model, X_test, y_test)

    return features_count, auc_log, auc_linear, f1_dict


def is_features_count_changed(features_count: np.array) -> bool:
    val = features_count[0]
    for i, e in enumerate(features_count):
        if e != val:
            return False
    return True


def mean_confidence_interval(
    data: np.array, confidence=0.95
) -> Tuple[float, float, float]:
    mean = np.mean(data)
    se = scipy.stats.sem(data)
    h = scipy.stats.norm.ppf(confidence) * se
    return mean, mean - h, mean + h


if __name__ == "__main__":
    ################
    ################
    ## Q1
    ################
    ################
    data_path_train = "linear_regression_train.csv"
    data_path_test = "linear_regression_test.csv"
    df_train, df_test = read_train_data(data_path_train), read_test_data(data_path_test)
    print("Training Data: ", df_train.shape)
    print(df_train.head())
    print("----------------------", "\n")
    print("Testing Data: ", df_test.shape)
    print(df_test.head())
    print("----------------------", "\n")

    train_X, train_y, test_X, test_y = prepare_data(df_train, df_test)
    model = build_model(train_X, train_y)

    # Make prediction with test set
    preds = pred_func(model, test_X)
    print(preds)

    mean_square_error = MSE(test_y, preds)
    print(mean_square_error)

    # plt.plot(test_y, label='label')
    # plt.plot(preds, label='pred')
    # plt.legend()
    # plt.savefig("./images/Q1plot")

    data_path_training = "Hitters.csv"

    train_df, df2, df_train_shape = read_training_data(data_path_training)
    s, df_train_mod = data_clean(train_df)
    features, label = feature_extract(df_train_mod)
    final_features = data_preprocess(features)
    final_label = label_transform(label)

    print("Before:")
    print("Features shape: ", features.shape)
    print(features.head())
    print("--------------------------------")
    print("Label shape: ", label.shape)
    print(label.head())
    print("\n")
    print(":::::::::::::::::::::::::::::::::::::::::::::")
    print("After:")
    print("Features shape: ", final_features.shape)
    print(final_features.head())
    print("--------------------------------")
    print("Label shape: ", final_label.shape)
    print(final_label.head())

    num_of_folds = 5
    max_iter = 100000008
    X = final_features
    y = final_features
    auc_log = []
    auc_linear = []
    features_count = []
    f1_dict = {"log_reg": [], "linear_reg": []}
    count_changed = True

    X_train, X_test, y_train, y_test = data_split(final_features, final_label)

    linear_model = train_linear_regression(X_train, y_train)

    logistic_model = train_logistic_regression(X_train, y_train)

    linear_coef, logistic_coef = models_coefficients(linear_model, logistic_model)

    (
        linear_y_pred,
        linear_reg_fpr,
        linear_reg_tpr,
        linear_threshold,
        linear_reg_area_under_curve,
    ) = linear_pred_and_area_under_curve(linear_model, X_test, y_test)

    log_y_pred, log_reg_fpr, log_reg_tpr, log_threshold, log_reg_area_under_curve = (
        logistic_pred_and_area_under_curve(logistic_model, X_test, y_test)
    )

    plt.plot(log_reg_fpr, log_reg_tpr, label="logistic")
    plt.plot(linear_reg_fpr, linear_reg_tpr, label="linear")
    plt.legend()
    plt.title("Linear vs. Logistic ROC")
    plt.savefig("./images/Q3plot")

    linear_threshod, linear_threshod = optimal_thresholds(
        linear_threshold,
        linear_reg_fpr,
        linear_reg_tpr,
        log_threshold,
        log_reg_fpr,
        log_reg_tpr,
    )

    skf = stratified_k_fold_cross_validation(num_of_folds, final_features, final_label)
    features_count, auc_log, auc_linear, f1_dict = train_test_folds(
        skf, num_of_folds, final_features, final_label
    )

    print("Does features_count stay the same in each fold?")

    # call is_features_count_changed function and return true if features count changes in each fold. else return false
    count_changed = is_features_count_changed(features_count)

    linear_threshold, log_threshold = optimal_thresholds(
        linear_threshold,
        linear_reg_fpr,
        linear_reg_tpr,
        log_threshold,
        log_reg_fpr,
        log_reg_tpr,
    )
    print(count_changed)

    print("------------------------")
    # print("Linear Model Coefficients: ")
    # print(linear_coef)
    # print("\n")
    print("Optimal threshold for Lineaer Model: ", linear_threshold)
    print("------------------------")
    # print("Logistic Model Coefficients: ")
    # print(logistic_coef)
    # print("\n")
    print("Optimal threshold for Logistic Model: ", log_threshold)
    print("------------------------")

    # print(auc_linear)
    # print(auc_log)
    auc_linear_mean, auc_linear_open_interval, auc_linear_close_interval = (
        mean_confidence_interval(auc_linear)
    )
    auc_log_mean, auc_log_open_interval, auc_log_close_interval = (
        mean_confidence_interval(auc_log)
    )

    f1_linear_mean, f1_linear_open_interval, f1_linear_close_interval = (
        mean_confidence_interval(f1_dict["linear_reg"])
    )
    f1_log_mean, f1_log_open_interval, f1_log_close_interval = mean_confidence_interval(
        f1_dict["log_reg"]
    )

    mean_confidence_interval(auc_log)
    mean_confidence_interval(auc_linear)
    mean_confidence_interval(f1_dict["log_reg"])
    mean_confidence_interval(f1_dict["linear_reg"])
    print("AUC for Logistic Model: ", auc_log)
    print("AUC for Linear Model: ", auc_linear)
    print(
        "AUC Confidence Interval for Logistic Model: ",
        mean_confidence_interval(auc_log),
    )
    print(
        "AUC Confidence Interval for Linear Model: ",
        mean_confidence_interval(auc_linear),
    )
    print(
        "F1 Score Confidence Interval for Logistic Model: ",
        mean_confidence_interval(f1_dict["log_reg"]),
    )
    print(
        "F1 Score Confidence Interval for Linear Model: ",
        mean_confidence_interval(f1_dict["linear_reg"]),
    )
