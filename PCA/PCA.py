#!/usr/bin/env python
# coding: utf-8

print("Acknowledgment:")
print("https://github.com/pritishuplavikar/Face-Recognition-on-Yale-Face-Dataset")

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import glob
from numpy import linalg as la
import random
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import os
import copy

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import sklearn
from typing import Tuple, List
from typeguard import typechecked


@typechecked
def qa1_load(folder_path: str) -> Tuple[np.ndarray, np.ndarray]:
    os.chdir(folder_path)
    x = []
    y = []
    for file in os.listdir():
        if file.startswith("s"):
            x.append(mpimg.imread(file))
            y.append(int(file[7] + file[8]))
    x = np.stack(x, axis=0)
    x = x.reshape(165, -1)
    os.chdir("..")
    return x, np.array(y)


@typechecked
def qa2_preprocess(dataset: np.ndarray) -> np.ndarray:
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(dataset)


@typechecked
def qa3_calc_eig_val_vec(
    dataset: np.ndarray, k: int
) -> Tuple[PCA, np.ndarray, np.ndarray]:
    pca = PCA()
    # pca.fit(dataset.reshape(165, -1))
    pca.fit(dataset)
    return pca, pca.explained_variance_[:k], pca.components_[:k]


def qb_plot_written(eig_values: np.ndarray):
    plt.plot(range(len(eig_values)), eig_values)
    plt.title("Eigen Values")
    plt.savefig("./images/Qbplot")


@typechecked
def qc1_reshape_images(pca: PCA, dim_x=243, dim_y=320) -> np.ndarray:
    orig = np.reshape(pca.components_, (-1, dim_x, dim_y))
    return orig


def qc2_plot(org_dim_eig_faces: np.ndarray):
    rints = np.random.randint(0, 165, size=10)
    for i in range(len(rints)):
        plt.imsave("./test/Qcplot" + str(i + 1) + ".png", org_dim_eig_faces[rints[i]])


@typechecked
def qd1_project(dataset: np.ndarray, pca: PCA) -> np.ndarray:
    return pca.transform(dataset.reshape(165, -1))


@typechecked
def qd2_reconstruct(projected_input: np.ndarray, pca: PCA) -> np.ndarray:
    return pca.inverse_transform(projected_input)


def qd3_visualize(dataset: np.ndarray, pca: PCA, dim_x=243, dim_y=320):
    comp = copy.deepcopy(pca.components_)
    rints = np.random.randint(0, 165, size=3)
    k = [1, 10, 20, 30, 40, 50]
    count = 0
    for j in rints:
        for i in k:
            pca.components_ = comp[:i]
            projected_input = qd1_project(dataset, pca)
            img = qd2_reconstruct(projected_input, pca)
            image = np.reshape(img, (-1, dim_x, dim_y))
            image = image[j]
            plt.imsave("./test/Qdplot" + str(count) + ".png", image)
            count += 1


@typechecked
def qe1_svm(trainX: np.ndarray, trainY: np.ndarray, pca: PCA) -> Tuple[int, float]:
    accuracy = []
    k = [10, 30, 50, 70, 90]
    comp = copy.deepcopy(pca.components_)
    for i in k:
        scores = []
        pca.components_ = comp[:i]
        projection = qd1_project(trainX, pca)
        skf = StratifiedKFold(5)
        for _, (train_index, test_index) in enumerate(skf.split(projection, trainY)):
            x_train, y_train = projection[train_index], trainY[train_index]
            x_test, y_test = projection[test_index], trainY[test_index]
            svm = SVC()
            svm = svm.fit(x_train, y_train)
            scores.append(svm.score(x_test, y_test))
        accuracy.append(np.array(scores).mean())
    accuracy = np.array(accuracy)
    maxdex = np.argmax(accuracy)
    return k[maxdex], accuracy[maxdex]


@typechecked
def qe2_lasso(trainX: np.ndarray, trainY: np.ndarray, pca: PCA) -> Tuple[int, float]:
    maxacc = 0
    best_k = 0
    best_a = 0
    accuracy = []
    k = [10, 30, 50, 70, 90]
    alphas = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    comp = copy.deepcopy(pca.components_)
    best_alphas = {}
    for a in alphas:
        accuracy = []
        for i in k:
            scores = []
            pca.components_ = comp[:i]
            projection = qd1_project(trainX, pca)
            skf = StratifiedKFold(5)
            for _, (train_index, test_index) in enumerate(
                skf.split(projection, trainY)
            ):
                x_train, y_train = projection[train_index], trainY[train_index]
                x_test, y_test = projection[test_index], trainY[test_index]
                lasso = Lasso(a, max_iter=1000000)
                lasso = lasso.fit(x_train, y_train)
                # pred = (lasso.predict(x_test)).round().astype(int)
                # scores.append(metrics.mean_squared_error(y_test, pred))
                scores.append(lasso.score(x_test, y_test))
            accuracy.append(np.array(scores).mean())
        accuracy = np.array(accuracy)
        maxdex = np.argmax(accuracy)
        best_alphas[a] = (k[maxdex], accuracy[maxdex])
    print(best_alphas)
    for i in alphas:
        if best_alphas[i][1] > maxacc:
            maxacc = best_alphas[i][1]
            best_k = best_alphas[i][0]
            best_a = i
    return best_k, maxacc


if __name__ == "__main__":

    faces, y_target = qa1_load("./data")
    dataset = qa2_preprocess(faces)
    pca, eig_values, eig_vectors = qa3_calc_eig_val_vec(dataset, len(dataset))

    qb_plot_written(eig_values)
    print(np.argmin(abs(pca.explained_variance_ratio_.cumsum() - 0.5)) + 1)
    num = len(dataset)
    org_dim_eig_faces = qc1_reshape_images(pca)
    qc2_plot(org_dim_eig_faces)

    qd3_visualize(dataset, pca)

    best_k, result = qe1_svm(dataset, y_target, pca)
    print(best_k, result)
    best_k, result = qe2_lasso(dataset, y_target, pca)
    print(best_k, result)
