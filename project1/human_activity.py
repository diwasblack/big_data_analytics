import statistics

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_validate
from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestClassifier


def visualize_data(X, Y):
    scatter_plot_dict = {
        "STANDING": {
            "c": "r",
            "marker": "x"
        },
        "SITTING": {
            "c": "g",
            "marker": "o"
        },
        "LAYING": {
            "c": "b",
            "marker": "^"
        },
        "WALKING": {
            "c": "c",
            "marker": "s"
        },
        "WALKING_DOWNSTAIRS": {
            "c": "m",
            "marker": "+"
        },
        "WALKING_UPSTAIRS": {
            "c": "y",
            "marker": "D"
        }
    }

    points_dict = {
        "STANDING": [],
        "SITTING": [],
        "LAYING": [],
        "WALKING": [],
        "WALKING_DOWNSTAIRS": [],
        "WALKING_UPSTAIRS": []
    }

    # Construct the TSNE object for dimensionality reduction
    reducer = TSNE()

    X_reduced = reducer.fit_transform(X)

    for index, value in enumerate(X_reduced):
        points_dict[Y[index]].append(value)

    fig, ax = plt.subplots(figsize=(20, 10))
    for key in points_dict:
        ax.scatter(*list(zip(*points_dict[key])), label=key, **scatter_plot_dict[key])

    ax.legend()
    plt.savefig("dataset.png")


def train_test_logistic_regression(x_train, x_test, y_train, y_test):
    # The hyperparameters to check
    c_values = [0.1, 0.5, 1, 10, 100]

    best_c = c_values[0]
    best_score = 0

    for c_value in c_values:
        clf = linear_model.LogisticRegression(C=c_value)
        cv_results = cross_validate(
            clf, x_train, y_train, cv=5, return_train_score=False, scoring=("accuracy"), n_jobs=-1)

        scores = cv_results["test_score"]
        score = statistics.mean(scores)

        if(score > best_score):
            best_score = score
            best_c = c_value

    print(f"Best C: {best_c}")

    clf = linear_model.LogisticRegression(C=c_value)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(accuracy_score(y_test, y_pred))


def train_test_svm(x_train, x_test, y_train, y_test):
    # The hyperparameters to check
    c_values = [0.1, 0.5, 1, 10, 100]

    best_c = c_values[0]
    best_score = 0

    for c_value in c_values:
        clf = svm.SVC(C=c_value)
        cv_results = cross_validate(
            clf, x_train, y_train, cv=5, return_train_score=False, scoring=("accuracy"), n_jobs=-1)

        scores = cv_results["test_score"]
        score = statistics.mean(scores)

        if(score > best_score):
            best_score = score
            best_c = c_value

    print(f"Best C: {best_c}")

    clf = svm.SVC(best_c)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print(accuracy_score(y_test, y_pred))


def train_test_random_forest(x_train, x_test, y_train, y_test):
    parameters_values = [
        {
            "n_estimators": 12
        },
        {
            "n_estimators": 10
        },
        {
            "n_estimators": 24
        }
    ]

    best_parameters = parameters_values[0]
    best_score = 0

    for parameters in parameters_values:
        clf = RandomForestClassifier(**parameters)

        cv_results = cross_validate(
            clf, x_train, y_train, cv=5, return_train_score=False, scoring=("accuracy"), n_jobs=-1)
        scores = cv_results["test_score"]
        score = statistics.mean(scores)

        if(score > best_score):
            best_score = score
            best_parameters = parameters

    print(best_parameters)

    clf = RandomForestClassifier(**best_parameters)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    print(accuracy_score(y_test, y_pred))


def main():
    # Load dataset
    df = pd.read_csv("human_activity.csv")

    # Delete index column
    df.drop("rn", axis=1, inplace=True)

    label_df = df["activity"]
    features_df = df.drop("activity", axis=1)

    # Convert features dataframe to numpy array
    X = features_df.values
    Y = label_df.values

    # Split data into train test
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

    # Visualize data
    visualize_data(X, Y)

    train_test_logistic_regression(x_train, x_test, y_train, y_test)
    train_test_svm(x_train, x_test, y_train, y_test)
    train_test_random_forest(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
