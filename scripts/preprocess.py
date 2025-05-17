import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def load_data(data_path="data/diabetes.csv"):
    """
    Load the dataset from the given path.
    """
    if not os.path.exists(data_path):
        # download the dataset from the url
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
        df = pd.read_csv(url)
        df.to_csv(data_path, index=False)

        return df

    df = pd.read_csv(data_path)
    return df


def clean_data(df):
    """
    Replace invalid zeros with NaN, then impute using median.
    """
    cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df


def preprocess_data(df):
    """
    Separate features and target, scale the features.
    """
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into train and test sets.
    """
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def load_and_preprocess_data(data_path="data/diabetes.csv"):
    """
    Full preprocessing pipeline: load, clean, scale, split.
    Returns: X_train, X_test, y_train, y_test
    """
    df = load_data(data_path)
    df = clean_data(df)
    X_scaled, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    return X_train, X_test, y_train, y_test
