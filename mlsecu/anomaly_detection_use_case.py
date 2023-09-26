import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import sys


def get_one_hot_encoded_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Apllique du one-hot encoding sur un dataframe.

    Arguments:
        dataframe  {pd.DataFrame} -- La dataframe à encoder.

    Returns:
        pd.DataFrame -- La dataframe encodée.
    """
    if dataframe is None:
        return None
    one_hot_encoded_dataframe = pd.get_dummies(dataframe)
    return one_hot_encoded_dataframe


def remove_nan_through_mean_imputation(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne: la dataframe avec les entrées NaN (Not a Number) remplacées par imputation de la moyenne
    Arguments:
        dataframe {pd.DataFrame} -- La dataframe à traiter
    Returns:
        pd.DataFrame -- La dataframe avec les entrées NaN remplacées par imputation de la moyenne
    """
    if dataframe is None:
        return None
    return dataframe.fillna(dataframe.mean())


def get_list_of_attack_types(dataframe: pd.DataFrame):
    f = ""
    for col in dataframe.columns:
        f += col + " "

    return list(dataframe[dataframe[f] == -1].index)


def get_nb_of_attack_types(dataframe):
    return len(get_list_of_attack_types(dataframe))


def get_list_of_if_outliers(dataframe: pd.DataFrame, outlier_fraction: float) -> list:
    """Extract the list of outliers according to Isolation Forest algorithm"""

    if dataframe is None:
        return None
    # get list of outliers with sklearn
    clf = IsolationForest(contamination=outlier_fraction, random_state=42)
    one_hot_df = get_one_hot_encoded_dataframe(dataframe)
    one_hot_df = remove_nan_through_mean_imputation(one_hot_df)
    clf.fit(one_hot_df)
    res = clf.predict(one_hot_df)

    return list(one_hot_df.index[res == -1])


def get_list_of_lof_outliers(dataframe: pd.DataFrame, outlier_fraction: float) -> list:
    """Extract the list of outliers according to Local Outlier Factor algorithm."""

    if dataframe is None:
        return None
    one_hot_df = get_one_hot_encoded_dataframe(dataframe)
    one_hot_df = remove_nan_through_mean_imputation(one_hot_df)
    clf = LocalOutlierFactor(n_neighbors=20, contamination=outlier_fraction)
    res = clf.fit_predict(one_hot_df)
    return list(one_hot_df.index[res == -1])


def get_nb_of_if_outliers(dataframe: pd.DataFrame, outlier_fraction: float) -> int:
    """Extract the number of outliers according to Isolation Forest algorithm."""
    if dataframe is None:
        return None

    return len(get_list_of_if_outliers(dataframe, outlier_fraction))


def get_nb_of_lof_outliers(dataframe: pd.DataFrame, outlier_fraction: float) -> int:
    """Extract the number of outliers according to Local Outlier Factor algorithm.
    Arguments:
        dataframe {pd.DataFrame} -- The dataframe to process
        outlier_fraction {float} -- The fraction of outliers to detect
    Returns:
        int -- The number of outliers
    """
    if dataframe is None:
        return None
    return len(get_list_of_lof_outliers(dataframe, outlier_fraction))


def get_nb_of_occurrences(dataframe: pd.DataFrame) -> int:
    """Retrieves the number of occurrences of a pandas dataframe."""
    if dataframe is None:
        return None
    nb_occurence = len(dataframe.index)
    return nb_occurence


def get_list_of_parameters(dataframe):
    """Retrieves the list of parameters of a pandas dataframe"""
    if dataframe is None:
        return None
    return list(dataframe.columns)


def get_nb_of_parameters(dataframe):
    if dataframe is None:
        return None
    return len(dataframe.columns)
