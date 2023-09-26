import pandas as pd


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
