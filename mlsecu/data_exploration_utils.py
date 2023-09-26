import pandas as pd


def get_column_names(dataframe: pd.DataFrame):
    """Returns a list of column names from a dataframe."""
    return list(dataframe.columns.values)


def get_nb_of_dimensions(dataframe: pd.DataFrame):
    """Returns the number of dimensions of a dataframe."""
    return len(dataframe.columns.values)


def get_nb_of_rows(dataframe):
    """Returns the number of rows of a dataframe."""
    if dataframe is None:
        return None
    return len(dataframe)


def get_number_column_names(dataframe: pd.DataFrame):
    """Returns a list of column names from a dataframe that contain numbers."""
    if dataframe is None:
        return None
    return list(dataframe.select_dtypes(include=["number"]).columns.values)


def get_object_column_names(dataframe: pd.DataFrame):
    """Returns a list of column names from a dataframe that contain objects."""
    if dataframe is None:
        return None
    return list(dataframe.select_dtypes(include=["object"]).columns.values)


def get_unique_values(dataframe: pd.DataFrame, column_name: str):
    """Returns a list of unique values from a column of a dataframe."""
    if dataframe is None:
        return None
    return list(dataframe[column_name].unique())
