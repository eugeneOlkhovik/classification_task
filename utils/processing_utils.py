import pandas as pd
import numpy as np


def load_transformed_data(file_path, columns_to_transform, column_types):
    """Load and transform a CSV file
    by replacing commas with decimal points in the specified
    columns and converting each column to the specified data type.

    Args:
        file_path (str): path to the CSV file to be read.
        columns_to_transform (list):  list of column names in the CSV file
        where commas should be replaced with decimal points.
        column_types (dict): dictionary containing the column names
        as keys and their desired data types as values.


    Returns:
        pd.DataFrame: transformed DataFrame with
        the specified columns having the desired data types.
    """
    df = pd.read_csv(file_path, delimiter=';')

    for col in columns_to_transform:
        if isinstance(df[col][0], str):
            df[col] = df[col].str.replace(',', '.')
            # If ‘coerce’, then invalid parsing will be set as NaN.
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col_name, col_type in column_types.items():
        if col_name not in columns_to_transform:
            df[col_name] = df[col_name].astype(col_type)

    return df


def get_ratio_of_nans(df: pd.DataFrame):
    """Calculate the percentage of missing values
    of the dataframe and return a new dataframe

    Args:
        df (pd.DataFrame): Target dataframe for calculation of missing values

    Returns:
        pd.DataFrame: returned dataframe with two columns:
        1) 'Missing values' - amount of Nans in column
        2) 'Percentage' - ratio of Nans to records amount
    """
    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df)) * 100

    return pd.DataFrame(
        {'Missing Values': missing_values, 'Percentage': missing_percentages}
    )


def get_one_hot_encode(df, var_name, is_dummy=False):
    """One-hot encodes a categorical variable with
    two unique values in a pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        var_name (str): The name of the categorical variable
        to be one-hot encoded.
        is_dummy (bool, optional): Whether to create a dummy
        variable instead of one-hot encoding.
            If True, a single dummy variable will be created
            instead of a binary vector. Defaults to False.

    Returns:
        pd.DataFrame: The updated DataFrame with the categorical
        variable replaced with one-hot encoded or dummy variable.
    """
    dummy = pd.get_dummies(df[var_name], prefix=var_name, drop_first=is_dummy)
    df = pd.concat([df, dummy], axis=1)
    df.drop(var_name, axis=1, inplace=True)
    return df


def frequency_encode(df, column):
    """Encode a categorical column using frequency encoding.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to be encoded.

    Returns:
        pd.DataFrame: The encoded DataFrame.
    """
    freq = df[column].value_counts(normalize=True)
    encoding = freq.to_dict()

    df[column] = df[column].map(encoding)
    return df


def mean_target_encode(df, column, target, min_samples_leaf=1, smoothing=1):
    """Encode a categorical column using mean target encoding with smoothing.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to be encoded.
        target (str): The name of the target variable column.
        min_samples_leaf (int, optional):
        The minimum number of samples required to perform smoothing.
            Defaults to 1.
        smoothing (int, optional):
        The smoothing factor. Higher values result in more smoothing.
            Defaults to 1.

    Returns:
        pd.DataFrame: The encoded DataFrame.
    """
    global_mean = df[target].mean()
    agg = df.groupby(column)[target].agg(['count', 'mean'])
    smooth_mean = (agg['count'] * agg['mean'] +
                   min_samples_leaf * global_mean) / \
                  (agg['count'] + min_samples_leaf)

    encoding = global_mean + (smooth_mean - global_mean) / (1 + smoothing)
    df[column] = df[column].map(encoding)
    return df


def impute_categorical_by_distribution(df, column):
    """
    Function to impute missing values in a categorical column
    according to the existing distribution of classes.

    Args:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name where missing values need to be imputed.

    Returns: pd.DataFrame: The DataFrame with imputed missing values.
    """
    class_counts = df[column].value_counts()
    class_proportions = class_counts / class_counts.sum()

    classes = class_proportions.index.tolist()
    proportions = class_proportions.values.tolist()

    missing_count = df[column].isna().sum()
    imputed_values = np.random.choice(classes,
                                      size=missing_count,
                                      p=proportions)
    df.loc[df[column].isna(), column] = imputed_values
    return df


def impute_var_with_extreme(df, column):
    """
    Impute a continuous column using extreme value.

    Args:
        df (pd.DataFrame): The input DataFrame
        column (str): The name of the column to be imputed

    Returns:
        pd.DataFrame: The DataFrame with the specified column
        imputed using the median
    """
    df[column] = df[column].fillna(max(df[column]) * 5)
    return df


def drop_columns(df, cols_to_drop):
    """
    Impute a continuous column using median imputation

    Args:
        df (pd.DataFrame): DataFrame to modify
        cols_to_drop (list): list of column names to drop

    Returns:
        pd.DataFrame: DataFrame with specified columns dropped
    """
    return df.drop(cols_to_drop, axis=1)
