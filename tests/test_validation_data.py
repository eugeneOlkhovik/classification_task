def test_no_nan_values(data):
    """Test if there are no NaN values in data."""
    assert not data.isnull().values.any(), "There are NaN values in the data"


def test_same_columns(train_data, validation_data):
    """Test if training and validation data have the same columns"""
    assert list(train_data.columns) == list(validation_data.columns), "Train and validation data have different columns."
