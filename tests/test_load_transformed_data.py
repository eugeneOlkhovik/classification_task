import io
from utils import load_transformed_data
import config


def test_transformed_record():
    """test if `load_transformed_data()`
    correctly transforms a CSV file by replacing commas and converting
    with decimal points in the specified columns and
    converting each column to the specified data type.

    This test function checks that the transformed value of 'v15'
    in the test data is equal to 800000.

    Raises:
        AssertionError: If the transformed value of 'v15'
        in the test data is not equal to 800000.
    """
    test_data = '''v1;v2;v3;v4;v5;v6;v7;v8;v9;v10;v11;v12;v13;v14;v15;v16;v17;classLabel
                   b;26,75;2,00E-04;u;1,025380832;0,440071835;0,75;f;f;0;t;g;80;0;8,00E+05;f;1;no.
                   a;16,33;2,10E-05;u;2,936522146;0,78376394;0,125;f;f;0;f;g;200;1;2,00E+06;f;1;no.'''

    test_df = load_transformed_data(io.StringIO(test_data),
                                    config.columns_to_transform,
                                    config.column_types)

    assert test_df.loc[0, 'v15'] == 800000
    assert test_df.loc[1, 'v15'] == 2000000
