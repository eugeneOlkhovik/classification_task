import os

config_dir = os.getcwd()

DATA_FOLDER = os.path.join(config_dir, 'data')
TRAIN_DATA_FILE = os.path.join(DATA_FOLDER, 'training.csv')
VALIDATION_DATA_FILE = os.path.join(DATA_FOLDER, 'validation.csv')

columns_to_transform = ['v2', 'v3', 'v5', 'v6', 'v7', 'v15']

category_variables = ['v1', 'v4', 'v8', 'v9', 'v10',
                      'v11', 'v12', 'v16', 'v17']
continuous_variables = ['v2', 'v3', 'v5', 'v6']
discrete_variables = ['v7', 'v13', 'v14', 'v15']

column_types = {
    'v1': 'category',
    'v2': 'float',
    'v3': 'float',
    'v4': 'category',
    'v5': 'float',
    'v6': 'float',
    'v7': 'float',
    'v8': 'category',
    'v9': 'category',
    'v10': 'int',
    'v11': 'category',
    'v12': 'category',
    'v13': 'float',
    'v14': 'int',
    'v15': 'float',
    'v16': 'category',
    'v17': 'int',
    'classLabel': 'category'
}

default_labels = ['yes.', 'no.']
