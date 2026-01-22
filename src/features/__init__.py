"""Feature engineering module."""

from .feature_definitions import (
    IDENTIFIER_COLS,
    TARGET_COL,
    NUMERIC_COLS,
    BINARY_COLS,
    CATEGORICAL_COLS,
    FEATURE_COLS,
    get_feature_columns,
    get_column_groups
)

from .preprocessing import (
    create_preprocessor,
    create_numeric_pipeline,
    create_binary_pipeline,
    create_categorical_pipeline,
    get_feature_names,
    preprocess_data
)

__all__ = [
    # Feature definitions
    'IDENTIFIER_COLS',
    'TARGET_COL',
    'NUMERIC_COLS',
    'BINARY_COLS',
    'CATEGORICAL_COLS',
    'FEATURE_COLS',
    'get_feature_columns',
    'get_column_groups',
    # Preprocessing
    'create_preprocessor',
    'create_numeric_pipeline',
    'create_binary_pipeline',
    'create_categorical_pipeline',
    'get_feature_names',
    'preprocess_data'
]
