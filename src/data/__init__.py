"""Data loading and validation module."""

from .loader import (
    load_raw_data,
    clean_total_charges,
    encode_target,
    validate_no_leakage,
    prepare_data
)

from .splitting import (
    create_splits,
    get_split_summary,
    print_split_summary
)

__all__ = [
    'load_raw_data',
    'clean_total_charges',
    'encode_target',
    'validate_no_leakage',
    'prepare_data',
    'create_splits',
    'get_split_summary',
    'print_split_summary'
]
