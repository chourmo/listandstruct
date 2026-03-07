"""Tests for the listandstruct package."""

import pandas as pd
import pyarrow as pa
import pytest

from listandstruct import list_array, struct_array


def test_struct_array_creation():
    """Test basic struct array creation from DataFrame."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = struct_array(df)

    assert isinstance(result, pd.Series)
    assert len(result) == 3
    assert isinstance(result.dtype, pd.ArrowDtype)
    assert isinstance(result.dtype.pyarrow_dtype, pa.StructType)


def test_struct_array_preserves_index():
    """Test that struct_array preserves DataFrame index."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}, index=["i", "j", "k"])
    result = struct_array(df)

    assert list(result.index) == ["i", "j", "k"]


def test_struct_array_expand():
    """Test expanding struct array back to DataFrame."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    structs = struct_array(df)
    expanded = structs.structarray.expand()

    assert isinstance(expanded, pd.DataFrame)
    assert list(expanded.columns) == ["a", "b"]
    pd.testing.assert_frame_equal(expanded, df, check_dtype=False)


def test_list_array_with_ids():
    """Test list array creation with ids."""
    values = pd.Series([1, 2, 3, 4, 5, 6])
    ids = pd.Series(["A", "A", "A", "B", "B", "B"])

    result = list_array(values, ids=ids)

    assert isinstance(result, pd.Series)
    assert len(result) == 2
    assert isinstance(result.dtype, pd.ArrowDtype)
    assert isinstance(result.dtype.pyarrow_dtype, pa.ListType)


def test_list_array_values_error():
    """Test that list_array raises error when both ids and offsets are None."""
    values = pd.Series([1, 2, 3])

    with pytest.raises(ValueError, match="ids or offsets must not be None"):
        list_array(values)


def test_list_array_flatten():
    """Test flattening list array."""
    values = pd.Series([1, 2, 3, 4])
    ids = pd.Series(["A", "A", "B", "B"])
    lists = list_array(values, ids=ids)

    flat = lists.listarray.flatten()

    assert isinstance(flat, pd.Series)
    assert len(flat) == 4
    assert list(flat.values) == [1, 2, 3, 4]


def test_list_array_aggregate_sum():
    """Test aggregate sum operation."""
    values = pd.Series([1, 2, 3, 4])
    ids = pd.Series(["A", "A", "B", "B"])
    lists = list_array(values, ids=ids)

    sums = lists.listarray.aggregate("sum")

    assert isinstance(sums, pd.Series)
    assert len(sums) == 2
    assert sums.iloc[0] == 3  # 1 + 2
    assert sums.iloc[1] == 7  # 3 + 4


def test_list_array_get_position():
    """Test getting elements at specific position."""
    values = pd.Series([10, 20, 30, 40, 50])
    ids = pd.Series(["A", "A", "A", "B", "B"])
    lists = list_array(values, ids=ids)

    first = lists.listarray.get(0)
    last = lists.listarray.get(-1)

    assert first.iloc[0] == 10
    assert first.iloc[1] == 40
    assert last.iloc[0] == 30
    assert last.iloc[1] == 50


def test_list_array_add_scalar():
    """Test adding scalar to list array."""
    values = pd.Series([1, 2, 3, 4])
    ids = pd.Series(["A", "A", "B", "B"])
    lists = list_array(values, ids=ids)

    result = lists.listarray.add(10)

    flat = result.listarray.flatten()
    assert list(flat.values) == [11, 12, 13, 14]


def test_list_array_filter():
    """Test filtering list array with boolean mask."""
    values = pd.Series([1, 2, 3, 4, 5, 6])
    ids = pd.Series(["A", "A", "A", "B", "B", "B"])
    lists = list_array(values, ids=ids)

    # Create mask: keep values >= 3
    mask = lists.listarray.is_between(start=3)

    filtered = lists.listarray.filter(mask)

    assert len(filtered) == 2
    flat = filtered.listarray.flatten()
    assert list(flat.values) == [3, 4, 5, 6]


def test_list_array_replace():
    """Test replacing values in list array."""
    values = pd.Series([1, 2, 3, 2, 1])
    ids = pd.Series(["A", "A", "B", "B", "B"])
    lists = list_array(values, ids=ids)

    result = lists.listarray.replace(2, 99)

    flat = result.listarray.flatten()
    assert list(flat.values) == [1, 99, 3, 99, 1]


def test_list_array_value_lengths():
    """Test getting value lengths of each sub-array."""
    values = pd.Series([1, 2, 3, 4, 5])
    ids = pd.Series(["A", "A", "A", "B", "B"])
    lists = list_array(values, ids=ids)

    lengths = lists.listarray.value_lengths

    assert len(lengths) == 2
    assert lengths[0] == 3
    assert lengths[1] == 2


def test_empty_dataframe_struct():
    """Test struct_array with empty DataFrame."""
    df = pd.DataFrame({"a": [], "b": []})
    result = struct_array(df)

    assert len(result) == 0
    assert isinstance(result, pd.Series)
