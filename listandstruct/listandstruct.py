"""listandstruct package."""

from collections.abc import Callable  # noqa: TC003
from functools import cached_property
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc


def struct_array(df: pd.DataFrame) -> pa.Array:
    """
    Convert a pandas DataFrame to a PyArrow StructArray wrapped in a pandas Series.

    This function takes a DataFrame and transforms it into a StructArray, where each row
    becomes a struct containing all columns as fields. The result is returned as a pandas
    Series with PyArrow struct dtype.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to be converted to a StructArray.

    Returns
    -------
    pd.Series
        A pandas Series containing PyArrow StructArray elements, where each element
        is a struct with fields corresponding to the DataFrame columns. The Series
        maintains the original DataFrame's index and has ArrowDtype of the struct type.

    Notes
    -----
    The function preserves the DataFrame index in the returned Series.
    Column dtypes are converted to their PyArrow equivalents automatically.
    """
    """Convert dataframe to a StructArray"""
    names = list(df.columns)
    values = [pa.array(df[col], from_pandas=True) for col in names]
    stype = pa.struct({col: df[col].dtype.pyarrow_dtype for col in df.columns})
    structarray = pa.StructArray.from_arrays(arrays=values, type=stype)
    result = pd.Series(structarray, index=df.index, dtype=pd.ArrowDtype(stype))
    return result


@pd.api.extensions.register_series_accessor("structarray")
class ArrowStructArray:
    """ExtensionArray for Arrow StructArray."""

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        """Validate obj is a series with pyrrow.list_ data."""
        if not isinstance(obj, pd.Series):
            raise AttributeError("Data must be a pd.Series")
        if not isinstance(obj.dtype.pyarrow_dtype, pa.StructType):
            raise AttributeError("Data arrow type must be StructArray")

    def expand(self) -> pd.DataFrame:
        """Convert a Series of StructArray to a DataFrame."""
        expanded = self._obj.explode()
        index = expanded.index
        struct_array = pa.Table.from_struct_array(pa.array(expanded.values))
        struct_df = struct_array.to_pandas(types_mapper=pd.ArrowDtype)
        struct_df.index = index

        return struct_df


# -----------------------------------------------------------------
# ListArray creation and pandas Extension


def list_array(
    df: pd.Series | pa.Array,
    ids: Optional[pd.Series | pa.Array] = None,
    offsets: Optional[pd.Series | pa.Array] = None,
    ignore_index: bool = True,
) -> pd.Series:
    """Create a ListArray Series.

    Arguments:
        df : a pandas Series to group in a ListArray
        ids : optional sorted Series, aligned to df, of unique values to group df
        offsets : optional offsets of Series of ListArray, as defined by pyarrow,
            used if ids is also set
        ignore_index : if True, use unique ids as index or unique offsets index

    Return:
        a ListArray Series with same type as df
    """
    if ids is None and offsets is None:
        raise ValueError("Reference or offsets must not be None")

    if isinstance(df, pd.Series):
        array = pa.array(df, from_pandas=True)
    elif isinstance(df, pa.ChunkedArray):
        array = df.combine_chunks()
    elif isinstance(df, pa.Array):
        array = df
    else:
        raise ValueError(f"{df} must be an array or a Series ")

    dtype = pd.ArrowDtype(pa.list_(array.type))

    if offsets is None and ids is not None:
        offs = _offsets(ids)
    elif offsets is not None and _is_list_series(offsets):
        offs = pa.array(offsets, from_pandas=True).offsets
    elif offsets is not None and isinstance(offsets, pa.Array):
        offs = offsets
    else:
        raise ValueError(f"{offsets} must be an array or a Series ")

    array = pa.ListArray.from_arrays(values=array, offsets=offs)

    if not ignore_index:
        if offsets is not None:
            index = offs.index.drop_duplicates(keep="first")
        elif ids is not None:
            index = ids.drop_duplicates(keep="first")
        else:
            raise ValueError("Nither ids and offsets can be noth None")
        return pd.Series(data=array, dtype=dtype, index=index.values)

    return pd.Series(data=array, dtype=dtype)


##----------------------------------------------------------------------------------------
## reusable functions


def _offsets(df: pd.Series | pd.DataFrame) -> pa.Array:
    """Return a pyarrow array for offsets where values in df change."""
    df2 = df.reset_index(drop=True)
    mask = df2 != df2.shift(1)
    if isinstance(df, pd.DataFrame):
        mask = mask.fillna(True)
        # flatten to one column mask
        flat_mask = pa.array(mask[df.columns[0]], from_pandas=True)
        for c in df.columns[1:]:
            flat_mask = pc.or_(flat_mask, pa.array(mask[c], from_pandas=True))

        offsets = df2.loc[pd.Series(flat_mask, index=df2.index)]
    else:
        offsets = df2.loc[mask]

    offsets = pa.array(offsets.index.astype("int64[pyarrow]").values)
    offsets = pa.concat_arrays([pa.array([0]), offsets, pa.array([len(df)])])
    return pc.unique(offsets)


def _is_list_series(df: pd.Series) -> bool:
    """Test is df is a Series of ListArrays."""
    if isinstance(df, pd.Series) and isinstance(df.dtype, pd.ArrowDtype):
        return isinstance(df.dtype.pyarrow_dtype, pa.ListType)
    else:
        return False


def _fill_nulls(array: pa.Array, dummy=None) -> pa.Array:
    """Return an array with a dummy values instead of None, and the dummy value."""
    # CANNOT USE PYARROW FLATTEN AS IT DROPS NULLS

    if not bool(pc.any(pc.is_null(array))):
        return array, None

    if not isinstance(array, pa.ListArray):
        if dummy is None:
            dummy = int(pc.max(array)) + 1
        return pc.fill_null(array, dummy), dummy

    if dummy is None:
        dummy = int(pc.max(array.flatten())) + 1
    dummies = pa.array([[dummy]] * len(array), type=array.type)
    return pc.fill_null(array, dummies), dummy


def _flatten(array: pa.Array) -> pa.Array:
    """Return a flatten list array with top level nulls, as pyarrow silently drops them.

    If array is None, use self.array.
    """
    arr2, dummy = _fill_nulls(array)

    arr2 = arr2.flatten()
    if dummy is None:
        return arr2
    arr2 = pc.fill_null(arr2, dummy)

    # replacing values by na seems impossible in pyarrow, use pandas
    df = arr2.to_pandas(types_mapper=pd.ArrowDtype)
    df.loc[df == dummy] = pd.NA
    return pa.array(df, from_pandas=True)


def _array_indices(array: pa.Array) -> pa.Array:
    """Return indices of an array."""
    arr2, _ = _fill_nulls(array)
    return pc.list_parent_indices(arr2)


def _inner_indices(array, as_list=False):
    """Return an arrayr or a listarray with values from 0 to length of each array."""
    final_size = len(array.value_parent_indices())
    value_lengths = array.value_lengths().cast("int64")
    sub_lengths = pc.cumulative_sum(value_lengths)[:-1]
    sub_lengths = pa.concat_arrays([pa.array([0], type=sub_lengths.type), sub_lengths])
    shift = _align_to_lengths(sub_lengths, value_lengths)
    indices = pa.arange(0, final_size)
    indices = pc.subtract(indices, shift)

    if not as_list:
        return indices

    # offsets with nulls
    offsets = _fill_nulls(array)[0].offsets
    return pa.ListArray.from_arrays(values=indices, offsets=offsets)


def _similar_arrays(arr1: pa.Array, arr2: pa.Array) -> pa.Array:
    """Test if arrays have the same length for each array."""
    if type(arr1) is not type(arr2):
        raise ValueError("Arrays must have same type")

    return pc.equal(_array_indices(arr1), _array_indices(arr2))


def _equal(arr1: pa.Array, arr2: pa.Array) -> pa.Array:
    """Test equality of arrays with nulls, if null on both side returns True, else False."""
    if isinstance(arr1, pa.ListArray):
        max1 = pc.max(arr1.flatten())
    else:
        max1 = pc.max(arr1)
    if isinstance(arr2, pa.ListArray):
        max2 = pc.max(arr2.flatten())
    else:
        max2 = pc.max(arr2)
    dummy = max(int(max1), int(max2)) + 1
    arr1_filled, _ = _fill_nulls(arr1, dummy)
    arr2_filled, _ = _fill_nulls(arr2, dummy)

    return pc.equal(arr1_filled, arr2_filled)


def _align(array: pa.Array, target: pa.Array) -> pa.Array:
    """Return array values repeated to match the lenghts of a ListArray."""
    if len(target) != len(array):
        raise ValueError("Arrays must have the same length")

    # list_array may contain Na, value_lengths considers them as null,
    # replace by 1 as they should be compared

    lengths = pc.fill_null(target.value_lengths(), 1)
    return _align_to_lengths(array, lengths)


def _align_to_lengths(array: pa.Array, lengths: pa.Array) -> pa.Array:
    """Return array values repeated to match the lenghts array."""
    np1 = array.to_numpy()
    np2 = lengths.to_numpy()
    return pa.array(np.repeat(np1, np2))


def _overflow_mask(lengths: pa.Array, position: int) -> pa.Array:
    if isinstance(position, pd.Series):
        pos = position.values
    else:
        pos = position
    negative = pc.and_(pc.less(pos, 0), pc.less_equal(pos, pc.negate(lengths)))
    positive = pc.and_(pc.greater_equal(pos, 0), pc.greater_equal(pos, lengths))
    return pc.or_(negative, positive)


def _get_at(array: pa.Array, position: int) -> pa.Array:
    lengths = array.value_lengths()
    lengths = pc.fill_null(lengths, 1)

    cum_lengths = pc.cumulative_sum(lengths)

    # add 0 for start of array
    cum_lengths = pa.concat_arrays([pa.array([0], type=lengths.type), cum_lengths])

    # add to indices and take from expanded df array
    if isinstance(position, pd.Series):
        df = position.copy()
        df.loc[df < 0] = df + lengths
        new_pos = pa.array(df, from_pandas=True)

    elif isinstance(position, pa.Array):
        negative_pos = pc.add(lengths, position)
        new_pos = pc.if_else(pc.less(position, 0), negative_pos, position)

    elif position >= 0:
        new_pos = pa.array([position] * len(lengths), type=lengths.type)
    else:
        new_pos = pc.add(lengths, position)

    overflow_mask = _overflow_mask(lengths, new_pos)

    # force position between bounds

    zeros = pa.array([0] * len(new_pos), type=new_pos.type)
    new_pos = pc.if_else(pc.less(new_pos, 0), zeros, new_pos)
    new_pos = pc.if_else(pc.less(new_pos, lengths), new_pos, pc.subtract(lengths, 1))
    new_pos = pc.add(cum_lengths[0:-1], new_pos)

    array = _flatten(array)
    array = array.take(new_pos)

    return pc.if_else(overflow_mask, pa.nulls(len(array), type=array.type), array)


##----------------------------------------------------------------------------------------


@pd.api.extensions.register_series_accessor("listarray")
class ArrowListArray:
    """ExtensionArray for Arrow ListArray."""

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    def _validate(self, obj):
        """Validate obj is a series with pyrrow.list_ data."""
        if not isinstance(obj, pd.Series):
            raise AttributeError("Data must be a pd.Series")
        if not _is_list_series(obj):
            raise AttributeError("Data arrow type must be ListArray")

    @cached_property
    def array(self):
        """Pyarrow Array of pandas Series."""
        return pa.array(self._obj, from_pandas=True)

    @cached_property
    def flat(self):
        """Flatten pyarrow Array of pandas Series, with nulls."""
        return _flatten(self.array)

    @property
    def type(self):
        """Pandas dtype."""
        return self._obj.dtype

    @property
    def arrow_type(self):
        """Pyarrow List Type."""
        return self.type.pyarrow_dtype

    @property
    def scalar_arrow_type(self):
        """Pyarrow scalar type contained in the list."""
        return self.arrow_type.value_type

    @cached_property
    def offsets(self):
        """Pyarrow offset array."""
        return self.array.offsets

    @cached_property
    def index(self):
        """Series index."""
        return self._obj.index

    @cached_property
    def array_indices(self):
        """Indices of list arrays."""
        return _array_indices(self.array)

    @cached_property
    def flat_inner_indices(self):
        """Flattened indices of list arrays."""
        return _inner_indices(self.array, as_list=False)

    @cached_property
    def value_lengths(self):
        """Value lenghts of flattened list array."""
        return self.array.value_lengths()

    @cached_property
    def length(self):
        """Lenghts of list array."""
        return len(self._obj)

    def validate_other(self, other, length=False, scalar=False):
        """Validate that other is a valid array or series for operations.

        Parameters
        ----------
        other : pd.Series, pa.ListArray, int, or float
            The object to validate.
        length : bool, default False
            If True, check that each array has the same length as self.
        scalar : bool, default False
            If True, allow scalar values (int or float).

        Raises
        ------
        ValueError
            If other is not a Series of ListArrays or a ListArray (when scalar=False).
        """
        if not scalar and not (isinstance(other, (pd.Series, pa.ListArray))):
            raise ValueError("Other must be a Series of ListArrays or a ListArray")

        if scalar and isinstance(other, (int, float)):
            return None

        if isinstance(other, pd.Series) and _is_list_series(other):
            if not length:
                return None
            _length = pa.array(other, from_pandas=True).value_lengths()
            if self.value_lengths == _length:
                return None
            else:
                raise ValueError("Each array length does not match")
        elif isinstance(other, pd.Series):
            return None

        if isinstance(other, pa.ListArray):
            if not length:
                return None
            _length = other.value_lengths()
            if self.value_lengths == _length:
                return None
            else:
                raise ValueError("Each array length does not match")

        raise ValueError("Other must be a scalar, a Series of ListArrays or a ListArray")

    # ------------------------------------------------------------------------------------------
    # decorators to pack results

    @staticmethod
    def flat_results(func: Callable) -> pd.Series:
        """Decorator to pack ListArray results in a Series with self._obj index."""  # noqa: D401

        def wrapper(self, *arg, **kw):
            array = func(self, *arg, **kw)
            if array is None:
                return None
            df = array.to_pandas(types_mapper=pd.ArrowDtype)
            df.index = self.index
            return df

        return wrapper

    @staticmethod
    def pack_results(func: Callable) -> pd.Series:
        """Decorator to pack results in a Series with same offsets and index as self._obj."""  # noqa: D401

        def wrapper(self, *arg, **kw):
            array = func(self, *arg, **kw)
            if array is None:
                return None
            listarray = pa.ListArray.from_arrays(values=array, offsets=self.offsets)
            df = listarray.to_pandas(types_mapper=pd.ArrowDtype)
            df.index = self.index
            return df

        return wrapper

    @staticmethod
    def expand_results(func: Callable) -> pd.Series:
        """Decorator to pack results in a flat Series with duplicated self index."""  # noqa: D401

        def wrapper(self, *arg, **kw):
            array = func(self, *arg, **kw)
            if array is None:
                return None
            df = array.to_pandas(types_mapper=pd.ArrowDtype)
            df.index = pd.Series(self.index).repeat(pc.list_value_length(self.array))
            return df

        return wrapper

    # ------------------------------------------------------------------------------------------
    # base data functions

    @expand_results
    def flatten(self) -> pa.Array:
        """Flatten to a scalar array, keeping null values (pyarrow drops them)."""
        return self.flat

    @flat_results
    def inner_indices(self, as_list=False):
        """Return an Array of inner indices."""
        if as_list:
            return _inner_indices(self.array, as_list=True)
        return self.flat_inner_indices

    @pack_results
    def equal_offsets(self, other: pa.Array) -> pa.Array:
        """Compare offsets of ListArray Series."""
        self.validate_other(other, length=False, scalar=False)

        array = pa.array(other, from_pandas=True)
        return pc.equal(self.array_indices, _array_indices(array))

    @pack_results
    def equal(self, other: pa.Array) -> pa.Array:
        """Return a boolean comparing each sub array with other dataframe."""
        self.validate_other(other, length=False, scalar=False)

        other_array = pa.array(other, from_pandas=True)

        _same_indices = _similar_arrays(self.array, other_array)
        _same_values = _equal(self.flat, _flatten(other_array))

        return pc.and_(_same_indices, _same_values)

    @pack_results
    def isna(self):
        """Return a ListArray mask where values are not null."""
        return pc.is_null(self.flat)

    # ------------------------------------------------------------------------------------------
    # math functions

    def aggregate(self, aggregator: str | list[str]):
        """Aggregate arrays with aggreagtor function or list of functions."""
        # create a table to use pyarrow aggregation functions
        table = pa.table([self.array_indices, self.flat], names=["keys", "values"])

        if isinstance(aggregator, str):
            aggs = [("values", aggregator)]
        elif isinstance(aggregator, list) and all([isinstance(x, str) for x in aggregator]):  # noqa: C419
            aggs = [("values", x) for x in aggregator]
        else:
            raise ValueError("aggregator must be a string or a list of strings")

        grped = table.group_by("keys", use_threads=True).aggregate(aggs)
        grped = grped.sort_by("keys")  # group_by returns rows in random order

        df = grped.remove_column(0).to_pandas(types_mapper=pd.ArrowDtype)
        df.index = self.index

        if isinstance(aggregator, list):
            return df.rename(columns={"values_" + x: x for x in aggregator})
        return df["values_" + aggregator]

    # ------------------------------------------------------------------------------------------
    # math or compare with other array or scalar functions

    def _combine(self, other, func: Callable) -> pa.Array:
        """Combine array with other Series, array or scalar based on func pyarrow function."""
        self.validate_other(other, length=True, scalar=True)

        if isinstance(other, (int, float)):
            other_array = other
        elif isinstance(other, pa.ChunkedArray):
            other_array = _flatten(other.combine_chunks())
        elif isinstance(other, pa.ListArray):
            other_array = _flatten(other)
        elif isinstance(other, pa.Array):
            other_array = pa.array(other)
        elif _is_list_series(other):
            other_array = _flatten(pa.array(other, from_pandas=True))
        elif isinstance(other, pd.Series):
            other_array = _align(pa.array(other, from_pandas=True), self.array)
        else:
            raise ValueError("Other type cannot be used")

        return func(self.flat, other_array)

    @pack_results
    def add(self, other):
        """Add other Array, Series or scalar value."""
        return self._combine(other, pc.add)

    @pack_results
    def subtract(self, other):
        """Subtract other Array, Series or scalar value."""
        return self._combine(other, pc.subtract)

    @pack_results
    def multiply(self, other):
        """Multiply other Array, Series or scalar value."""
        return self._combine(other, pc.multiply)

    @pack_results
    def divide(self, other):
        """Divide other Array, Series or scalar value."""
        return self._combine(other, pc.divide)

    @pack_results
    def and_(self, other):
        """Boolean and with other Array, Series or scalar value."""
        return self._combine(other, pc.and_)

    @pack_results
    def or_(self, other):
        """Boolean or with other Array, Series or scalar value."""
        return self._combine(other, pc.or_)

    @pack_results
    def xor(self, other):
        """Boolean xor with other Array, Series or scalar value."""
        return self._combine(other, pc.xor)

    @flat_results
    def contains(self, other):
        """Test if array contains other, as a scalar, an array or Series with same size."""
        self.validate_other(other, length=True, scalar=True)

        if isinstance(other, (int, float)):
            array = self.flat
            mask = _equal(array, other)

        elif isinstance(other, pa.Array | pd.Series):
            other_array = _align(pa.array(other, from_pandas=True), self.array)

            # replace null by [Null] to avoid dropping during flatten
            array = self.flat
            mask = self._equal(array, other_array)
        else:
            raise ValueError("Other must be a scalar, a Series or an array")

        results = pa.table([self.array_indices, mask], names=["keys", "values"])
        return results.group_by("keys").aggregate([("values", "any")])["values_any"]

    @flat_results
    def intersects(self, other):
        """Compare values in self and other, both arrays must be sorted."""
        self.validate_other(other, length=True, scalar=False)

        array = pa.array(self, from_pandas=True)
        if isinstance(other, pa.Array | pd.Series):
            other_array = pa.array(other, from_pandas=True)
        else:
            raise ValueError("Other must be a Series or an array")

        mask1 = pc.less(_get_at(other_array, 0), _get_at(array, -1))
        mask2 = pc.less(_get_at(array, 0), _get_at(other_array, -1))

        return pc.or_(mask1, mask2)

    # ------------------------------------------------------------------------------------------
    # query listarray

    @flat_results
    def get(self, position):
        """
        Values at position Array, Series or integer, if negative, return value from end.

        Return NA if position is higher than array length.
        """
        if not isinstance(position, (int, pa.Array, pd.Series)):
            raise ValueError("Position must be an integer, an array or a Series")
        return _get_at(self.array, position)

    @pack_results
    def is_between(self, start=None, end=None, inclusive="both"):
        """Test if values are beween start and end, inclusive of start, end or both."""
        flat = self.flat
        if start is None:
            mask = None
        elif inclusive in {"start", "both"}:
            mask = pc.greater_equal(flat, start)
        else:
            mask = pc.greater(flat, start)
        if end is not None:
            if inclusive in {"end", "both"}:
                end_mask = pc.less_equal(flat, end)
            else:
                end_mask = pc.less(flat, end)
            if mask is None:
                mask = end_mask
            else:
                mask = pc.and_(mask, end_mask)

        return mask

    def slice(self, start=None, end=None, inclusive="both"):
        """Slice each array between start and end."""
        if start is None and end is None:
            raise ValueError("start and end must not be None")

        flat_indices = self.flat_inner_indices
        if end is not None:
            end_array = _align(pa.array(end), self.array)
            if inclusive in {"start", "both"}:
                mask = pc.less_equal(flat_indices, end_array)
            else:
                mask = pc.less(flat_indices, end_array)
        else:
            mask = None

        if start is not None:
            start_array = _align(pa.array(start), self.array)
            if mask is None and inclusive in {"start", "both"}:
                mask = pc.greater_equal(flat_indices, start_array)
            elif mask is None:
                mask = pc.greater(flat_indices, start_array)
            else:
                mask = pc.and_(mask, pc.greater_equal(flat_indices, start_array))

        new_index = _align(pa.array(self.index, from_pandas=True), self.array)
        new_index = new_index.filter(mask)
        new_offsets = _offsets(pd.Series(self.array_indices.filter(mask)))
        new_values = self.flat.filter(mask)

        array = pa.ListArray.from_arrays(values=new_values, offsets=new_offsets)
        new_index = pa.ListArray.from_arrays(values=new_index, offsets=new_offsets)
        new_index = _get_at(new_index, 0)

        return pd.Series(data=array, dtype=self.type, index=new_index)

    @pack_results
    def match(self, other, direction="forward"):
        """Return values from other array that are closest to each value in each array."""
        other_array = pa.array(other, from_pandas=True)

        if direction == "forward":
            side = "left"
        elif direction == "backward":
            side = "right"
        else:
            raise ValueError("Direction must be either forward or backward")

        df = pd.Series(self.array).to_frame("_right")
        df["_left"] = other_array
        df = pd.Series(np.unstack(df.to_numpy()), index=self.index)
        max1 = int(pc.max(self.array.flatten()))
        max2 = int(pc.max(other_array.flatten()))
        dummy = max(max1, max2) + 1

        def _match_func(array, side, dummy):
            arr = np.concat([array[1], [dummy]])
            match = np.searchsorted(arr, array[0], side=side)
            return arr[match]

        array = pa.array(
            df.apply(
                _match_func,
                args=(
                    side,
                    dummy,
                ),
            ),
            from_pandas=True,
        )

        # replace dummy values
        array = array.flatten()
        return pc.if_else(pc.equal(array, dummy), None, array)

    def filter(self, mask):
        """Filter by mask boolean listarray."""
        flat_mask = _flatten(pa.array(mask, from_pandas=True))
        array_indices = self.array_indices.filter(flat_mask)
        ref = self.flat.filter(flat_mask)

        # mask index
        base_index = _align(pa.array(self.index, from_pandas=True), self.array)
        base_index = pc.unique(base_index.filter(flat_mask))

        offsets = _offsets(pd.Series(array_indices, dtype=pd.ArrowDtype(array_indices.type)))

        res = pa.ListArray.from_arrays(values=ref, offsets=offsets, type=self.array.type)
        return pd.Series(res, index=base_index, dtype=self.type)

    @pack_results
    def replace(self, to_replace, new_values):
        """Replace values by value in each array, both can be scalar or same sized arrays/series."""
        return pc.if_else(pc.equal(self.flat, to_replace), new_values, self.flat)

    def insert(self, values, position):
        """Insert values at position (array or integer) in array."""
        lengths = self.value_lengths

        if isinstance(position, int):
            pos = pa.repeat(position, len(values))
        elif isinstance(position, pd.Series):
            pos = pa.array(position, from_pandas=True)
        elif isinstance(position, pa.Array):
            pos = position

        if pc.any(pc.less(pos, 0)):
            raise ValueError("positions must be positive")
        if pc.any(pc.greater(pos, lengths)):
            raise ValueError("positions must be smaller or equal to array lengths")

        # result size is array + nb subarray
        final_lengths = pc.add(lengths, 1)

        # align values and position to final result format
        final_values = _align_to_lengths(values, final_lengths)
        final_pos = _align_to_lengths(pos, final_lengths)

        # inner array indices in final format
        full_indices = pa.arange(0, len(final_pos))
        shift = pc.cumulative_sum(pa.array([0, *final_lengths.tolist()]))[:-1]
        shift = _align_to_lengths(shift, final_lengths)
        final_indices = pc.subtract(full_indices, shift)

        # take position in input array in final format
        shift = _align_to_lengths(pa.arange(0, self.length), final_lengths)
        final_take = pc.subtract(full_indices, shift)
        take_above = pc.subtract(final_take, 1)

        # take in array if before or equal position and
        # final_take not last row
        # as adding to last last position of last row is outside initial indices
        # else take position above (-1)
        mask = pc.and_(
            pc.less_equal(final_indices, final_pos),
            pc.not_equal(final_take, len(self.flat)),
        )
        final_take = pc.if_else(mask, final_take, take_above)

        # map original data to new format, select values at position
        final_res = self.flat.take(final_take)
        final_res = pc.if_else(pc.equal(final_indices, final_pos), final_values, final_res)

        # new offsets
        offsets = pc.add(self.offsets, pa.arange(0, self.length + 1))
        res = pa.ListArray.from_arrays(values=final_res, offsets=offsets)
        return pd.Series(res, index=self.index, dtype=self.type)

    @flat_results
    def interpolation_ratio(self, position, values):
        """Linearly interpolate between position and previous position if values were inserted."""
        pos_array = pa.array(position)
        values_array = pa.array(values)

        if pc.any(pc.equal(pos_array, self.value_lengths)):
            raise NotImplementedError("Insertion after last position")

        after = _get_at(self.array, pos_array)
        before = _get_at(self.array, pc.subtract(pos_array, 1))

        ratio = pc.divide(pc.subtract(values_array, before), pc.subtract(after, before))

        return ratio

    @flat_results
    def interpolate(self, position, ratio):
        """Linearly interpolate a value at position based on ratio, return na for first position."""
        pos_array = pa.array(position, from_pandas=True)
        ratio_array = pa.array(ratio, from_pandas=True)

        if pc.any(pc.equal(pos_array, self.value_lengths)):
            raise NotImplementedError("Interpolation after last position")

        after = _get_at(self.array, pos_array)
        before = _get_at(self.array, pc.subtract(pos_array, 1))

        values = pc.add(before, pc.multiply(ratio_array, pc.subtract(after, before)))

        # if position is 0, use next_position
        values = pc.if_else(pc.equal(pos_array, 0), after, values)

        return values

    def search_sorted(self, values, inclusive=True):
        """Return position in sorted ListArray that value must be inserted to maintain order."""
        if isinstance(values, pd.Series):
            val_array = pa.array(values, from_pandas=True)
        elif isinstance(values, pa.Array):
            val_array = values
        else:
            raise ValueError("Values must be a Series or an Array")

        if inclusive:
            mask = pc.less_equal(_align(val_array, self.array), self.flat)
        else:
            mask = pc.less(_align(val_array, self.array), self.flat)

        # create list array for mask is true
        indices = self.flat_inner_indices.filter(mask)
        array_indices = self.array_indices.filter(mask)

        table = pa.table([array_indices, indices], names=["keys", "values"])
        table = table.group_by("keys", use_threads=True).aggregate([("values", "min")])

        # convert to self
        df = table["values_min"].to_pandas(types_mapper=pd.ArrowDtype)
        df.index = self.index[table["keys"]]
        df = df.reindex(self.index)

        # if insertion position is after last point, return length
        return df.fillna(pd.Series(self.value_lengths, index=df.index))
