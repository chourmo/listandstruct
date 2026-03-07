# listandstruct

Minimal helpers around pandas and pyarrow list/struct arrays.

## Installation

```bash
pixi install
```

## Usage Examples

### StructArray

Convert a DataFrame to a Series of structs:

```python
import pandas as pd
from listandstruct import struct_array

# Create a DataFrame
df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': ['x', 'y', 'z']
})

# Convert to StructArray
structs = struct_array(df)
print(structs)

# Expand back to DataFrame
df_expanded = structs.structarray.expand()
```

### ListArray

Group a Series into a ListArray by ids:

```python
import pandas as pd
from listandstruct import list_array

# Create data with grouping ids
values = pd.Series([1, 2, 3, 4, 5, 6])
ids = pd.Series(['A', 'A', 'A', 'B', 'B', 'B'])

# Create ListArray
lists = list_array(values, ids=ids)
print(lists)
# Output: 0    [1, 2, 3]
#         1    [4, 5, 6]

# Flatten back
flat = lists.listarray.flatten()

# Aggregate operations
sums = lists.listarray.aggregate('sum')
means = lists.listarray.aggregate('mean')

# Get elements at position
first = lists.listarray.get(0)  # First element of each array
last = lists.listarray.get(-1)  # Last element of each array
```

## Development

Run tests:

```bash
pixi run test
```

Open a Python shell:

```bash
pixi run python
```
