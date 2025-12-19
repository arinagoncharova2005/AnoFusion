import pickle

# Replace with your actual file paths
test_pkl = "../nmiMatrix/mobservice2test_nmiMatrix.pk"
train_pkl = "../nmiMatrix/mobservice2train_nmiMatrix.pk"

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# ...existing code...
train_data = load_pkl(train_pkl)
test_data = load_pkl(test_pkl)

# Print first 10 elements/rows
print("\nFirst 10 elements from train_data:")
if hasattr(train_data, 'head'):
    print(train_data.head(10))
elif isinstance(train_data, (list, tuple)):
    print(train_data[:10])
elif hasattr(train_data, 'shape'):
    print(train_data[:10])
else:
    print(list(train_data)[:10])

print("\nFirst 10 elements from test_data:")
if hasattr(test_data, 'head'):
    print(test_data.head(10))
elif isinstance(test_data, (list, tuple)):
    print(test_data[:10])
elif hasattr(test_data, 'shape'):
    print(test_data[:10])
else:
    print(list(test_data)[:10])
# ...existing code...
# train_data = load_pkl(train_pkl)
# test_data = load_pkl(test_pkl)

# # If your PKL files contain DataFrames
# if hasattr(train_data, 'head') and hasattr(test_data, 'head'):
#     print('df')
#     print("Train shape:", train_data.shape)
#     print("Test shape:", test_data.shape)
#     # Compare columns
#     print("Columns equal:", list(train_data.columns) == list(test_data.columns))
#     # Compare overlapping timestamps (if present)
#     if 'timestamp' in train_data.columns and 'timestamp' in test_data.columns:
#         overlap = set(train_data['timestamp']) & set(test_data['timestamp'])
#         print("Overlapping timestamps:", len(overlap))
# else:
#     # For numpy arrays or lists
#     print("Train type:", type(train_data))
#     print("Test type:", type(test_data))
#     if isinstance(train_data, (list, tuple)) and isinstance(test_data, (list, tuple)):
#         print("Train length:", len(train_data))
#         print("Test length:", len(test_data))
#     elif hasattr(train_data, 'shape') and hasattr(test_data, 'shape'):
#         print("Train shape:", train_data.shape)
#         print("Test shape:", test_data.shape)
#     # Add more checks as needed

# # For dicts, you can compare keys
# if isinstance(train_data, dict) and isinstance(test_data, dict):
#     print("Train keys:", train_data.keys())
#     print("Test keys:", test_data.keys())