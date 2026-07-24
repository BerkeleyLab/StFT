import pickle

with open('StFT/plasma_(1).pkl', 'rb') as f:
    f_obj = pickle.load(f)

print(type(f_obj))

for key, value in f_obj.items():
    info = type(value).__name__
    if hasattr(value, 'shape'):
        info += f" shape={value.shape} dtype={getattr(value, 'dtype', '')}"
    elif hasattr(value, '__len__'):
        info += f" len={len(value)}"
    print(f"{key!r}: {info}")