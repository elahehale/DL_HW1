## Laser Data Loader

This module loads and preprocesses laser time-series data for one-step-ahead prediction. It supports both training mode and test
mode (using a pre-fitted scaler to transform unseen data)

Pipeline in Train mode:
1. Split data (train/validation)
2. Scale data (fit on training only)
3. Create sequences (sliding window sequences) and return DataLoaders

Pipline in Test mode:
1. Scale data with pre-fitted scaler
2. Create sequences  
### Usage

```python

from dataset import LaserData
from sklearn.preprocessing import MinMaxScaler

dataset = LaserData(
    path="data/Xtrain.mat",
    split_ratio=0.8,
    sequence_length=20,
    scaler=MinMaxScaler(),
    mode="train"
)

train_loader, val_loader = dataset.get_loaders(batch_size=32)

X_train, y_train = train_loader.dataset.tensors

print(X_train.shape)  # (samples, sequence_length, 1)
print(y_train.shape)  # (samples, 1)
```
