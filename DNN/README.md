We have to install hdf5:

```BASH
 brew install hdf5
 brew install graphviz
```
With the system setup like that, we can go and create our environment and install tensorflow

```BASH
pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
```
If you already have hdf5
```BASH
export HDF5_DIR=/opt/homebrew/Cellar/hdf5/1.12.2
```
otherwise, if you have just installed hdf5 with brew, then
```BASH
export HDF5_DIR=/opt/homebrew/Cellar/hdf5/1.12.2_2
```

```BASH
pip install -U pip
pip install --no-binary=h5py h5py
pip install -r requirements.txt
```
