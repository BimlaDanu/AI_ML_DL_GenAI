- The notebook [artsy_challenge.ipynb](artsy_challenge.ipynb)  solve the Artsy challenge [data_analyst_artsy.md](data_analyst_artsy.md) 

- The notebook [triad_challenge.ipynb](triad_challenge.ipynb) solve the TRI-AD challenge [data_scientist_triad.md](data_scientist_triad.md)

- The notebook [amboss_challenge.ipynb](amboss_challenge.ipynb) solve the challenge [product_analyst_amboss.md](product_analyst_amboss.md)

- The notebook [Notification_Bundler/solution_walkthrough.ipynb](Notification_Bundler/solution_walkthrough.ipynb) solve the challenge [Notification_Bundler/senior_data_scientist_engineer_k.md](Notification_Bundler/senior_data_scientist_engineer_k.md)

To install hdf5
```BASH
 brew install hdf5
 brew install graphviz
```

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
