```python
import sys
print(sys.executable)
import numpy
import pandas as pd
print("numpy:", numpy.__version__)
print("pandas:", pd.__version__)
```

```bash
source dashboard_env/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=dashboard_env --display-name "Python (dashboard_env)"
```

```bash
 make clean
 make
 source dashboard_env/bin/activate
```

