### ML Dashboard (dash + python + plotly)

This project presents an interactive Machine Learning dashboard built with Python for visualizing, analyzing, and monitoring machine learning workflows and model performance. The dashboard is designed to simplify data exploration and provide an intuitive interface for understanding how machine learning models behave during training and prediction.

The project includes:
- Data preprocessing and feature analysis
- Interactive visualizations and performance metrics
- Machine learning model training and evaluation
- Accuracy, loss, and prediction monitoring
- Graphical comparison of model outputs
- Dashboard interface for real-time analysis and insights

The dashboard supports machine learning workflows by helping users explore datasets, evaluate model performance, identify patterns in predictions, and better understand training behavior through visual analytics. It can be used for educational purposes, model experimentation, and basic deployment-style monitoring of machine learning applications.


Technologies used include Python, Plotly, Pandas, NumPy, Matplotlib, Scikit-learn, and dashboard/visualization tools integrated into the application.

This project is build with AI assistance.


Python environment and bash commands:

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

