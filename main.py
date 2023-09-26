from mlsecu.anomaly_detection_use_case import get_list_of_if_outliers
import pandas as pd
import numpy as np

array = np.random.randint(10, size=(100))
print(list(np.unique(array)))