#----------------- F E A T U R E S       S E L E C T I O N ----------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("diabetes.csv")
print(data)
print(data.columns)
print(data.corr())

