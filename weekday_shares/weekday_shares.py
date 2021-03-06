import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

"""
Data information
- Link to website: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
- Reference: Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", 
    Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.
"""

bike_day_data = pd.read_csv("bike_sharing_business\day.csv")
bike_day_data = bike_day_data[['weekday', 'cnt']]
bike_day_data = np.array(bike_day_data)

weekday_bike_shares = [0] * 7

for row in bike_day_data:
    weekday_bike_shares[row[0]] += row[1]


bike_x_train, bike_x_test, bike_y_train, bike_y_test = sklearn.model_selection.train_test_split(bike_x, bike_y,
                                                                                                test_size=0.10)


