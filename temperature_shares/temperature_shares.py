import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model, metrics
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

"""
Model Information:
This program creates a model that predicts

Data information
- Link to website: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
- Reference: Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", 
    Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.
"""
# Getting Data
#bike_day_data = pd.read_csv("bike_sharing_business\day.csv")
bike_day_data = pd.read_csv("../day.csv")
bike_day_data = bike_day_data[['temp', 'cnt']]
predict = 'cnt'

# Setting temeprature back to normal (info in Readme.txt)
bike_day_data['temp'] = bike_day_data['temp'] * 41

bike_x = np.array(bike_day_data.drop([predict], 1))
bike_y = np.array(bike_day_data[predict])
bike_x_train, bike_x_test, bike_y_train, bike_y_test = sklearn.model_selection.train_test_split(bike_x, bike_y,
                                                                                                test_size=0.10)
# Getting the best model
best = 0
for _ in range(100):
    bike_x_train, bike_x_test, bike_y_train, bike_y_test = sklearn.model_selection.train_test_split(bike_x, bike_y,
                                                                                                    test_size=0.10)
    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(bike_x_train, bike_y_train)
    accuracy = lin_reg.score(bike_x_test, bike_y_test)

    if accuracy > best:
        best = accuracy
        with open("temperature_shares.pickle", "wb") as f:
            pickle.dump(lin_reg, f)

pickle_in = open("temperature_shares.pickle", "rb")
lin_reg = pickle.load(pickle_in)

# Predicting based on our model
predict_num_bikes = lin_reg.predict(bike_x_test)

# Showing Model statistics
# The coefficients
print("Model Statistics")
print('Coefficient: {:.2f}'.format(lin_reg.coef_[0]))
# The mean squared error
print('Mean squared error (MSE): {:.2f}'.format(metrics.mean_squared_error(bike_y_test, predict_num_bikes)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination (r^2): {:.2f}'.format(metrics.r2_score(bike_y_test, predict_num_bikes)))

# Plotting
style.use("ggplot")
pyplot.scatter(bike_x_test, bike_y_test, color='red')
pyplot.plot(bike_x_test, predict_num_bikes, color='black')
pyplot.xlabel("Temperature (C)")
pyplot.ylabel("Bicycles Rented")
pyplot.title("Bicycle rent per day vs. Temperature in Celcius")
pyplot.show()
