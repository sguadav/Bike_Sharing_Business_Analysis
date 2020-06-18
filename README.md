# Bike_Sharing_Business_Analysis
This is an program that analyzes data from a csv file on a Bicycle Sharing Business. For this program I focused on Supervised Machine Learning and statistical analysis of the data gather, dividing it to have training and testing data. Additionally, the program uses a library that saves the most accurate model from a repetition of 100 times. Currently, two analysis are made: 
1) Analyzes how the outside temperature affects the number of bicycles shared that day (temperature_shares).
2) Analyzes how the weekday affects the number of bicycles shared (weekday_shares).

Getting Started
-
To properly run this program, install:
- Python 3.6
- Pandas
- Numpy
- Sklearn
- Pickle
- Matplotlib

Running the program
-
This project consists on two analysis on how some factor might affect a Bicycle Sharing Business and getting the data from a csv file.

To stat the program, go to the folder that you want to see on how that external factor can affect the business. Mkae sure that the refernce to the csv file is correct.

For the Temperature vs Bike Shares analysis, you should get the following output

<img src='images/graphs.PNG' width=250>

<img src='images/Model_stats.PNG' width250>

Since the program looks for a new model every time, the answers will not be the same everytime.

References
-
- Link to website: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
- Data Reference: Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge" Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.
