# import libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# access data
url = "Student_Grade.csv"

# read data
df = pd.read_csv(url)

# create two lists one of study time in minutes and one of the grade achived
x = df["study_min"]
y = df["grade"]

# split our data in half, one for training and one for testing
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.5, random_state = 42)

# format the data to make it readable for the model
x_train = np.array(x_train).reshape(-1,1)
x_test = np.array(x_test).reshape(-1,1)
x = np.array(x).reshape(-1,1)

# train our model
model = LinearRegression()
model.fit(x_train, y_train)

# run the model and make predections with test data
predictions = model.predict(x_test)

# make final predections with original data
preds = model.predict(x)

# format the prediction output
df["predictions"] = model.predict(np.array(df["study_min"]).reshape(-1, 1))
print(df["predictions"])

# import a library that helps with graphs
import matplotlib.pyplot as plt

# put data points onto the graph in blue
plt.scatter(x,y, color = "b")

# plot predicted data point in green
plt.scatter(x,preds, color = "g")

# plot line of best fit from our prediction model in red
plt.plot(x , preds, color = "r")

# show the graph
plt.show()

# convert our grade and prediction grade to a percent
df["predictions"] = df["predictions"]*10
df["grade"] = df["grade"]*10

# save original data with predictions in a new csv file
df.to_csv("blank.csv", index=False)