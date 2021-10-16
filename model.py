import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
import pickle

# Load the csv file
Data = pd.read_csv('EasyGoDS.csv')
Data

print(Data.head())

dummy_month = pd.get_dummies(Data['Month'])
dummy_month

dummy_day = pd.get_dummies(Data['Day'])
dummy_day

dummy_category = pd.get_dummies(Data['Category'])
dummy_category

dummy_traffic = pd.get_dummies(Data['Traffic'])
dummy_traffic

dummy_Peak = pd.get_dummies(Data['Peak_hour'])
dummy_Peak

egdataset = pd.concat([Data,  dummy_month, dummy_day, dummy_category, dummy_Peak, dummy_traffic], axis=1)
egdataset

final = egdataset.drop(['Month','Day','Category','Traffic','Peak_hour'],axis='columns')
final

final['Date'] = pd.to_datetime(final['Date'], infer_datetime_format=True)
Dataset = final.set_index(['Date'])
Dataset

Dataset = Dataset[['Year','Time','Passengers','NOB','April','December','January','Fri','Mon','Sat','Sun','Tue','Wed','Holiday','Weekday','Weekend','off-peak','peak','no','serious','slight']]
Dataset[['Year','Time','April','December','January','Fri','Mon','Sat','Sun','Tue','Wed','Holiday','Weekday','Weekend','off-peak','peak','no','serious','slight','Passengers','NOB']]

# Select independent and dependent variable
X = Dataset.drop('NOB', axis = 1)
Y = Dataset['NOB']

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=50)

# Feature scaling
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# Fit the model
model = LinearRegression()
model.fit(X_train,y_train)

# Make pickle file of our model
pickle.dump(model, open("model.pickle", "wb"))