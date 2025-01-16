import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost

data = pd.read_csv('Dataset--Heart-Disease-Prediction-using-ANN.csv')

# Create the target and variables
X = data.drop(columns=['target'])
y = data['target']

# Split the data into training data and test data
# Set a random state so that comparisons between parameters can be made
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.2, random_state=42)

# Fit the model
from xgboost import XGBRegressor
from xgboost import XGBClassifier
my_model = XGBClassifier(n_estimators=500, learning_rate=0.02,n_jobs=10)
my_model.fit(X_train, y_train)

# Make predictions and evaluate the model using Mean Absolute Error
from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))



