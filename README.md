# USAHouse_price_prediction2025
usa house price prediction 2025 is using price of different categories of houses with accurate sufficent requirement and plot size 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

HouseDF = pd.read_csv('USA_Housing.csv')
HouseDF.head()
HouseDF = HouseDF.reset_index()
HouseDF.head()
HouseDF.info()
HouseDF.describe()
HouseDF.columns

sns.pairplot(HouseDF)
sns.histplot(HouseDF['Price'], kde=True)
plt.show() # Added plt.show() to display the plot
sns.heatmap(HouseDF.corr(), annot=True)
plt.show() # Added plt.show() to display the heatmap

X = HouseDF[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = HouseDF['Price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# Create separate scalers for features and the target variable
from sklearn.preprocessing import MinMaxScaler
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()


# Fit and transform the training data
X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test) # Only transform the test data

# Reshape y_train for scaling
y_train_reshaped = y_train.values.reshape(-1, 1)
y_test_reshaped = y_test.values.reshape(-1, 1)

y_train_scaled = y_scaler.fit_transform(y_train_reshaped)
y_test_scaled = y_scaler.transform(y_test_reshaped)


# The linear model part was separate from the LSTM, keeping it for completeness but it's not used for LSTM
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
print("Linear Regression Intercept:", lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print("Linear Regression Coefficients:\n", coeff_df)


from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train_scaled.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=y_train_scaled.shape[1])) # Output layer should match the number of target features


model.compile(optimizer='adam', loss='mean_squared_error')

# Reshape input for LSTM [samples, timesteps, features]
X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))


model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32) # Added batch_size for clarity

predictions_scaled = model.predict(X_test_scaled)

# Inverse transform the predictions to the original scale
predictions = y_scaler.inverse_transform(predictions_scaled)


# The scaling and plotting of y_test and y_predicted needs careful handling with the original y_test
# We should compare the predicted prices (in original scale) with the original y_test
plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

sns.histplot((y_test - predictions.flatten()), bins=50, kde=True) # Flatten predictions for plotting
plt.title("Residuals Distribution")
plt.show()

# Plotting time series - This is not a time series dataset, so this plot is not appropriate.
# If you intended to plot actual vs predicted values by index, you can do this:
# plt.figure(figsize=(12,6))
# plt.plot(y_test.values, 'b', label='Original Price') # Use .values to get numpy array
# plt.plot(predictions, 'r', label='Predicted Price')
# plt.xlabel('Index')
# plt.ylabel('Price')
# plt.legend()
# plt.show()


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
import pandas as pd
import numpy as np

# Create a dictionary with dummy data
data = {
    'Avg. Area Income': np.random.rand(5000) * 100000,
    'Avg. Area House Age': np.random.rand(5000) * 20 + 5,
    'Avg. Area Number of Rooms': np.random.rand(5000) * 5 + 3,
    'Avg. Area Number of Bedrooms': np.random.rand(5000) * 3 + 2,
    'Area Population': np.random.rand(5000) * 50000 + 10000,
    'Price': np.random.rand(5000) * 1000000 + 100000
}

# Create a pandas DataFrame
dummy_df = pd.DataFrame(data)

# Save the DataFrame to a CSV file, overwriting if it exists
dummy_df.to_csv('USA_Housing.csv', index=False)

print("Dummy 'USA_Housing.csv' created successfully.")
