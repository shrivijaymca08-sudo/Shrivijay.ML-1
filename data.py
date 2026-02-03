import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("student-data.csv")

# Print column names
print(data.columns)

# Select only numeric columns to avoid errors
numeric_data = data.select_dtypes(include=['int64', 'float64'])

# Take first column as X and second column as y
X = numeric_data.iloc[:, [0]]
y = numeric_data.iloc[:, 1]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Output results
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)
print("Predicted Values:", y_pred)
