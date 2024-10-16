# ====================Evalutation Matrices Example===========================

from sklearn.metrics import accuracy_score, precision_score, recall_score, 
f1_score, mean_squared_error, r2_score

# Sample data
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1]

# Evaluation Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score
: {f1}')

# For regression example
y_true_reg = [3, -0.5, 2, 7]
y_pred_reg = [2.5, 0.0, 2, 8]

mse = mean_squared_error(y_true_reg, y_pred_reg)
r_squared = r2_score(y_true_reg, y_pred_reg)

print(f'MSE: {mse}, R-squared: {r_squared}')



# =========================Descriptive Statistics Example=============================

import numpy as np
from scipy import stats

data = [1, 2, 2, 3, 4, 5, 5, 6, 7]

mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data)[0][0]
data_range = np.ptp(data)  # Peak to peak (max - min)
variance = np.var(data)
std_deviation = np.std(data)

print(f'Mean: {mean}, Median: {median}, Mode: {mode}, Range: {data_range}, Variance: 
{variance}, Standard Deviation: {std_deviation}')



#=============Data Preprocessing. EDA and Feature Engineering Example==================

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Display the first few rows of the dataset
print("Initial Dataset:")
print(data.head())

# Data Cleaning
# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Fill missing values or drop columns
data['Age'].fillna(data['Age'].median(), inplace=True)  # Fill missing Age with median
data.drop(columns=['Cabin', 'Ticket'], inplace=True)    # Drop Cabin and Ticket columns
data.dropna(subset=['Embarked'], inplace=True)          # Drop rows where Embarked is NaN

# Verify the changes
print("\nMissing Values After Cleaning:")
print(data.isnull().sum())

# Statistical Analysis
# Display basic statistics
print("\nStatistical Summary:")
print(data.describe())

# EDA Techniques
# Visualize the distribution of 'Age'
plt.figure(figsize=(10, 5))
sns.histplot(data['Age'], bins=30, kde=True)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Visualize the survival rate by gender
plt.figure(figsize=(8, 5))
sns.countplot(x='Sex', hue='Survived', data=data)
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Visualize survival rate by class
plt.figure(figsize=(8, 5))
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()

# Finding Useful Insights
# Calculate survival rate
survival_rate = data['Survived'].mean()
print(f"\nOverall Survival Rate: {survival_rate:.2%}")

# Group by 'Sex' and 'Pclass' for more insights
grouped = data.groupby(['Sex', 'Pclass'])['Survived'].mean()
print("\nSurvival Rate by Gender and Class:")
print(grouped)

# Feature Engineering
# Convert categorical variables to numerical
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})  # Male: 0, Female: 1
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})  # C: 0, Q: 1, S: 2

# Create a new feature: Family Size
data['Family_Size'] = data['SibSp'] + data['Parch']
data['Is_Alone'] = np.where(data['Family_Size'] == 0, 1, 0)  # 1 if alone, else 0

# Display the final dataset after feature engineering
print("\nFinal Dataset After Feature Engineering:")
print(data.head())

# Save the cleaned and processed dataset
data.to_csv('cleaned_titanic_data.csv', index=False)
print("\nCleaned dataset saved as 'cleaned_titanic_data.csv'.")



# ========================KNN example==============================================

# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating the KNN model
k = 3  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)

# Training the model
knn.fit(X_train, y_train)

# Making predictions
y_pred = knn.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Printing results
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:\n', confusion)
print('Classification Report:\n', report)


# =======================Linear Regression Example==============================

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generating synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # 100 samples of a single feature
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relation with noise

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the Linear Regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Printing results
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Plotting the results
plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', label='Predicted line')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()


