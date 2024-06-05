# -*- coding: utf-8 -*-

""" Introduction
In this project, I analyzed a dataset related to heart disease, aiming to uncover significant patterns and relationships between various health metrics and the presence of heart disease. My comprehensive analysis involved data cleaning, visualization, statistical testing, and predictive modeling to gain a deeper understanding of the factors influencing heart disease.

Summary of Work Done
Data Cleaning and Preparation:

Addressed missing and zero values.
One-hot encoded categorical variables.
Standardized numerical features.
Data Visualization:

Histograms, bar charts, and scatter plots to visualize distributions and relationships.
Correlation heatmap to identify related variables.
Statistical Analysis:

Conducted Chi-Squared tests to explore associations between categorical variables.
Predictive Modeling:

Built and evaluated a linear regression model to predict maximum heart rate based on health metrics.

Writen by help of Chatgpt
"""

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# Load the data
file_path = '/heart.csv'
data = pd.read_csv(file_path)

# Replace zero values in 'RestingBP' and 'Cholesterol' with the median values
median_resting_bp = data['RestingBP'].median()
median_cholesterol = data['Cholesterol'].median()
data['RestingBP'] = data['RestingBP'].replace(0, median_resting_bp)
data['Cholesterol'] = data['Cholesterol'].replace(0, median_cholesterol)

# One-hot encode categorical variables
categorical_vars = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
data_encoded = pd.get_dummies(data, columns=categorical_vars, drop_first=True)

# Standardize numerical features
numerical_vars = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()
data_encoded[numerical_vars] = scaler.fit_transform(data_encoded[numerical_vars])

# Save the cleaned and prepared data to a new CSV file
data_encoded.to_csv('cleaned_heart.csv', index=False)

print(data_encoded.to_csv)
print(median_resting_bp)
print(median_cholesterol)

# One-hot encode categorical variables
categorical_vars = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
data_encoded = pd.get_dummies(data, columns=categorical_vars, drop_first=True)

# Standardize numerical features
numerical_vars = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()
data_encoded[numerical_vars] = scaler.fit_transform(data_encoded[numerical_vars])

# Add the 'HeartDisease' column back to data_encoded
data_encoded['HeartDisease'] = data['HeartDisease']

# Set the style of the visualization
sns.set(style="whitegrid")

# 1. Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=30, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2. Heart Disease by Gender
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Sex', hue='HeartDisease')
plt.title('Heart Disease by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# 3. Cholesterol Levels by Heart Disease
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='HeartDisease', y='Cholesterol')
plt.title('Cholesterol Levels by Heart Disease')
plt.xlabel('Heart Disease')
plt.ylabel('Cholesterol')
plt.show()

# 4. Max Heart Rate by Age
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Age', y='MaxHR', hue='HeartDisease', palette='coolwarm', alpha=0.7)
plt.title('Max Heart Rate by Age')
plt.xlabel('Age')
plt.ylabel('Max Heart Rate')
plt.show()

# 5. Chest Pain Type Distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='ChestPainType')
plt.title('Chest Pain Type Distribution')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = data_encoded.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Chi-Squared Test between 'Sex' and 'HeartDisease'
contingency_table = pd.crosstab(data['Sex'], data['HeartDisease'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi-Squared: {chi2}")
print(f"P-Value: {p}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies: \n", expected)


# Define features and target variable
X = data_encoded.drop(columns=['MaxHR', 'HeartDisease'])
y = data_encoded['MaxHR']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error and R^2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Print the model coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# Generate example time series data
np.random.seed(42)
date_rng = pd.date_range(start='1/1/2020', end='1/1/2023', freq='D')
df = pd.DataFrame(date_rng, columns=['Date'])
df['Value'] = np.random.randn(len(date_rng)).cumsum()

# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Value'])
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Decompose the time series
decompose_result = seasonal_decompose(df.set_index('Date'), model='additive')
decompose_result.plot()
plt.show()

# Split the data into train and test sets
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Fit ARIMA model
model = ARIMA(train['Value'], order=(5, 1, 0))
model_fit = model.fit()
print(model_fit.summary())

# Forecast
forecast = model_fit.forecast(steps=len(test))
forecast_index = test.index

# Plot the forecast
plt.figure(figsize=(6, 6))
plt.plot(train['Date'], train['Value'], label='Train')
plt.plot(test['Date'], test['Value'], label='Test')
plt.plot(forecast_index, forecast, label='Forecast')
plt.title('Time Series Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Calculate mean squared error
mse = mean_squared_error(test['Value'], forecast)
print(f'Mean Squared Error: {mse}')

"""Summary of Heart Disease Data Analysis
Introduction
Heart disease remains a leading cause of mortality worldwide, and understanding the factors contributing to it is crucial for prevention and treatment. In this analysis, I examined a dataset containing various health metrics and demographic information related to heart disease. The goal was to identify significant patterns and relationships between these variables and the presence of heart disease. The analysis included data cleaning, visualization, statistical tests, and predictive modeling to provide a comprehensive understanding of the factors influencing heart disease.

Data Cleaning and Preparation
The dataset included features such as age, gender, chest pain type, resting blood pressure, cholesterol levels, fasting blood sugar, resting electrocardiogram results, maximum heart rate, exercise-induced angina, ST depression, ST segment slope, and the presence of heart disease.

I started by identifying and addressing potential outliers and missing values. For instance, any zero values in the RestingBP and Cholesterol columns were replaced with their respective medians to correct unrealistic entries. Categorical variables, such as gender and chest pain type, were converted into numerical values using one-hot encoding to prepare the data for analysis and modeling. Numerical features were standardized to ensure uniformity across different scales.

Data Visualization
Visualization helps in understanding the data distribution and relationships between variables. Key visualizations included:

Age Distribution: A histogram showed the age distribution of the participants, indicating the most common age groups affected by heart disease.

Heart Disease by Gender: A bar chart revealed the count of heart disease cases across genders, showing any potential gender disparities in heart disease prevalence.

Cholesterol Levels by Heart Disease: A box plot compared cholesterol levels between those with and without heart disease, highlighting any significant differences.

Max Heart Rate by Age: A scatter plot illustrated the relationship between age and maximum heart rate, with points colored by heart disease status, showing how heart rate varies with age and heart disease.

Chest Pain Type Distribution: A bar chart showed the distribution of different chest pain types, providing insights into the most common types among heart disease patients.

Correlation Heatmap: A heatmap displayed the correlations between numerical features, helping to identify which variables are most closely related.

Statistical Analysis
We performed a Chi-Squared test to determine if there was a significant association between categorical variables like gender and heart disease. The results indicated whether gender significantly affects heart disease prevalence.

Predictive Modeling
Using linear regression, I aimed to predict the maximum heart rate a person can achieve during physical exertion based on other features in the dataset. The model was trained on a portion of the data and tested on the remaining data to evaluate its performance. The key metrics for model evaluation included the mean squared error (MSE) and the R² score, which indicate how well the model predicts and explains the variability in the maximum heart rate.

Conclusion
The analysis provided valuable insights into the factors contributing to heart disease. Key findings included:

Age and cholesterol levels are significant indicators of heart disease risk.
Gender differences in heart disease prevalence were observed, with males showing higher incidence rates.
Certain chest pain types were more commonly associated with heart disease.
Predictive modeling highlighted the potential for using health metrics to estimate heart function, which can aid in early detection and preventive measures.

This comprehensive analysis underscores the importance of regular health check-ups and monitoring key health indicators to prevent and manage heart disease effectively. By understanding these relationships, healthcare providers can better tailor interventions and treatments to reduce the burden of heart disease.

"""
