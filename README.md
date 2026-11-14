# 🏡 House Price Prediction Using Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Supervised-orange" />
  <img src="https://img.shields.io/badge/Algorithm-Multiple%20Linear%20Regression-green" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML%20Pipeline-F7931E?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>

<p align="center">
  <b>A Machine Learning project to predict residential property prices using Multiple Linear Regression.</b>
</p>

---

## 📌 1. Project Overview

House prices depend on several factors, such as the size of the property, number of bedrooms, bathrooms, location-related facilities, and furnishing status.

In this project, we develop a **House Price Prediction Model** using Multiple Linear Regression.

The model learns relationships between different property characteristics (input features) and their corresponding prices (target variable).

The project covers the complete Machine Learning workflow:

- Understanding and exploring the dataset.
- Identifying numerical and categorical features.
- Performing Exploratory Data Analysis (EDA).
- Handling missing values and duplicate records.
- Encoding categorical variables.
- Standardizing numerical features.
- Building a preprocessing and Machine Learning pipeline.
- Training a Multiple Linear Regression model.
- Evaluating model performance.

## 🎯 2. Project Objectives

1. Understand how different property characteristics influence house prices.
2. Learn the difference between numerical and categorical data.
3. Understand why categorical encoding is required for Machine Learning.
4. Understand feature standardization and its importance.
5. Implement preprocessing using Scikit-learn.
6. Train a Multiple Linear Regression model.
7. Evaluate predictions using regression performance metrics.

---

## 📂 3. Dataset Information

**Dataset:** `Housing.csv`

The dataset contains information about residential properties and their corresponding prices.

| Dataset Information | Details |
|---|---|
| Total Records | 545 |
| Total Columns | 13 |
| Input Features | 12 |
| Target Variable | 1 |
| Problem Type | Regression |
| Learning Type | Supervised Learning |

### What is a record?

A record represents one individual observation in the dataset.

For example, one row represents one house with its area, bedrooms, bathrooms, facilities, furnishing status, and price.

### What is a feature?

A feature is an independent variable used by a Machine Learning model to make predictions.

For example:
- Area
- Number of bedrooms
- Number of bathrooms
- Number of stories

### What is a target variable?

The target variable is the value that the model is trained to predict.

In this project:

**Target Variable: `price`**

The model uses the remaining 12 columns to predict the price of a house.

---

## 📊 4. Understanding Dataset Columns

The dataset contains 13 columns: 12 independent features and 1 dependent target variable.

| Column | Description | Data Type |
|---|---|---|
| price | Selling price of the house | Numerical (Target) |
| area | Total area of the property | Numerical |
| bedrooms | Number of bedrooms | Numerical |
| bathrooms | Number of bathrooms | Numerical |
| stories | Number of floors | Numerical |
| mainroad | Whether the house is connected to the main road | Categorical |
| guestroom | Whether a guest room is available | Categorical |
| basement | Whether the house has a basement | Categorical |
| hotwaterheating | Whether hot water heating is available | Categorical |
| airconditioning | Whether air conditioning is available | Categorical |
| parking | Number of parking spaces | Numerical |
| prefarea | Whether the property is in a preferred area | Categorical |
| furnishingstatus | Furnishing condition of the property | Categorical |

### Feature Classification

Our input features are divided into two major categories.

**A. Numerical Features**

Numerical data represents measurable or countable quantities.

Examples:
- Area: 5000 square feet
- Bedrooms: 3
- Bathrooms: 2
- Stories: 2
- Parking: 1

These values represent quantities that can be used in mathematical calculations.

We have **5 numerical input features**:
- `area`
- `bedrooms`
- `bathrooms`
- `stories`
- `parking`

**B. Categorical Features**

Categorical data represents groups, labels, or categories rather than continuous measurements.

Examples:
- Main road: Yes / No
- Guest room: Yes / No
- Furnishing status: Furnished / Semi-furnished / Unfurnished

These values describe the characteristics of a property.

We have **7 categorical input features**:
- `mainroad`
- `guestroom`
- `basement`
- `hotwaterheating`
- `airconditioning`
- `prefarea`
- `furnishingstatus`

### Why do we need to identify these data types?

Machine Learning algorithms perform mathematical operations on numerical values.

Therefore, categorical values such as `yes`, `no`, `furnished`, and `unfurnished` must be converted into numerical representations before they can be used by the model.

Different types of data also require different preprocessing techniques.

---

## 🔍 5. Exploratory Data Analysis (EDA)

Exploratory Data Analysis helps us understand the structure, distribution, and relationships within our dataset before training a model.

### Step 1: Import Libraries

We use the following Python libraries:

| Library | Purpose |
|---|---|
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, model training, and evaluation |

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### Step 2: Load the Dataset

```python
df = pd.read_csv("Housing.csv")

df.head()
```

`read_csv()` loads the CSV file into a Pandas DataFrame.

`head()` displays the first five records by default.

### Step 3: Understand Dataset Structure

```python
df.shape
df.info()
df.describe()
df.columns
```

**What do these functions do?**

- `shape`: Returns the number of rows and columns.
- `info()`: Displays column names, data types, and non-null counts.
- `describe()`: Generates statistical summaries of numerical columns.
- `columns`: Displays all column names.

### Step 4: Check Missing Values

```python
df.isnull().sum()
```

Missing values are records where information is unavailable.

For example, a house may have a missing area or furnishing status.

We check missing values because incomplete data can affect preprocessing and model training.

### Step 5: Check Duplicate Records

```python
df.duplicated().sum()
```

Duplicate records are repeated rows in a dataset.

They may introduce unnecessary repetition and affect the analysis.

### Step 6: Visualize Numerical Features

Histograms help us understand the distribution of numerical variables.

```python
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()
```

We can also use box plots to identify potential outliers.

### Step 7: Understand Correlation

```python
sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True,
    cmap="coolwarm"
)

plt.show()
```

Correlation measures the strength and direction of a linear relationship between numerical variables.

- Positive correlation: Both variables tend to increase together.
- Negative correlation: One variable tends to decrease as the other increases.
- Near-zero correlation: Little linear relationship is observed.

---

## ⚙️ 6. Data Preprocessing

Data preprocessing transforms raw data into a format suitable for Machine Learning algorithms.

Our dataset contains both numerical and categorical features, so we need separate preprocessing techniques.

### Step 1: Separate Features and Target

```python
X = df.drop("price", axis=1)
y = df["price"]
```

Here:

- `X` contains the 12 independent input features.
- `y` contains the target variable, `price`.

The model learns the relationship between X and y.

### Step 2: Identify Numerical and Categorical Columns

```python
numerical_features = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking"
]

categorical_features = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus"
]
```

We separate these features because numerical and categorical data require different preprocessing methods.

---

## 🔠 7. One-Hot Encoding

### What is One-Hot Encoding?

One-Hot Encoding is a technique used to convert categorical values into numerical columns.

Machine Learning algorithms cannot directly perform mathematical calculations on text categories.

For example, consider the `furnishingstatus` column:

| Original Value |
|---|
| Furnished |
| Semi-furnished |
| Unfurnished |

We cannot directly pass these text values to a Linear Regression model.

One-Hot Encoding creates separate binary columns for each category.

**After One-Hot Encoding:**

| Furnished | Semi-furnished | Unfurnished |
|---:|---:|---:|
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |

Each category gets its own column.

- `1` means the category is present.
- `0` means the category is absent.

### Why not simply assign 1, 2, and 3?

Suppose we encode:

- Furnished = 1
- Semi-furnished = 2
- Unfurnished = 3

This introduces an artificial numerical ordering and spacing between categories.

A Linear Regression model may interpret these numbers as meaningful quantities.

One-Hot Encoding avoids introducing that artificial numerical relationship.

### Implementation

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown="ignore"
)
```

**What does `handle_unknown="ignore"` do?**

If a category appears during prediction that was not present during training, the encoder handles it without raising an error.

---

## 📏 8. Standardization of Numerical Features

### What is Standardization?

Standardization is a feature scaling technique that transforms numerical data so that it has:

- Mean approximately equal to 0.
- Standard deviation approximately equal to 1.

Consider these features:

| Feature | Example Value |
|---|---:|
| Area | 5000 |
| Bedrooms | 3 |
| Bathrooms | 2 |
| Parking | 1 |

Notice that the numerical values have very different scales.

Area may have values in thousands, while bedrooms and bathrooms may have values in single digits.

### Why do we need Standardization?

Features with larger numerical scales can create difficulties for some Machine Learning algorithms and optimization methods.

Standardization puts numerical features on a comparable scale.

This is especially useful for algorithms such as:
- Linear Regression with regularization.
- Logistic Regression.
- Support Vector Machines.
- K-Nearest Neighbors.
- Gradient-based optimization methods.

Standardization is not strictly required for ordinary least-squares Linear Regression predictions, but it makes preprocessing consistent and is useful for many other algorithms.

### Standardization Formula

\[
z = \frac{x-\mu}{\sigma}
\]

Where:

- \(x\) = Original feature value.
- \(\mu\) = Mean of the feature.
- \(\sigma\) = Standard deviation of the feature.
- \(z\) = Standardized value.

### Example

Suppose the average area is 3000 and the standard deviation is 1000.

For a house with an area of 5000:

\[
z = \frac{5000-3000}{1000}=2
\]

The standardized value becomes 2.

### Implementation

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

**Important:** The scaler must learn its mean and standard deviation from training data only. It then uses those same values to transform test data. This prevents data leakage.

---

## 🔗 9. ColumnTransformer

### What is ColumnTransformer?

`ColumnTransformer` allows us to apply different preprocessing operations to different groups of columns.

In our project:

- Numerical columns → StandardScaler
- Categorical columns → OneHotEncoder

Instead of preprocessing each column manually, we can combine both transformations into one object.

### Implementation

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ]
)
```

### How does it work?

| Feature Type | Transformation |
|---|---|
| Numerical | StandardScaler |
| Categorical | OneHotEncoder |

The `ColumnTransformer` applies the appropriate transformation to each selected group of features and combines the transformed outputs.

---

## 🔀 10. Train-Test Split

### What is Train-Test Split?

We divide the dataset into two parts:

1. Training Dataset
2. Testing Dataset

The training dataset is used to teach the model.

The testing dataset is used to evaluate how well the trained model performs on unseen data.

### Why do we need it?

If we evaluate a model using the same data it learned from, we may get an overly optimistic understanding of its performance.

Testing on unseen records provides a better indication of its generalization ability.

### Our Dataset Split

| Dataset | Percentage | Records |
|---|---:|---:|
| Training Data | 80% | 436 |
| Testing Data | 20% | 109 |
| Total | 100% | 545 |

### Implementation

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

**Understanding the parameters:**

- `X`: Input features.
- `y`: Target variable.
- `test_size=0.2`: Allocates 20% of records for testing.
- `random_state=42`: Makes the random split reproducible.

---

## 🔄 11. Machine Learning Pipeline

### What is a Pipeline?

A Pipeline combines multiple preprocessing and Machine Learning steps into one sequential workflow.

Instead of manually transforming the data and then training a model, we can combine the complete process.

Our pipeline contains:

1. Data preprocessing using ColumnTransformer.
2. Model training using LinearRegression.

### Implementation

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)
```

### How does the Pipeline work?

**During training:**

1. Receives the training dataset.
2. Standardizes numerical features.
3. One-Hot Encodes categorical features.
4. Combines the transformed features.
5. Trains the Linear Regression model.

**During prediction:**

1. Receives new input data.
2. Applies the same fitted preprocessing transformations.
3. Passes transformed data to the trained model.
4. Returns the predicted house price.

### Why use a Pipeline?

- Keeps preprocessing and training organized.
- Prevents inconsistent transformations.
- Reduces the possibility of data leakage.
- Makes the model easier to reuse.
- Simplifies deployment and prediction.

---

## 🤖 12. Multiple Linear Regression

### What is Linear Regression?

Linear Regression is a supervised Machine Learning algorithm used to predict continuous numerical values.

For example:
- House prices.
- Salary prediction.
- Sales forecasting.
- Temperature prediction.

### What is Multiple Linear Regression?

Simple Linear Regression uses one independent feature to predict a target.

Multiple Linear Regression uses multiple independent features to predict a target.

In our project, the model uses 12 input features to predict house prices.

### Mathematical Representation

\[
y = \beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_nx_n
\]

Where:

- \(y\) = Predicted house price.
- \(\beta_0\) = Intercept.
- \(\beta_1,\beta_2,\ldots,\beta_n\) = Model coefficients.
- \(x_1,x_2,\ldots,x_n\) = Input features.
- \(n\) = Number of transformed input features.

The model learns the coefficients that minimize the sum of squared prediction errors during ordinary least-squares training.

### Training the Model

```python
model.fit(X_train, y_train)
```

The `fit()` method performs two major operations:

1. Learns the preprocessing transformations from training data.
2. Trains the Linear Regression model using the transformed features.

---

## 🔮 13. Making Predictions

After training, we can predict house prices using unseen testing data.

```python
y_pred = model.predict(X_test)
```

The model processes the test features using the fitted preprocessing steps and generates predicted prices.

We can compare actual and predicted values:

```python
comparison = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

print(comparison.head())
```

This helps us understand how close the predictions are to the actual house prices.

---

## 📈 14. Model Evaluation

Model evaluation helps us understand how accurately our model predicts house prices.

We use four evaluation metrics.

### A. Mean Absolute Error (MAE)

MAE calculates the average absolute difference between actual and predicted values.

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

A lower MAE indicates better prediction accuracy.

### B. Mean Squared Error (MSE)

MSE calculates the average squared difference between actual and predicted values.

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

Squaring the errors gives larger errors more influence.

A lower MSE is better.

### C. Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

\[
RMSE = \sqrt{MSE}
\]

It expresses prediction error in the same unit as the target variable.

A lower RMSE indicates better performance.

### D. R² Score (Coefficient of Determination)

R² measures how much of the variation in the target variable is explained by the model.

\[
R^2 = 1-\frac{SS_{res}}{SS_{tot}}
\]

General interpretation:

- R² = 1: Perfect predictions.
- R² = 0: No improvement over predicting the target mean.
- R² below 0: Worse than predicting the target mean.

### Implementation

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
```

---

## 🏆 15. Model Results

| Evaluation Metric | Result |
|---|---:|
| Algorithm | Multiple Linear Regression |
| Training Records | 436 |
| Testing Records | 109 |
| R² Score | ~0.653 |

**R² Score Interpretation:**

An R² score of approximately 0.653 means that the model explains around 65.3% of the variation in house prices on the test dataset.

This does not mean that every prediction is 65.3% accurate.

Model performance can potentially be improved through feature engineering, outlier analysis, feature selection, and experimenting with other regression algorithms.

---

## 🧠 16. Complete Machine Learning Workflow

```text
             Housing.csv
                  |
                  ▼
          Load Dataset
                  |
                  ▼
       Exploratory Data Analysis
                  |
                  ▼
       Identify Feature Types
                  |
          ┌───────┴────────┐
          ▼                ▼
     Numerical         Categorical
      Features           Features
          |                |
          ▼                ▼
   StandardScaler     OneHotEncoder
          |                |
          └───────┬────────┘
                  ▼
          ColumnTransformer
                  |
                  ▼
            Train-Test Split
                  |
                  ▼
            ML Pipeline
                  |
                  ▼
         Linear Regression
                  |
                  ▼
          Model Prediction
                  |
                  ▼
          Model Evaluation
                  |
                  ▼
            Final Results
```

---

## 🛠️ 17. Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data analysis and manipulation |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning and preprocessing |
| Jupyter Notebook / Google Colab | Development environment |

---

## 🚀 18. How to Run This Project

### Step 1: Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
```

### Step 2: Navigate to the Project Directory

```bash
cd House-Price-Prediction
```

### Step 3: Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Step 4: Open the Notebook

```bash
jupyter notebook
```

### Step 5: Run the Notebook

Open `HousePrediction.ipynb` and execute the cells sequentially.

Make sure `Housing.csv` is available in the expected dataset path.

---

## 📚 19. Key Machine Learning Concepts Learned

Through this project, we explored:

- Supervised Learning
- Regression Problems
- Independent and Dependent Variables
- Numerical and Categorical Data
- Exploratory Data Analysis
- Feature Engineering
- One-Hot Encoding
- Feature Standardization
- Train-Test Split
- Data Leakage
- ColumnTransformer
- Scikit-learn Pipeline
- Multiple Linear Regression
- Model Prediction
- Regression Evaluation Metrics

---

## 🔮 20. Future Improvements

- Experiment with Ridge and Lasso Regression.
- Try Decision Tree and Random Forest Regressors.
- Perform feature selection and feature engineering.
- Analyze and handle outliers.
- Perform cross-validation.
- Tune model hyperparameters.
- Build an interactive house price prediction application.
- Deploy the trained model using Flask or FastAPI.

---

## 🏷️ GitHub Topics

`python` `machine-learning` `supervised-learning` `regression` `linear-regression` `multiple-linear-regression` `house-price-prediction` `pandas` `numpy` `matplotlib` `seaborn` `scikit-learn` `data-preprocessing` `one-hot-encoding` `standardization` `exploratory-data-analysis` `machine-learning-pipeline`

---

## 👨‍💻 Author

**Prajjal Dhar**

⭐ If you find this project helpful, don't forget to star the repository!

