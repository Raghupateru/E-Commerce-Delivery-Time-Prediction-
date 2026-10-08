### README.md

```markdown
# E-Commerce Delivery Time Prediction Using PySpark

## Project Overview

This project uses Apache Spark (PySpark) and Linear Regression to predict e-commerce delivery time based on order-related features.

The project workflow includes:

- Loading an e-commerce CSV dataset into a Spark DataFrame
- Creating derived features
- Preparing numeric features using VectorAssembler
- Splitting the data into training and testing sets
- Training a PySpark Linear Regression model
- Evaluating predictions using RMSE and MAE
- Converting Spark predictions to Pandas
- Visualizing actual vs. predicted delivery time
- Visualizing delivery-time distributions
- Comparing actual and predicted values for the first 100 orders
- Classifying predictions as "On Time" or "Delayed" for visualization

## Technologies Used

- Python
- Apache Spark / PySpark
- Pandas
- NumPy
- Matplotlib

## Dataset

The project uses an e-commerce dataset containing fields such as:

- Warehouse_block
- Mode_of_Shipment
- Customer_care_calls
- Customer_rating
- Cost_of_the_Product
- Prior_purchases
- Product_importance
- Gender
- Discount_offered
- Weight_in_gms

The source code reads the dataset as `E_Commerce.csv`.

## Project Workflow

1. Start a Spark session.
2. Load `E_Commerce.csv` into a Spark DataFrame.
3. Create a `Distance` feature from `Cost_of_the_Product`.
4. Create a `Delivery_Time` target using:
   - Weight_in_gms
   - Customer_care_calls
   - A random component
5. Select Weight, Distance, and Delivery_Time.
6. Use VectorAssembler to create the features vector.
7. Split the data into 80% training and 20% testing data using seed 42.
8. Train a PySpark Linear Regression model.
9. Generate predictions on the test set.
10. Evaluate the model using:
    - RMSE
    - MAE
11. Convert predictions to Pandas.
12. Generate visualizations.

## Model

The implementation uses PySpark Linear Regression to predict continuous `Delivery_Time`.

```python
from pyspark.ml.regression import LinearRegression
```

The task description in the report also mentions Logistic Regression for predicting the binary `Reached.on.Time_Y.N` target. However, the provided source code implements Linear Regression for predicting continuous delivery time.

This README documents the actual implementation in the source code.

## Evaluation Metrics

The model is evaluated using:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

RMSE measures the square root of the average squared prediction error.

MAE measures the average absolute difference between actual and predicted delivery time.

## Visualizations

The project generates the following visualizations:

1. Actual vs Predicted Delivery Time Scatter Plot
2. Actual Delivery Time Distribution
3. Predicted Delivery Time Distribution
4. Actual vs Predicted Delivery Time for the First 100 Orders
5. On Time vs Delayed Prediction Outcome

The final visualization in the report shows approximately:

- On Time: 50.1%
- Delayed: 49.9%

## Installation

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Project Structure

```text
ecommerce_delivery_prediction/
│
├── E_Commerce.csv
├── README.md
├── requirements.txt
└── your_script.py
```

## Dataset Setup

Place `E_Commerce.csv` in the project directory.

The CSV can be loaded using:

```python
df = spark.read.csv(
    "E_Commerce.csv",
    header=True,
    inferSchema=True
)
```

## Running the Project

Run the Python script using:

```bash
python your_script.py
```

Or use Spark submit:

```bash
spark-submit your_script.py
```

## Expected Output

The program prints the RMSE and MAE values and displays the generated Matplotlib charts.

The prediction results are also converted into a Pandas DataFrame for visualization and analysis.

## Important Note

The report contains a difference between the task description and the provided source code.

The task description specifies Logistic Regression for predicting whether a delivery is on time.

The provided source code uses Linear Regression to predict continuous delivery time.

Therefore, this README documents the actual Linear Regression implementation.

If the objective is specifically to predict the binary `Reached.on.Time_Y.N` target, the implementation should use StringIndexer, VectorAssembler, LogisticRegression, and classification metrics such as Accuracy and F1 Score.
```

### requirements.txt

```text
pyspark
pandas
numpy
matplotlib
```
