from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, expr
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import pandas as pd

spark = SparkSession.builder.appName("EcommerceDeliveryPrediction").getOrCreate()
df = spark.read.csv(r"C:\Users\pater\Desktop\E_Commerce.csv", header=True, inferSchema=True)
df = df.withColumn("Distance", col("Cost_of_the_Product") * 0.5)
df = df.withColumn("Delivery_Time",
   (col("Weight_in_gms") / 1000) * 0.8
+ col("Customer_care_calls") * 1.2
+ (rand() * 5))numeric_df = df.select(
 col("Weight_in_gms").alias("Weight"),
 col("Distance"),
 col("Delivery_Time")
)
assembler = VectorAssembler(
 inputCols=["Weight", "Distance"],
 outputCol="features"
)
final_df = assembler.transform(numeric_df).select("features", "Delivery_Time")
train, test = final_df.randomSplit([0.8, 0.2], seed=42)
lr = LinearRegression(featuresCol="features", labelCol="Delivery_Time")
model = lr.fit(train)
pred = model.transform(test)
evaluator_rmse = RegressionEvaluator(labelCol="Delivery_Time", predictionCol="prediction", 
metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="Delivery_Time", predictionCol="prediction", 
metricName="mae")
print("RMSE:", evaluator_rmse.evaluate(pred))
print("MAE:", evaluator_mae.evaluate(pred))
pdf = pred.select("Delivery_Time", "prediction").toPandas()
plt.figure(figsize=(7,5))
plt.scatter(pdf["Delivery_Time"], pdf["prediction"])
plt.xlabel("Actual Delivery Time")
plt.ylabel("Predicted Delivery Time")
plt.title("Actual vs Predicted Delivery Time")
plt.show()
plt.figure(figsize=(7,5))
plt.hist(pdf["Delivery_Time"], bins=20)
plt.title("Actual Delivery Time Distribution")
plt.show()
plt.figure(figsize=(7,5))
plt.hist(pdf["prediction"], bins=20)
plt.title("Predicted Delivery Time Distribution")
plt.show()
plt.figure(figsize=(7,5))
plt.plot(pdf["Delivery_Time"][:100], label="Actual")
plt.plot(pdf["prediction"][:100], label="Predicted")
plt.legend()
plt.title("Actual vs Predicted (Sample 100 Orders)")
plt.show()pdf["Status"] = pdf.apply(
 lambda x: "On Time" if x["prediction"] <= x["Delivery_Time"] else "Delayed",
 axis=1
)
status_counts = pdf["Status"].value_counts()
plt.figure(figsize=(5,5))
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Prediction Outcome: On Time vs Delayed")
plt.show()
