from pyspark.sql import SparkSession
from model import *
import os
import sys


def main():

    # Set the environment variables
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["SPARK_HOME"] = r"spark-3.3.1-bin-hadoop2"
    os.environ["HADOOP_HOME"] = r"spark-3.3.1-bin-hadoop2"
    os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

    # Create a SparkSession
    spark = SparkSession.builder.getOrCreate()

    # Load the dataframe
    df = spark.createDataFrame()

    # Build your pipeline with multiple entries and export as onnx
    build_model_pipeline(df, 'target', spark)


if __name__ == '__main__':
    main()
