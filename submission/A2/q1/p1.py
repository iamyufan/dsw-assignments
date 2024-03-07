import sys
from pyspark.sql import SparkSession
import argparse

from pyspark.sql.functions import udf, col, count, when, sum
from pyspark.sql.types import DoubleType, IntegerType


# feel free to def new functions if you need


def create_dataframe(filepath, format, spark):
    """
    Create a spark df given a filepath and format.

    :param filepath: <str>, the filepath
    :param format: <str>, the file format (e.g. "csv" or "json")
    :param spark: <str> the spark session

    :return: the spark df uploaded
    """

    # add your code here
    if format == "csv":
        spark_df = spark.read.format(format).option("header", "true").load(filepath)
    elif format == "json":
        spark_df = spark.read.format("json").option("multiline", "false").load(filepath)

    print()
    print(f"{filepath} | Count: {spark_df.count()}")
    spark_df.printSchema()
    spark_df.show(5)

    return spark_df


def transform_ethnicity_data(MRACBPI2, HISPAN_I):
    """Assign the _IMPRACE value based on MRACBPI2 and HISPAN_I"""
    if MRACBPI2 == None or HISPAN_I == None:
        return 6
    HISPAN_I = int(HISPAN_I)
    MRACBPI2 = int(MRACBPI2)
    # Hispanic
    if HISPAN_I != 12:
        return 5
    # White
    if MRACBPI2 == 1:
        return 1
    # Black/African American
    if MRACBPI2 == 2:
        return 2
    # Indian (American) (includes Eskimo, Aleut)
    if MRACBPI2 == 3:
        return 4
    # Asian
    if MRACBPI2 == 6 or MRACBPI2 == 7 or MRACBPI2 == 12:
        return 3
    # Other race
    if MRACBPI2 == 16 or MRACBPI2 == 17:
        return 6


def transform_age_data(AGE_P):
    """Assign the _AGEG5YR value based on AGE_P"""
    if AGE_P == None:
        return 14
    AGE_P = int(AGE_P)
    if AGE_P >= 18 and AGE_P <= 24:
        return 1
    if AGE_P >= 25 and AGE_P <= 29:
        return 2
    if AGE_P >= 30 and AGE_P <= 34:
        return 3
    if AGE_P >= 35 and AGE_P <= 39:
        return 4
    if AGE_P >= 40 and AGE_P <= 44:
        return 5
    if AGE_P >= 45 and AGE_P <= 49:
        return 6
    if AGE_P >= 50 and AGE_P <= 54:
        return 7
    if AGE_P >= 55 and AGE_P <= 59:
        return 8
    if AGE_P >= 60 and AGE_P <= 64:
        return 9
    if AGE_P >= 65 and AGE_P <= 69:
        return 10
    if AGE_P >= 70 and AGE_P <= 74:
        return 11
    if AGE_P >= 75 and AGE_P <= 79:
        return 12
    if AGE_P >= 80 and AGE_P <= 99:
        return 13
    else:
        return 14


def transform_nhis_data(nhis_df):
    """
    Transform df elements

    :param nhis_df: spark df
    :return: spark df, transformed df
    """

    # add your code here
    # Map MRACBPI2 and HISPAN_I values to _IMPRACE equivalent
    transform_ethnicity_data_udf = udf(transform_ethnicity_data, IntegerType())
    nhis_df = nhis_df.withColumn(
        "_IMPRACE",
        transform_ethnicity_data_udf(nhis_df["MRACBPI2"], nhis_df["HISPAN_I"]),
    )

    # Map AGE_P values to _AGEG5YR equivalent
    transform_age_data_udf = udf(transform_age_data, IntegerType())
    nhis_df = nhis_df.withColumn("_AGEG5YR", transform_age_data_udf(nhis_df["AGE_P"]))

    # Keep only the columns necessary to match BRFSS format and the additional NHIS columns
    transformed_df = nhis_df.select("SEX", "_AGEG5YR", "_IMPRACE", "DIBEV1")

    # Convert columns into double
    transformed_df = (
        transformed_df.withColumn("SEX", col("SEX").cast("double"))
        .withColumn("_AGEG5YR", col("_AGEG5YR").cast("double"))
        .withColumn("_IMPRACE", col("_IMPRACE").cast("double"))
        .withColumn("DIBEV1", col("DIBEV1").cast("double"))
    )

    print()
    print("Transformed NHIS dataframe | Count: ", transformed_df.count())
    transformed_df.printSchema()
    transformed_df.show(5)

    return transformed_df


def _calculate_prevalence(joined_df, group_by_col):
    prevalence_df = (
        joined_df.groupBy(group_by_col)
        .agg(
            count("*").alias("total"),
            sum(when(col("DIBEV1") == 1.0, 1).otherwise(0)).alias("diabetes_yes"),
        )
        .withColumn(
            "prevalence", (col("diabetes_yes") / col("total") * 100).cast(DoubleType())
        )
    )

    # Order the results for better readability
    prevalence_df = prevalence_df.orderBy(group_by_col)

    # Show the result
    prevalence_df.show(truncate=False)


def calculate_statistics(joined_df):
    """
    Calculate prevalence statistics

    :param joined_df: the joined df

    :return: None
    """

    # add your code here
    for group_by_col in ["_IMPRACE", "SEX", "_AGEG5YR"]:
        print(f"Prevalence statistics for {group_by_col}")
        _calculate_prevalence(joined_df, group_by_col)
        print()


def join_data(brfss_df, nhis_df):
    """
    Join dataframes

    :param brfss_df: spark df
    :param nhis_df: spark df after transformation
    :return: the joined df

    """
    # add your code here
    joined_df = brfss_df.join(nhis_df, on=["SEX", "_AGEG5YR", "_IMPRACE"], how="inner")

    # Drop any rows with null values
    joined_df = joined_df.na.drop()

    print()
    print("Joined dataframe | Count: ", joined_df.count())
    joined_df.printSchema()
    joined_df.show(5)

    return joined_df


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument("nhis", type=str, default=None, help="brfss filename")
    arg_parser.add_argument("brfss", type=str, default=None, help="nhis filename")
    arg_parser.add_argument(
        "-o", "--output", type=str, default=None, help="output path(optional)"
    )

    # parse args
    args = arg_parser.parse_args()
    if not args.nhis or not args.brfss:
        arg_parser.usage = arg_parser.format_help()
        arg_parser.print_usage()
    else:
        brfss_filename = args.brfss
        nhis_filename = args.nhis

        # Start spark session
        spark = SparkSession.builder.getOrCreate()

        # load dataframes
        print("==== Loading dataframes ====")
        brfss_df = create_dataframe(brfss_filename, "json", spark)
        nhis_df = create_dataframe(nhis_filename, "csv", spark)

        # Perform mapping on nhis dataframe
        print("==== Transforming NHIS dataframe ====")
        nhis_df = transform_nhis_data(nhis_df)

        # nhis_df.write.csv("transformed_nhis", mode='overwrite', header=True)
        # Join brfss and nhis df
        print("==== Joining dataframes ====")
        joined_df = join_data(brfss_df, nhis_df)
        # Calculate statistics
        print("==== Calculating statistics ====")
        calculate_statistics(joined_df)

        # Save
        if args.output:
            print("==== Saving dataframe ====")
            joined_df.write.csv(args.output, mode="overwrite", header=True)
            print(f"Saved to {args.output}")

        # Stop spark session
        spark.stop()
