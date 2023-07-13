package org.db


import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession, functions}

import scala.language.postfixOps

object SparkChallenge extends App {

  System.setProperty("hadoop.home.dir", "C:\\hadoop-3.2.2")

  val spark: SparkSession = SparkSession.builder()
    .master("local[1]")
    .appName("SparkChallenge")
    .getOrCreate()

  val df1 = readAndProcessCSV(spark, "src/main/resources/googleplaystore_user_reviews.csv")

  // Part 1
  val df_1 = df1
    .select(
      col("App").cast("String"),
      col("Sentiment_Polarity").cast("double"))
    .withColumn("Sentiment_Polarity",
      when(col("Sentiment_Polarity").isNaN, 0)
        .otherwise(col("Sentiment_Polarity")))
    .groupBy("App")
    .agg(avg("Sentiment_Polarity").as("Average_Sentiment_Polarity"))
    .sort("App")
  df_1.show()

  val df2 = readAndProcessCSV(spark, "src/main/resources/googleplaystore.csv")

  // Part 2
  val df_2 = df2.select(col("App").cast("string"), col("Rating").cast("double"))
    .filter(col("Rating") >= 4.0 && !col("Rating").isNaN)
    .sort(col("Rating").desc)
  df_2.show()

  df_2.coalesce(1)
    .write
    .option("header", "true")
    .option("delimiter", "§")
    .mode(SaveMode.Overwrite)
    .csv("src/out/csv/")

  // Part 3
  val df2_categories = df2.groupBy(col("App"))
    .agg(collect_set(col("Category")).as("Categories"))

  val df_3 = df2
    .join(df2_categories, Seq("App"), "left")
    .withColumn("row", row_number().over(Window.partitionBy("App").orderBy(col("Reviews").cast("long").desc)))
    .filter(col("row") === 1)
    .drop("row")
    .dropDuplicates("App")
    .filter(!col("Rating").isNaN && !(col("App") === ""))
    .withColumn("Reviews", col("Reviews").cast("long"))
    .withColumn("Size", regexp_replace(col("Size"), "M", "").cast("double"))
    .withColumn("Price", regexp_replace(col("Price").cast("string"), "\\$", ""))
    .withColumn("Price", col("Price").cast("double") * lit(0.9))
    .withColumn("Genres", functions.split(col("Genres"), ";"))
    .withColumn("Last Updated", to_date(col("Last Updated"), "MMMM d, yyyy"))
    .withColumnRenamed("Content Rating", "Content_Rating")
    .withColumnRenamed("Current Ver", "Current_Version")
    .withColumnRenamed("Android Ver", "Minimum_Android_Version")
    .na.fill(0, Seq("Reviews"))
    .drop("Category")
  df_3.show(10, false)

  // Part 4
  val df_part4 = df_3.join(df_1, Seq("App"))
  df_part4.show()

  df_part4
    .write
    .mode(SaveMode.Overwrite)
    .option("compression", "gzip")
    .parquet("src/out/par/")

  val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
  val files = fs.listStatus(new Path("src/out/par/"))

  val parquetFile = files.find(_.getPath.getName.endsWith(".parquet"))
  parquetFile.foreach { file =>
    val originalPath = file.getPath.toString
    val directory = originalPath.substring(0, originalPath.lastIndexOf("/") + 1)
    val renamedPath = directory + "googleplaystore_cleaned.parquet"
    fs.rename(new Path(originalPath), new Path(renamedPath))
  }

  val savedDataframe = spark.read.parquet("src/out/par/")
  savedDataframe.show()

  // Part 5
  val df_4 = df_part4
    .select(explode(
      col("Genres")).as("Genre"),
      col("Rating"),
      col("Average_Sentiment_Polarity"))
    .groupBy("Genre")
    .agg(
      count("Genre").as("Count"),
      avg("Rating").as("Average_Rating"),
      sum("Average_Sentiment_Polarity").as("Average_Sentiment_Polarity")
    )
  df_4.show()


  def readAndProcessCSV(spark: SparkSession, filePath: String): DataFrame = {
    val df = spark.read
      .option("header", value = true)
      .option("quote", "\"")
      .option("escape", "\"")
      .csv(filePath)

    val processedDF = df
      .withColumn("App", regexp_replace(col("App"), "\\++", "PlusPlus"))
      .withColumn("App", trim(regexp_replace(col("App"), "\\W", "")))

    processedDF
  }
}
