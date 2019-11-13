import org.apache.spark.sql.DataFrame

val df: DataFrame = spark
  .read
  .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
  .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
  .csv("/home/p5hngk/Downloads/GitHub/INF_729---Introduction_au_framework_Hadoop/cours-spark-telecom-master/data/train_clean.csv")


df.printSchema()

val dfCasted: DataFrame = df
  .withColumn("goal", $"goal".cast("Int"))
  .withColumn("deadline" , $"deadline".cast("Int"))
  .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
  .withColumn("created_at", $"created_at".cast("Int"))
  .withColumn("launched_at", $"launched_at".cast("Int"))
  .withColumn("backers_count", $"backers_count".cast("Int"))
  .withColumn("final_status", $"final_status".cast("Int"))

dfCasted.printSchema()

dfCasted
  .select("goal", "backers_count", "final_status")
  .describe()
  .show

val dfNoFutur: DataFrame = dfCasted
  .drop("disable_communication")
  .drop("backers_count", "state_changed_at")


def cleanCountry(country: String, currency: String): String = {
  if (country == "False")
    currency
  else
    country
}

def cleanCurrency(currency: String): String = {
  if (currency != null && currency.length != 3)
    null
  else
    currency
}

val cleanCountryUdf = udf(cleanCountry _)
val cleanCurrencyUdf = udf(cleanCurrency _)

val dfCountry: DataFrame = dfNoFutur
  .withColumn("country2", cleanCountryUdf($"country", $"currency"))
  .withColumn("currency2", cleanCurrencyUdf($"currency"))
  .drop("country", "currency")
  .filter($"final_status" === 0 || $"final_status" === 1)
  .withColumn("days_campaign", datediff(from_unixtime($"deadline") , from_unixtime($"launched_at")))
  .withColumn("hours_prepa", ((($"launched_at" - $"created_at")/3.6).cast("Int")/1000))


dfCountry.printSchema

    
def cleanHoursPrepa(created_at: Int, launched_at: Int, hours_prepa: Int): Int = {
  if (hours_prepa < 0)
    launched_at
  else
    created_at
}

val cleanHoursPrepaUdf = udf(cleanHoursPrepa _)

val dfCountry2: DataFrame = dfCountry
  .withColumn("created_at2", cleanHoursPrepaUdf($"created_at", $"launched_at", $"hours_prepa"))
  .withColumn("hours_prepa", ((($"launched_at" - $"created_at2")/3.6).cast("Int")/1000))
  .drop("created_at")
  .drop("launched_at", "created_at2", "deadline")
  .withColumn("name", lower($"name"))
  .withColumn("desc", lower($"desc"))
  .withColumn("keywords", lower($"keywords"))
  .withColumn("text", concat_ws(" ", $"name", $"desc", $"keywords"))
  .withColumn("days_campaign", when($"days_campaign".isNull, -1).otherwise("days_campaign"))
  .withColumn("hours_prepa", when($"hours_prepa".isNull, -1).otherwise("hours_prepa"))
  .withColumn("goal", when($"goal".isNull, -1).otherwise("goal"))
  .withColumn("country2", when($"country2".isNull, "unknown").otherwise("country2"))
  .withColumn("currency2", when($"currency2".isNull, "unknown").otherwise("currency2"))


val monDataFrameFinal: DataFrame = dfCountry2

monDataFrameFinal.write.parquet("/home/p5hngk/Downloads/GitHub/INF_729---Introduction_au_framework_Hadoop/cours-spark-telecom-master/monDataFrameFinal")
