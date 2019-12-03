import org.apache.spark.sql.DataFrame

val df: DataFrame = spark
  .read
  .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
  .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
  .csv("/home/p5hngk/Downloads/GitHub/INF_729---Introduction_au_framework_Hadoop/cours-spark-telecom-master/data/train_clean.csv")

println(s"Nombre de lignes : ${df.count}")
println(s"Nombre de colonnes : ${df.columns.length}")

df.show()

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

dfCasted.groupBy("disable_communication").count.orderBy($"count".desc).show(100)

dfCasted.groupBy("country").count.orderBy($"count".desc).show(100)

dfCasted.groupBy("currency").count.orderBy($"count".desc).show(100)

dfCasted.select("deadline").dropDuplicates.show()

dfCasted.groupBy("state_changed_at").count.orderBy($"count".desc).show(100)

dfCasted.groupBy("backers_count").count.orderBy($"count".desc).show(100)

dfCasted.select("goal", "final_status").orderBy($"goal".desc).show(30)

dfCasted.select("goal", "final_status").show(30)

dfCasted.groupBy("country", "currency").count.orderBy($"count".desc).show(50)

val df2: DataFrame = dfCasted.drop("disable_communication")

val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")

df.filter($"country" === "False")
  .groupBy("currency")
  .count
  .orderBy($"count".desc)
  .show(50)

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

dfNoFutur.show(10)

dfCountry.groupBy("final_status").count.orderBy($"count".desc).show(30)

dfCountry.printSchema

val dfCountry2: DataFrame = dfCountry
  .filter($"final_status" === 0 || $"final_status" === 1)

dfCountry2.groupBy("final_status").count.orderBy($"count".desc).show()

val dfCountry3: DataFrame = dfCountry2
    .withColumn("days_campaign", datediff(from_unixtime($"deadline") , from_unixtime($"launched_at")))
    .withColumn("hours_prepa", ((($"launched_at" - $"created_at")/3.6).cast("Int")/1000))

dfCountry3.select("project_id","days_campaign", "hours_prepa", "deadline", "launched_at", "created_at").orderBy($"hours_prepa".asc).show(10)

dfCountry3.groupBy($"hours_prepa" < 0).count.orderBy($"count".desc).show()

def cleanHoursPrepa(created_at: Int, launched_at: Int, hours_prepa: Int): Int = {
  if (hours_prepa < 0)
    launched_at
  else
    created_at
}

val cleanHoursPrepaUdf = udf(cleanHoursPrepa _)

val dfCountry4: DataFrame = dfCountry3
  .withColumn("created_at2", cleanHoursPrepaUdf($"created_at", $"launched_at", $"hours_prepa"))
  .withColumn("hours_prepa", ((($"launched_at" - $"created_at2")/3.6).cast("Int")/1000))
  .drop("created_at")


dfCountry4.groupBy($"hours_prepa" < 0).count.orderBy($"count".desc).show()

dfCountry4.filter($"project_id" === "kkst1677718959").show()

val dfCountry5: DataFrame = dfCountry4
 .drop("launched_at", "created_at2", "deadline")

dfCountry5.show(3)

val dfCountry6: DataFrame = dfCountry5
 .withColumn("name", lower($"name"))
 .withColumn("desc", lower($"desc"))
 .withColumn("keywords", lower($"keywords"))

dfCountry6.show(5)

dfCountry6.withColumn("text", concat_ws(" ", $"name", $"desc", $"keywords")).select($"text").head()

val dfCountry7: DataFrame = dfCountry6
 .withColumn("text", concat_ws(" ", $"name", $"desc", $"keywords"))

dfCountry7.show(3)

dfCountry7.filter("days_campaign is null").count

dfCountry7.filter("days_campaign is null").show(70)

dfCountry7.filter("hours_prepa is null").count

dfCountry7.filter("goal is null").count

dfCountry7.filter("country2 is null").count

dfCountry7.filter("currency2 is null").count

dfCountry7.printSchema

val test : DataFrame = dfCountry7.na.fill(Map("currency2" -> "unknown", "days_campaign" -> -1))  //.filter($"currency2" === "unknown").count
test.filter("days_campaign is null").count
test.filter($"days_campaign" === -1).count

test.printSchema

val dfCountry8: DataFrame = dfCountry7
    .na.fill(Map("currency2" -> "unknown", "country2" -> "unknown", "days_campaign" -> -1, "hours_prepa" -> -1, "goal" -> -1))

dfCountry8.filter($"goal" === -1).count

val monDataFrameFinal: DataFrame = dfCountry8

monDataFrameFinal.write.mode("overwrite").parquet("/home/p5hngk/Downloads/GitHub/INF_729---Introduction_au_framework_Hadoop/cours-spark-telecom-master/monDataFrameFinal")
