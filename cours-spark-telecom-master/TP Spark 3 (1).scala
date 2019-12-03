import org.apache.spark.sql.DataFrame

val df: DataFrame = spark
  .read
  .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
  .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
  .parquet("/home/p5hngk/Downloads/GitHub/INF_729---Introduction_au_framework_Hadoop/cours-spark-telecom-master/monDataFrameFinal") //data/prepared_trainingset")  monDataFrameFinal

println("Training Dataframe")
df.show()

df.printSchema

import org.apache.spark.ml.feature.{CountVectorizer, IDF, OneHotEncoderEstimator, RegexTokenizer, StringIndexer}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.StopWordsRemover

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}

val tokenizer = new RegexTokenizer()
  .setPattern("\\W+")
  .setGaps(true)
  .setInputCol("text")
  .setOutputCol("tokens")

val stopWordsRemover = new StopWordsRemover()
  .setInputCol("tokens")
  .setOutputCol("filtered")

val countVectorizedModel = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("vectorized")

val idf = new IDF()
      .setInputCol("vectorized")
      .setOutputCol("tfidf")

val stringIndexer = new StringIndexer()
    .setInputCol("country2")
    .setOutputCol("country_indexed")
    .setHandleInvalid("skip")

val stringIndexer2 = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("skip")

val oneHotEncoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))

val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

println("OUTPUT FEATURES")

val lr = new LogisticRegression()
  .setElasticNetParam(0.0)
  .setFitIntercept(true)
  .setFeaturesCol("features")
  .setLabelCol("final_status")
  .setStandardization(true)
  .setPredictionCol("predictions")
  .setRawPredictionCol("raw_predictions")
  .setThresholds(Array(0.7, 0.3))
  .setTol(1.0e-6)
  .setMaxIter(20)

val stages10 = Array(tokenizer, stopWordsRemover, countVectorizedModel, idf, stringIndexer, stringIndexer2, oneHotEncoder, vectorAssembler, lr)
    val pipeline = new Pipeline().setStages(stages10)

val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = 1991)

val model = pipeline.fit(training)
println(s"Model 1 was fit using parameters: ${model.parent.extractParamMap}")

val dfWithSimplePredictions = model.transform(test)

dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()

val evaluator = new MulticlassClassificationEvaluator()
    .setMetricName("f1")
    .setLabelCol("final_status")
    .setPredictionCol("predictions")

val f1score = evaluator.evaluate(dfWithSimplePredictions)

println("Le f1-score est de " + f1score)

val paramGrid = new ParamGridBuilder()
    .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
    .addGrid(countVectorizedModel.minDF, Array(55.0, 75.0, 95.0))
    .build()

//  TrainValidationSplit requiert un estimateur, un set d'estimateur ParamMaps, et un Evaluator.
val trainValidationSplit = new TrainValidationSplit()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setTrainRatio(0.7)

// Entrainement du modèle avec l'échantillon training
println("Entrainement du modèle avec l'échantillon training")
val validationModel = trainValidationSplit.fit(training)

val dfWithPredictions = validationModel.transform(test).select("features","final_status","predictions")

dfWithPredictions.groupBy("final_status", "predictions").count.show()

val score = evaluator.evaluate(dfWithPredictions)

dfWithPredictions.groupBy("final_status","predictions").count.show()

println("F1 Score est " + score)

// Saving model

validationModel.save("/home/p5hngk/Downloads/GitHub/INF_729---Introduction_au_framework_Hadoop/cours-spark-telecom-master/model/LogisticRegression2")

// Evaluer la precision (accuracy)
    val evaluator_acc = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("accuracy")

    // obtention de la mesure de performance
    val accuracy = evaluator_acc.evaluate(dfWithPredictions)
    println("Precision obtenue : " + accuracy)


