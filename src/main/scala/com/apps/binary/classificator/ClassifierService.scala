package com.apps.binary.classificator

import java.io.{File, PrintWriter}

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, GBTClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import SparkInstance._
import SparkInstance.sqlContext.implicits._

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.tree.{GradientBoostedTrees, RandomForest}

object ClassifierService {

  def main(args: Array[String]) = {
    println("Start classifier")
    val train = readTrainData()

    //crossValidation(train)

    val test = readTestData()
    /*val knnClassifier = new KnnClassifier(6, train, test)
    val result = knnClassifier.kNearestNeighbors*/


    val trainRdd = sc.parallelize(train).map {x =>
      new LabeledPoint(x._2.toDouble, Vectors.dense(x._1))
    }
    val testRdd = sc.parallelize(test).map {x =>
      new LabeledPoint(0.0, Vectors.dense(x))
    }

    val result = randomForest(trainRdd, testRdd)
    writeLabelsToFile(result.map(x => x._2.toInt))

    println("Stop classifier")

  }

  def crossValidation(train: Array[(Array[Double], Int)]) = {
    val knnCvError = getCvErrorKnn(train, 6)
    println("KNN classifier: K = " + knnCvError._1._1 + ", CVerror = " + knnCvError._1._2)

    val trainRdd = SparkInstance.sc.parallelize(train).map {x =>
      new LabeledPoint(x._2.toDouble, Vectors.dense(x._1))
    }
    val splitData = trainRdd.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splitData(0), splitData(1))

    val trainDF = trainingData.map(l => Record(l.label.toString, l.features)).toDF("label", "features")
    val testDF = testData.map(l => Record(l.label.toString, l.features)).toDF("label", "features")

    val rfModel = getCvErrorRf(trainDF)
    println("Random forest")
    val rfRes = rfModel.transform(testDF).foreach(println)

    /*val gbtModel = getCvErrorGbt(trainDF)
    println("\n\nGBT")
    val gbtRes = gbtModel.transform(testDF).foreach(println)

    val svmModel = getCvErrorSvm(trainDF)
    println("\n\nSVM")
    val svmRes = svmModel.transform(testDF).foreach(println)*/

    knnCvError
  }

  def getCvErrorRf(trainingData: DataFrame) = {
    val numTrees: Int = 3
    val nFolds: Int = 6

    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(numTrees)

    val pipeline = new Pipeline().setStages(Array(rf))

    val paramGrid = new ParamGridBuilder().build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("precision")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(nFolds)

    cv.fit(trainingData)
  }

  def randomForest(trainingData: RDD[LabeledPoint], testData: RDD[LabeledPoint]) = {
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 3
    val featureSubsetStrategy = "auto"
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32

    val rfModel = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    val labelAndPreds = testData.map { point =>
      val prediction = rfModel.predict(point.features)
      (point.label, prediction)
    }
    labelAndPreds.collect()
  }

  def getCvErrorGbt(trainingData: DataFrame) = {
    val maxDepth: Int = 4
    val nFolds: Int = 6

    val gbt = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxDepth(maxDepth)

    val pipeline = new Pipeline().setStages(Array(gbt))

    val paramGrid = new ParamGridBuilder().build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("precision")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(nFolds)

    cv.fit(trainingData)
  }

  def gradBoostTrees(trainingData: RDD[LabeledPoint], testData: RDD[LabeledPoint]) = {
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.numIterations = 3
    boostingStrategy.treeStrategy.numClasses = 2
    boostingStrategy.treeStrategy.maxDepth = 5
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

    val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    labelAndPreds.collect()
  }

  def getCvErrorSvm(trainingData: DataFrame) = {
    val nFolds: Int = 6

    val svm = new LogisticRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")

    val pipeline = new Pipeline().setStages(Array(svm))

    val paramGrid = new ParamGridBuilder().build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("precision")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(nFolds)

    cv.fit(trainingData)
  }

  def svm(trainingData: RDD[LabeledPoint], testData: RDD[LabeledPoint]) = {
    val numIterations = 100
    val model = SVMWithSGD.train(trainingData, numIterations)

    model.clearThreshold()

    val scoreAndLabels = testData.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    scoreAndLabels.collect()
  }

  def getCvErrorKnn(train: Array[(Array[Double], Int)], k: Int) ={
    val cvErrors = ArrayBuffer[(Int, Double)]()
    val cvTrainTestInc = ArrayBuffer[(Array[Int], Array[Int])]()
    val kLenght: Int = train.length/k
    var cTrain = Array[(Array[Double], Int)]()
    var cTest = Array[(Array[Double], Int)]()
    val accsKnn = ArrayBuffer[Double]() //на каждом этапе будем сохранять точность
    for (i <- 0 to k - 1) {
        //промежуток длины kLenght берется в качестве тестовой выборки
      val testIndices = (i*kLenght to (i+1)*kLenght-1).toArray
      cTest = for (f <- testIndices) yield { train(f)}

      var trainIndices = Array[Int]()
      var inc1 = Array[Int]()
      var inc2 = Array[Int]()
      for (j <- 0 to i-1) {
        inc1 = inc1 ++ (j*kLenght to (j+1)*kLenght-1)
      }
      for (j <- i+1 to k-1) {
        inc2 = inc2 ++ (j*kLenght to (j+1)*kLenght-1)
      }
      trainIndices = inc1 ++ inc2

      val c = for (f <- trainIndices) yield { train(f)}
      cTrain = cTest ++ c

      val cvKnnClassifier = new KnnClassifier(k, cTrain, cTest.map(x => x._1))
      val knnRes = cvKnnClassifier.kNearestNeighbors
      accsKnn.append(calculateAccurace(knnRes.map(x => x._2), cTest.map(x => x._2)))
      cvTrainTestInc.append((trainIndices, testIndices))
    }

    val cvError = getVariance(accsKnn.toArray)
    ((k, cvError), cvTrainTestInc.toArray)
  }

  def getVariance(vals: Array[Double]): Double = {
    val m = vals.sum/vals.length
    val d = vals.map(x => Math.pow(x - m, 2))
    d.sum/(vals.length - 1)
  }

  def calculateAccurace(res: Array[Int], labels: Array[Int]): Double = {
    var sum: Double = 0
    for (i <- res.indices) {
      if (res(i) == labels(i)) {
        sum += 1
      }
    }
    sum
  }


  def readTrainData() = {
    val absPath = new File(".").getAbsolutePath
    val trainDataFile = new File(absPath.substring(0, absPath.length-1) + "/data/train/train.txt")
    val trainLines = SparkInstance.sc.textFile(trainDataFile.getAbsolutePath).map(_.toString.split(" ")).map { x =>
      x.map(y => y.toDouble)
    }.collect()

    val lblFile = new File(absPath.substring(0, absPath.length-1) + "/data/train_labels.txt")
    val labels = SparkInstance.sc.textFile(lblFile.getAbsolutePath).map(_.toString).filter(s => s.indexOf("\n") < 0).collect()

    val train = ArrayBuffer[(Array[Double], Int)]()
    var i = 0
    for (line <- trainLines) {
      train.append((line, labels(i).toInt))
      i += 1
    }
    train.toArray
  }

  def readTestData() = {
    val absPath = new File(".").getAbsolutePath
    val trainDataFile = new File(absPath.substring(0, absPath.length-1) + "/data/test.txt")
    val test = SparkInstance.sc.textFile(trainDataFile.getAbsolutePath).map(_.toString.split(" ")).map(x => x.map(y => y.toDouble)).collect()
    test.toArray
  }

  def writeLabelsToFile(results: Array[Int]) = {
    val absPath = new File(".").getAbsolutePath
    val outFile = new File(absPath.substring(0, absPath.length-1) + "/data/test_labels.txt")
    val pw = new PrintWriter(outFile)
    for (res <- results) {
      pw.write(res.toString + "\n")
    }
    pw.close()

  }

}

case class Record(category: String, features: Vector)
