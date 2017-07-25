package com.apps.binary.classificator

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

object SparkInstance {
  val conf = new SparkConf().setAppName("Binary Classificator").setMaster("local[*]")
  val sc: SparkContext = new SparkContext(conf)
  val sqlContext: SQLContext = SQLContext.getOrCreate(sc)
}
