package com.apps.binary.classificator

class KnnClassifier(kNN: Int, trains: Array[(Array[Double], Int)], tests: Array[Array[Double]]) extends Serializable {


  def kNearestNeighbors: Array[(Array[Double], Int)] = {
    val testRdd = SparkInstance.sc.parallelize(tests)
    val res = testRdd.map { testLine =>
      val disort = {
        for (t <- trains.indices) yield { (distance(testLine, trains(t)._1), trains(t)._2) }
      }.sortBy(_._1)

      val partdisort = disort.take(kNN)
      val highest = highestMultipleFrequency(partdisort)
      highest match {
        case x: Some[(Int, Int)] =>
          (testLine, x.get._2)
        case _ =>
          (testLine, -1)
      }
    }
    res.collect()
  }

  def highestMultipleFrequency[T](items: IndexedSeq[T]): Option[T] = {
    type Frequencies = Map[T, Int]
    type Frequency = Pair[T, Int]

    def freq(acc: Frequencies, item: T) = acc.contains(item) match {
      case true => acc + Pair(item, acc(item) + 1)
      case _ => acc + Pair(item, 1)
    }
    def mostFrequent(acc: Option[Frequency], item: Frequency) = acc match {
      case None if item._2 >= 0 => Some(item)
      case Some((value, count)) if item._2 > count => Some(item)
      case _ => acc
    }
    items.foldLeft(Map[T, Int]())(freq).foldLeft[Option[Frequency]](None)(mostFrequent) match {
      case Some((value, count)) => Some(value)
      case _ => None
    }
  }

  def distance(a: Array[Double], b: Array[Double]): Int = {
    var sum = 0
    for (i <- a.indices) {
      if (a(i) != b(i)) {
        sum += 1
      }
    }
    sum
  }

}
