import sys

from pyspark import SparkContext, SparkConf

conf = SparkConf()
sc = SparkContext(conf=conf)

# Read data from text file and split each line into words
words = sc.textFile(sys.argv[1]).flatMap(lambda line: line.split(" "))

# New count the occurrences of each word
wordCounts = words.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

# Finally save the output to another text file
wordCounts.coalesce(1, shuffle=True).saveAsTextFile(sys.argv[2])
sc.stop()
