import sys
import re

from pyspark import SparkContext, SparkConf


def preprocess_line(line):
    """Give a line of text, remove non-alphanumeric characters and convert to lowercase.

    Args:
        line (str): A string of text.

    returns:
        list: A list of words.
    """
    line = line.lower()
    line = re.sub(r"[^a-z0-9\s]", "", line)
    words = line.split()
    return words


def convert_to_bigram(words):
    """Given a list of words, convert into a list of bigrams.

    Args:
        words (list): A list of words.

    returns:
        list: A list of bigrams.
    """
    bigrams = []
    for i in range(len(words) - 1):
        bigrams.append((words[i], words[i + 1]))
    return bigrams


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: bigram <input> <output>", file=sys.stderr)
        sys.exit(-1)

    conf = SparkConf()
    sc = SparkContext(conf=conf)

    # Read data from text file and preprocess each line into words
    words = sc.textFile(sys.argv[1]).flatMap(preprocess_line)

    # Generate bigrams
    bigrams = words.mapPartitions(lambda words: convert_to_bigram(list(words)))

    # Count the occurrences of each bigram
    bigramCounts = bigrams.map(lambda bigram: (bigram, 1)).reduceByKey(
        lambda a, b: a + b
    )

    bigramCountsForPrint = bigramCounts.map(lambda x: (x[0][0] + " " + x[0][1], x[1]))

    # Save the bigram counts to another text file
    bigramCountsForPrint.coalesce(1, shuffle=True).saveAsTextFile(
        sys.argv[2] + "_bigram_counts"
    )

    # Count the occurrences of each word
    wordCounts = words.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

    # bigramCounts: ((w1, w2), count) -> (w1, (w2, count))
    bigramCountsAdjusted = bigramCounts.map(lambda x: (x[0][0], (x[0][1], x[1])))

    # wordCounts: (w, count) -> (w, (w, count))
    wordCountsAdjusted = wordCounts.map(lambda x: (x[0], (x[0], x[1])))

    # Join the two RDDs
    joined = bigramCountsAdjusted.join(wordCountsAdjusted)

    # Compute the conditional probability
    bigramProbabilities = joined.map(
        lambda x: (x[0] + " " + x[1][0][0], x[1][0][1] / x[1][1][1])
    )

    # Save the output to another text file
    bigramProbabilities.coalesce(1, shuffle=True).saveAsTextFile(
        sys.argv[2] + "_bigram_probabilities"
    )

    sc.stop()
