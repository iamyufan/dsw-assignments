from pyspark import SparkContext, SparkConf
import sys

def generate_bigrams(text):
    """Generate bigrams from a list of words."""
    return [((text[i], text[i + 1]), 1) for i in range(len(text) - 1)]

if __name__ == "__main__":
    # Check for input arguments
    if len(sys.argv) != 3:
        print("Usage: bigram_model.py <input_file> <output_dir>", file=sys.stderr)
        sys.exit(-1)

    # Create SparkConf and SparkContext
    conf = SparkConf().setAppName("BigramModel")
    sc = SparkContext(conf=conf)

    # Load the text file into an RDD
    lines = sc.textFile(sys.argv[1])

    # Generate bigrams
    bigrams = lines.flatMap(lambda line: generate_bigrams(line.lower().split()))
    
    # Count the occurrences of each bigram
    bigram_counts = bigrams.reduceByKey(lambda a, b: a + b)
    
    # Count the occurrences of each word for the denominator
    word_counts = lines.flatMap(lambda line: line.lower().split()) \
                       .map(lambda word: (word, 1)) \
                       .reduceByKey(lambda a, b: a + b)
    
    # Calculate the probability of each bigram: P(wn|wn-1) = count(wn-1 wn) / count(wn-1)
    # Join word_counts with bigram_counts to calculate probabilities
    bigram_probabilities = bigram_counts.map(lambda x: (x[0][0], (x[0][1], x[1]))) \
                                        .join(word_counts) \
                                        .map(lambda x: (x[1][0][0], x[0], x[1][0][1] / x[1][1]))
    
    # Reformat and save the result
    bigram_probabilities.map(lambda x: f"({x[1]} {x[0]}) = {x[2]}") \
                        .saveAsTextFile(sys.argv[2])

    # Stop the Spark context
    sc.stop()