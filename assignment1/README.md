# Assignment 1: Programming in Spark

Student: [**Yufan Zhang**](https://yufanbruce.com/) (NetID: yz2894)

---

## How to run the code

### Step 4: Word Count

```bash
spark-submit ./wordcount.py ./wiki.txt ./output
```

The results are stored in the `output` directory.

### Step 5: Bigrams

```bash
spark-submit ./bigram.py ./wiki.txt ./output
```

The count of all bigram words are stored in the `output_bigram_counts` directory.

The conditional probability of all bigram words are stored in the `output_bigram_probabilities` directory.

## Interpretation of the results in Step 5

The output of the PySpark application consists of two parts: the count of all bigrams and the conditional bigram frequency distribution, each saved to separate text files.

1. **Bigram Counts Result:** This result lists each unique bigram found in the document along with its count. A bigram is a pair of consecutive words, and the count indicates how many times that pair appears in the text. For example, the entry `('and so', 1097)` means the bigram "and so" occurs 1,097 times in the document.

2. **Bigram Probabilities Result:** This result shows the conditional probability of encountering the second word of a bigram given the first word. Each entry is formatted as `('first_word second_word', probability)`, where the probability is a decimal number representing the likelihood of the second word following the first in the text. For instance, the entry `('defend the', 0.2621359223300971)` indicates that, given the word "defend," there's approximately a 26.21% chance the next word is "the." This data helps understand the contextual relationships between words in the document.

## Google drive links to the results

- [All results (as described above)](https://drive.google.com/drive/folders/1r5o5u-CkSWyAm1bwfch_x7k1YvJQTIwL?usp=sharing)
- [Word count results (output)](https://drive.google.com/drive/folders/1piRiWBt0OyRQ67kxpRlArX6FiCeCFxCR?usp=drive_link)
- [Count of all bigram words (output_bigram_counts)](https://drive.google.com/drive/folders/1mqWaCQlFczskhSd1n1-jO0kCd9C6YRCg?usp=drive_link)
- [Conditional probability of all bigram words (output_bigram_probabilities)](https://drive.google.com/drive/folders/1yVr82wGLC8fEkrRfACk_TItqnlRKr-VX?usp=drive_link)

## Python implementation of Step 5: Bigrams

The file can be found as `bigram.py` in the submission.

```python
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
```
