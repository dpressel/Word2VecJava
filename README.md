# Word2vecJava (Lite)

This is a lightweight-oriented fork of the [Medallia implementation of Word2Vec in Java](https://github.com/medallia/Word2VecJava) with several aims in mind.  While the original base is fairly elegant, there were several things that made it impractical to use on larger corpora.

* Efficiency
  * I removed a copy and data structure mod from Neural Net model to Word2VecModel.  This copy could be perilous with huge models, and the data structure in Word2VecModel was problematic for Java WRT large files, and all of this is easy to avoid by not copying
    * Code now assumes that the model should never be copied (keep in mind, models will often be HUGE)
  * Now uses floats, not doubles like the original Word2Vec code, which should mean bigger models and faster computations
  * I removed the NormalizedWord2Vec model, and simply added a normalize() method on the base which is public.  This prevents a second copy that was happening when building a Searcher

* Better Interop with Mikolov's original models/Code simplicity
  * Use original code's file format for read and write, handle large files!
    * There is an outstanding pull request from another developer since mid-year last year, that aimed to fix a problem with reading in large word2vec files from binary.  Without this, the code is broken, though it worked in earlier versions
    * To get the ability to write Word2Vec binary files, upgrading to the broken binary loader was required as the raw data structures arent easily accessible to add ones own serialization
  * Can now easily access raw underlying embedding matrix from user code
  
* Dependencies
  * Removed Thrift and Joda deps from the project (all files read and written now in format compatible with Mikolov's code only).
    * However, now that data structures are more accessible, should be easy to add a layer to do Thrift, or any other serialization layer without impacting the base

* Caveats to this Fork

 * For the time-being, GLoVe file support is gone, though I will look at adding this without any deps in the future -- very convenient functionality
 * Tests still being reworked, as I need to do a bit more validation against the original

# Notes from the original Java [code](https://github.com/medallia/Word2VecJava):

For more background information about word2vec and neural network training for the vector representation of words, please see the following papers.
* http://ttic.uchicago.edu/~haotang/speech/1301.3781.pdf
* http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf

For comprehensive explanation of the training process (the gradiant descent formula calculation in the back propagation training), please see:
* http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf

Note that this isn't a completely faithful rewrite, specifically: 

### When building the vocabulary from the training file:
1. The original version does a reduction step when learning the vocabulary from the file when the vocab size hits 21 million words, removing any words that do not meet the minimum frequency threshold. This Java port has no such reduction step.
2. The original version injects a </s> token into the vocabulary (with a word count of 0) as a substitute for newlines in the input file. This Java port's vocabulary excludes the token.
3. The original version does a quicksort which is not stable, so vocabulary terms with the same frequency may be ordered non-deterministically.  The Java port does an explicit sort first by frequency, then by the token's lexicographical ordering.

### In partitioning the file for processing
1. The original version assumes that sentences are delimited by newline characters and injects a sentence boundary per 1000 non-filtered tokens, i.e. valid token by the vocabulary and not removed by the randomized sampling process. Java port mimics this behavior for now ...
2. When the original version encounters an empty line in the input file, it re-processes the first word of the last non-empty line with a sentence length of 0 and updates the random value. Java port omits this behavior.

### In the sampling function
1. The original C documentation indicates that the range should be between 0 and 1e-5, but the default value is 1e-3. This Java port retains that confusing information.
2. The random value generated for comparison to determine if a token should be filtered uses a float. This Java port uses double precision for twice the fun.

### In the distance function to find the nearest matches to a target query
1. The original version includes an unnecessary normalization of the vector for the input query which may lead to tiny inaccuracies. This Java port foregoes this superfluous operation.
2. The original version has an O(n * k) algorithm for finding top matches and is hardcoded to 40 matches. This Java port uses Google's lovely com.google.common.collect.Ordering.greatestOf(java.util.Iterator, int) which is O(n + k log k) and takes in arbitrary k.

Note: The k-means clustering option is excluded in the Java port

Please do not hesitate to peek at the source code. It should be readable, concise, and correct. Please feel free to reach out if it is not.

## Building the Project
To verify that the project is building correctly, run 
```bash
./gradlew build && ./gradlew test
```

It should run 7 tests without any error.

Note: this project requires gradle 2.2+, if you are using older version of gradle, please upgrade it and run:
```bash
./gradlew clean test
```

to have a clean build and re-run the tests.


## Contact
Andrew Ko (wko@medallia.com)
