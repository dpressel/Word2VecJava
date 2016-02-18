package word2vec.lite;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import org.junit.After;
import org.junit.Rule;
import org.junit.rules.ExpectedException;

import com.google.common.annotations.VisibleForTesting;
import word2vec.lite.neuralnetwork.NeuralNetworkType;

/**
 * Tests for {@link Word2VecModel} and related classes.
 * <p>
 * Note that the implementation is expected to be deterministic if numThreads is
 * set to 1
 */
public class Word2VecTest {
	@Rule
	public ExpectedException expected = ExpectedException.none();

	/** Clean up after a test run */
	@After
	public void after() {
		// Unset the interrupted flag to avoid polluting other tests
		Thread.interrupted();
	}

	/** Test {@link NeuralNetworkType#CBOW} */
	/*@Test
	public void testCBOW() throws IOException, TException, InterruptedException {
		assertModelMatches("cbowBasic.model",
				Word2VecModel.trainer()
						.setMinVocabFrequency(6)
						.useNumThreads(1)
						.setWindowSize(8)
						.type(NeuralNetworkType.CBOW)
						.useHierarchicalSoftmax()
						.setLayerSize(25)
						.setDownSamplingRate(1e-3)
						.setNumIterations(1)
						.train(testData())
		);
	}*/

	/** Test {@link NeuralNetworkType#CBOW} with 15 iterations */
	/*@Test
	public void testCBOWwith15Iterations() throws IOException, TException, InterruptedException {
		assertModelMatches("cbowIterations.model",
				Word2VecModel.trainer()
					.setMinVocabFrequency(5)
					.useNumThreads(1)
					.setWindowSize(8)
					.type(NeuralNetworkType.CBOW)
					.useHierarchicalSoftmax()
					.setLayerSize(25)
					.useNegativeSamples(5)
					.setDownSamplingRate(1e-3)
					.setNumIterations(15)
					.train(testData())
			);
	}*/

	/** Test {@link NeuralNetworkType#SKIP_GRAM} */
	/*@Test
	public void testSkipGram() throws IOException, TException, InterruptedException {
		assertModelMatches("skipGramBasic.model",
				Word2VecModel.trainer()
					.setMinVocabFrequency(6)
					.useNumThreads(1)
					.setWindowSize(8)
					.type(NeuralNetworkType.SKIP_GRAM)
					.useHierarchicalSoftmax()
					.setLayerSize(25)
					.setDownSamplingRate(1e-3)
					.setNumIterations(1)
					.train(testData())
			);
	}*/

	/** Test {@link NeuralNetworkType#SKIP_GRAM} with 15 iterations */
	/*@Test
	public void testSkipGramWith15Iterations() throws IOException, TException, InterruptedException {
		assertModelMatches("skipGramIterations.model",
				Word2VecModel.trainer()
					.setMinVocabFrequency(6)
					.useNumThreads(1)
					.setWindowSize(8)
					.type(NeuralNetworkType.SKIP_GRAM)
					.useHierarchicalSoftmax()
					.setLayerSize(25)
					.setDownSamplingRate(1e-3)
					.setNumIterations(15)
					.train(testData())
			);
	}*/

	/** Test that we can interrupt the huffman encoding process */
	/*@Test
	public void testInterruptHuffman() throws IOException, InterruptedException {
		expected.expect(InterruptedException.class);
		trainer()
			.type(NeuralNetworkType.SKIP_GRAM)
			.setNumIterations(15)
			.setListener(new TrainingProgressListener() {
					@Override public void update(Stage stage, double progress) {
						if (stage == Stage.CREATE_HUFFMAN_ENCODING)
							Thread.currentThread().interrupt();
						else if (stage == Stage.TRAIN_NEURAL_NETWORK)
							fail("Should not have reached this stage");
					}
				})
			.train(testData());
	}*/

	/** Test that we can interrupt the neural network training process */
	/*@Test
	public void testInterruptNeuralNetworkTraining() throws InterruptedException, IOException {
		expected.expect(InterruptedException.class);
		trainer()
			.type(NeuralNetworkType.SKIP_GRAM)
			.setNumIterations(15)
			.setListener(new TrainingProgressListener() {
					@Override public void update(Stage stage, double progress) {
						if (stage == Stage.TRAIN_NEURAL_NETWORK)
							Thread.currentThread().interrupt();
					}
				})
			.train(testData());
	}*/

  /**
   * Test the search results are deterministic Note the actual values may not
   * make sense since the model we train isn't tuned
   */
	/*@Test
	public void testSearch() throws InterruptedException, IOException, UnknownWordException {
		Word2VecModel model = trainer()
			.type(NeuralNetworkType.SKIP_GRAM)
			.train(testData());

		List<Match> matches = model.forSearch().getMatches("anarchism", 5);

		assertEquals(
				ImmutableList.of("anarchism", "feminism", "trouble", "left", "capitalism"),
				Lists.transform(matches, Match.TO_WORD)
			);
	}*/

  /**
   * Test that the model can retrieve words by a vector.
   */
  /*@Test
    public void testGetWordByVector() throws InterruptedException, IOException, UnknownWordException {
        Word2VecModel model = trainer()
            .type(NeuralNetworkType.SKIP_GRAM)
            .train(testData());

        // This vector defines the word "anarchism" in the given model.
        float[] vectors = new float[] { 0.11410251703652753f, 0.271180824514185f, 0.03748515103121994f, 0.20888126888511183f, 0.009713531343874777f, 0.4769425625416319f, 0.1431890482445165f, -0.1917578875330224f, -0.33532561802423366f,
            -0.08794543238607992f, 0.20404593606213406f, 0.26170074241479385f, 0.10020961212561065f, 0.11400571893146201f, -0.07846426915175395f, -0.19404092647187385f, 0.13381991303455204f, -4.6749635342694615E-4f, -0.0820905789076496f,
            -0.30157145455251866f, 0.3652037905836543f, -0.16466827556950117f, -0.012965932276668056f, 0.09896568721267748f, -0.01925755122093615f };

        List<Match> matches = model.forSearch().getMatches(vectors, 5);

        assertEquals(
                ImmutableList.of("anarchism", "feminism", "trouble", "left", "capitalism"),
                Lists.transform(matches, Match.TO_WORD)
            );
    }*/
  
  /**
   * Test that the model can retrieve words by a vector.
   */
  /*@Test
    public void testGetWordByNotExistantVector() throws InterruptedException, IOException, UnknownWordException {
        Word2VecModel model = trainer()
            .type(NeuralNetworkType.SKIP_GRAM)
            .train(testData());

        float[] vectors = new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0 };

        List<Match> matches = model.forSearch().getMatches(vectors, 5);

        assertEquals(
                ImmutableList.of("the", "of", "and", "in", "a"),
                Lists.transform(matches, Match.TO_WORD)
            );
    }*/


	/** @return {@link Word2VecTrainer} which by default uses all of the supported features */
	@VisibleForTesting
	public static Word2VecTrainerBuilder trainer() {
		return Word2VecModel.trainer()
			.setMinVocabFrequency(6)
			.useNumThreads(1)
			.setWindowSize(8)
			.type(NeuralNetworkType.CBOW)
			.useHierarchicalSoftmax()
			.setLayerSize(25)
			.setDownSamplingRate(1e-3)
			.setNumIterations(1);
	}

	/** @return raw test dataset. The tokens are separated by newlines. */
	/*@VisibleForTesting
	public static Iterable<List<String>> testData() throws IOException {
		List<String> lines = Common.readResource(Word2VecTest.class, "word2vec.short.txt");
		Iterable<List<String>> partitioned = Iterables.partition(lines, 1000);
		return partitioned;
	}

	private void assertModelMatches(String expectedResource, Word2VecModel model) throws TException {
		final String thrift;
		try {
			thrift = Common.readResourceToStringChecked(getClass(), expectedResource);
		} catch (IOException ioe) {
			String filename = "/tmp/" + expectedResource;
			try {
				FileUtils.writeStringToFile(
						new File(filename),
						ThriftUtils.serializeJson(model.toThrift())
				);
			} catch (IOException e) {
				throw new AssertionError("Could not read resource " + expectedResource + " and could not write expected output to /tmp");
			}
			throw new AssertionError("Could not read resource " + expectedResource + " wrote to " + filename);
		}

		Word2VecModelThrift expected = ThriftUtils.deserializeJson(
				new Word2VecModelThrift(),
				thrift
		);

		assertEquals("Mismatched vocab", expected.getVocab().size(), Iterables.size(model.getVocab()));

		assertEquals(expected, model.toThrift());
	}*/
}
