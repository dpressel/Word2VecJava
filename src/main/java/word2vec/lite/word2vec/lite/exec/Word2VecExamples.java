package word2vec.lite.word2vec.lite.exec;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import word2vec.lite.Searcher;
import word2vec.lite.Searcher.Match;
import word2vec.lite.Searcher.UnknownWordException;
import word2vec.lite.Word2VecModel;
import word2vec.lite.Word2VecTrainerBuilder;
import word2vec.lite.neuralnetwork.NeuralNetworkType;
import word2vec.lite.util.AutoLog;
import word2vec.lite.util.Common;
import word2vec.lite.util.Format;
import word2vec.lite.util.ProfilingTimer;
import word2vec.lite.util.Strings;
import org.apache.commons.logging.Log;
import org.apache.thrift.TException;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

// TODO: rework this to be more useful as standalone program
/** Example usages of {@link Word2VecModel} */
public class Word2VecExamples {
	private static final Log LOG = AutoLog.getLog();
	
	/** Runs the example */
	public static void main(String[] args) throws IOException, TException, UnknownWordException, InterruptedException {

		if (args.length < 1)
		{
			System.err.println("Usage: <raw-text-file> (output-file)");
		}

		String pathToTextFile = args[0];
		String outputPath = args.length > 1 ? args[1]: pathToTextFile + ".bin";
		demoWord(pathToTextFile, outputPath);
		//loadModel(outputPath);
	}
	
	/** 
	 * Trains a model and allows user to find similar words
	 * demo-word.sh example from the open source C implementation
	 */
	public static void demoWord(String inputFile, String outputFile) throws IOException, TException, InterruptedException, UnknownWordException {
		File f = new File(inputFile);
		if (!f.exists())
	       	       throw new IllegalStateException("Please download and unzip the text8 example from http://mattmahoney.net/dc/text8.zip");
		List<String> read = Common.readToList(f);
		List<List<String>> partitioned = Lists.transform(read, new Function<String, List<String>>() {
			@Override
			public List<String> apply(String input) {
				return Arrays.asList(input.split(" "));
			}
		});
		
		Word2VecModel model = Word2VecModel.trainer()
				.setMinVocabFrequency(5)
				.useNumThreads(6)
				.setWindowSize(8)
				.type(NeuralNetworkType.CBOW)
				.setLayerSize(50) // 200
				.useNegativeSamples(25)
				.setDownSamplingRate(1e-4)
				.setNumIterations(2) // 5
				.setListener(new Word2VecTrainerBuilder.TrainingProgressListener() {
					@Override public void update(Stage stage, double progress) {
						System.out.println(String.format("%s is %.2f%% complete", Format.formatEnum(stage), progress * 100));
					}
				})
				.train(partitioned);


		try(final OutputStream os = Files.newOutputStream(Paths.get(outputFile))) {
			model.toBinFile(os);
		}
		
		interact(model.forSearch());


	}
	
	/** Loads a model and allows user to find similar words */
	public static void loadModel(String outputFile) throws IOException, TException, UnknownWordException {
		final Word2VecModel model;
		try (ProfilingTimer timer = ProfilingTimer.create(LOG, "Reloading model")) {
			model = Word2VecModel.fromBinFile(new File(outputFile));
		}
		interact(model.forSearch());
	}
	
	/** Example using Skip-Gram model */
	public static void skipGram() throws IOException, TException, InterruptedException, UnknownWordException {
		List<String> read = Common.readToList(new File("sents.cleaned.word2vec.txt"));
		List<List<String>> partitioned = Lists.transform(read, new Function<String, List<String>>() {
			@Override
			public List<String> apply(String input) {
				return Arrays.asList(input.split(" "));
			}
		});
		
		Word2VecModel model = Word2VecModel.trainer()
				.setMinVocabFrequency(100)
				.useNumThreads(20)
				.setWindowSize(7)
				.type(NeuralNetworkType.SKIP_GRAM)
				.useHierarchicalSoftmax()
				.setLayerSize(300)
				.useNegativeSamples(0)
				.setDownSamplingRate(1e-3)
				.setNumIterations(5)
				.setListener(new Word2VecTrainerBuilder.TrainingProgressListener() {
					@Override public void update(Stage stage, double progress) {
						System.out.println(String.format("%s is %.2f%% complete", Format.formatEnum(stage), progress * 100));
					}
				})
				.train(partitioned);
		
		try (ProfilingTimer timer = ProfilingTimer.create(LOG, "Writing output to file")) {
			model.toBinFile(new FileOutputStream("300layer.20threads.5iter.model"));

		}
		
		interact(model.forSearch());
	}
	
	private static void interact(Searcher searcher) throws IOException, UnknownWordException {
		try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
			while (true) {
				System.out.print("Enter word or sentence (EXIT to break): ");
				String word = br.readLine();
				if (word.equals("EXIT")) {
					break;
				}
				List<Match> matches = searcher.getMatches(word, 20);
				System.out.println(Strings.joinObjects("\n", matches));
			}
		}
	}
}
