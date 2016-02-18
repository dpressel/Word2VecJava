package word2vec.lite;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import word2vec.lite.util.ProfilingTimer;
import word2vec.lite.util.AC;


/**
 * Represents the Word2Vec model, containing vectors for each word
 * <p/>
 * Instances of this class are obtained via:
 * <ul>
 * <li> {@link #trainer()}
 * </ul>
 *
 * @see {@link #forSearch()}
 */
public class Word2VecModel {
    final List<String> vocab;
    final int layerSize;

	private final static long ONE_GB = 1024 * 1024 * 1024;
	/** Size of the layers */
	public int getLayerSize()
	{
		return layerSize;
	}
	/** Resulting vectors */
	public float[][] getVectors() { return vectors; }

	float[][] vectors;

	Word2VecModel(Iterable<String> vocab, float[][] vectors)
	{
		layerSize = vectors[0].length;
		this.vocab = Lists.newArrayList(vocab);
		this.vectors = vectors;
	}

	/** @return Vocabulary */
	public Iterable<String> getVocab() {
		return vocab;
	}

	/** @return {@link Searcher} for searching */
	public Searcher forSearch() {
		return new SearcherImpl(this);
	}

	/**
	 * Forwards to {@link #fromBinFile(File, ByteOrder, ProfilingTimer)} with the default
	 * ByteOrder.LITTLE_ENDIAN and no ProfilingTimer
	 */
	public static Word2VecModel fromBinFile(File file)
			throws IOException {
		return fromBinFile(file, ByteOrder.LITTLE_ENDIAN, ProfilingTimer.NONE);
	}

	/**
	 * Forwards to {@link #fromBinFile(File, ByteOrder, ProfilingTimer)} with no ProfilingTimer
	 */
	public static Word2VecModel fromBinFile(File file, ByteOrder byteOrder)
			throws IOException {
		return fromBinFile(file, byteOrder, ProfilingTimer.NONE);
	}

	/**
	 * @return {@link Word2VecModel} created from the binary representation output
	 * by the open source C version of word2vec using the given byte order.
	 */
	public static Word2VecModel fromBinFile(File file, ByteOrder byteOrder, ProfilingTimer timer)
			throws IOException {

		try (
				final FileInputStream fis = new FileInputStream(file);
				final AC ac = timer.start("Loading vectors from bin file")
		) {
			final FileChannel channel = fis.getChannel();
			timer.start("Reading gigabyte #1");
			MappedByteBuffer buffer =
					channel.map(
							FileChannel.MapMode.READ_ONLY,
							0,
							Math.min(channel.size(), Integer.MAX_VALUE));
			buffer.order(byteOrder);
			int bufferCount = 1;
			// Java's NIO only allows memory-mapping up to 2GB. To work around this problem, we re-map
			// every gigabyte. To calculate offsets correctly, we have to keep track how many gigabytes
			// we've already skipped. That's what this is for.

			StringBuilder sb = new StringBuilder();
			char c = (char) buffer.get();
			while (c != '\n') {
				sb.append(c);
				c = (char) buffer.get();
			}
			String firstLine = sb.toString();
			int index = firstLine.indexOf(' ');
			Preconditions.checkState(index != -1,
					"Expected a space in the first line of file '%s': '%s'",
					file.getAbsolutePath(), firstLine);

			final int vocabSize = Integer.parseInt(firstLine.substring(0, index));
			final int layerSize = Integer.parseInt(firstLine.substring(index + 1));
			timer.appendToLog(String.format(
					"Loading %d vectors with dimensionality %d",
					vocabSize,
					layerSize));

			List<String> vocabs = new ArrayList<String>(vocabSize);
			float[][] vectors = new float[vocabSize][layerSize];

			long lastLogMessage = System.currentTimeMillis();
			final float[] floats = new float[layerSize];
			for (int lineno = 0; lineno < vocabSize; lineno++) {
				// read vocab
				sb.setLength(0);
				c = (char) buffer.get();
				while (c != ' ') {
					// ignore newlines in front of words (some binary files have newline,
					// some don't)
					if (c != '\n') {
						sb.append(c);
					}
					c = (char) buffer.get();
				}
				vocabs.add(sb.toString());

				// read vector
				final FloatBuffer floatBuffer = buffer.asFloatBuffer();
				floatBuffer.get(floats);
				for (int i = 0; i < floats.length; ++i) {
					vectors[lineno][i] = floats[i];
				}
				buffer.position(buffer.position() + 4 * layerSize);

				// print log
				final long now = System.currentTimeMillis();
				if (now - lastLogMessage > 1000) {
					final double percentage = ((double) (lineno + 1) / (double) vocabSize) * 100.0;
					timer.appendToLog(
							String.format("Loaded %d/%d vectors (%f%%)", lineno + 1, vocabSize, percentage));
					lastLogMessage = now;
				}

				// remap file
				if (buffer.position() > ONE_GB) {
					final int newPosition = (int) (buffer.position() - ONE_GB);
					final long size = Math.min(channel.size() - ONE_GB * bufferCount, Integer.MAX_VALUE);
					timer.endAndStart(
							"Reading gigabyte #%d. Start: %d, size: %d",
							bufferCount,
							ONE_GB * bufferCount,
							size);
					buffer = channel.map(
							FileChannel.MapMode.READ_ONLY,
							ONE_GB * bufferCount,
							size);
					buffer.order(byteOrder);
					buffer.position(newPosition);
					bufferCount += 1;
				}
			}
			timer.end();

			return new Word2VecModel(vocabs, vectors);
		}
	}

    /** Normalizes the vectors in this model */
    public void normalize() {

        for(int i = 0; i < vectors.length; ++i)
        {
            double len = 0;
            for (int j = 0; j < layerSize; ++j)
            {
                len += vectors[i][j] * vectors[i][j];
            }
            len = Math.sqrt(len);

            for (int j = 0; j < layerSize; ++j)
            {
                vectors[i][j] /= len;
            }
        }
    }

	/**
	 * Saves the model as a bin file that's compatible with the C version of Word2Vec
	 */
	public void toBinFile(final OutputStream out) throws IOException {
		final Charset cs = Charset.forName("UTF-8");
		final String header = String.format("%d %d\n", vocab.size(), layerSize);
		out.write(header.getBytes(cs));


		final ByteBuffer buffer = ByteBuffer.allocate(4 * layerSize);
		buffer.order(ByteOrder.LITTLE_ENDIAN);	// The C version uses this byte order.
		for(int i = 0; i < vocab.size(); ++i) {
			out.write(String.format("%s ", vocab.get(i)).getBytes(cs));

			// we have vocab x layer so to get to the next is i * layerSize, but in 1D its just i
			buffer.clear();
			for(int j = 0; j < layerSize; ++j)
				buffer.putFloat(vectors[i][j]);
			out.write(buffer.array());

			out.write('\n');
		}

		out.flush();
	}

	/** @return {@link Word2VecTrainerBuilder} for training a model */
	public static Word2VecTrainerBuilder trainer() {
		return new Word2VecTrainerBuilder();
	}
}
