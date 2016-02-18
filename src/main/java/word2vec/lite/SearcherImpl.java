package word2vec.lite;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import word2vec.lite.util.Pair;

import java.util.List;

/** Implementation of {@link Searcher} */
class SearcherImpl implements Searcher {
  private final Word2VecModel model;
  private final ImmutableMap<String, Integer> word2vectorOffset;

  SearcherImpl(final Word2VecModel model) {
	this.model = model;
	this.model.normalize();
	final ImmutableMap.Builder<String, Integer> result = ImmutableMap.builder();
	for (int i = 0; i < model.vocab.size(); i++) {
	  result.put(model.vocab.get(i), i);
	}

	word2vectorOffset = result.build();
  }

  @Override public List<Match> getMatches(String s, int maxNumMatches) throws UnknownWordException {
	return getMatches(getVector(s), maxNumMatches);
  }

  @Override public double cosineDistance(String s1, String s2) throws UnknownWordException {
	return calculateDistance(getVector(s1), getVector(s2));
  }

  @Override public boolean contains(String word) {
	return word2vectorOffset.containsKey(word);
  }

  @Override public List<Match> getMatches(final float[] vec, int maxNumMatches) {
	return Match.ORDERING.greatestOf(
		Iterables.transform(model.vocab, new Function<String, Match>() {
		  @Override
		  public Match apply(String other) {
			float[] otherVec = getVectorOrNull(other);
			double d = calculateDistance(otherVec, vec);
			return new MatchImpl(other, d);
		  }
		}),
		maxNumMatches
	);
  }

  private double calculateDistance(float[] otherVec, float[] vec) {
	double d = 0;
	for (int a = 0; a < model.layerSize; a++)
	  d += vec[a] * otherVec[a];
	return d;
  }

  @Override public float[] getRawVector(String word) throws UnknownWordException {
	return getVector(word);
  }

  /**
   * @return Vector for the given word
   * @throws UnknownWordException If word is not in the model's vocabulary
   */
  private float[] getVector(String word) throws UnknownWordException {
	final float[] result = getVectorOrNull(word);
	if(result == null)
	  throw new UnknownWordException(word);

	return result;
  }

  private float[] getVectorOrNull(final String word) {
	final Integer index = word2vectorOffset.get(word);
	  if(index == null)
		return null;

	return model.vectors[index];
  }

  /** @return Vector difference from v1 to v2 */
  private float[] getDifference(float[] v1, float[] v2) {
	float[] diff = new float[model.layerSize];
	for (int i = 0; i < model.layerSize; i++)
	  diff[i] = v1[i] - v2[i];
	return diff;
  }

  @Override public SemanticDifference similarity(String s1, String s2) throws UnknownWordException {
	  float[] v1 = getVector(s1);
	  float[] v2 = getVector(s2);
	  final float[] diff = getDifference(v1, v2);

	return new SemanticDifference() {
	  @Override public List<Match> getMatches(String word, int maxMatches) throws UnknownWordException {
		float[] target = getDifference(getVector(word), diff);
		return SearcherImpl.this.getMatches(target, maxMatches);
	  }
	};
  }

  /** Implementation of {@link Match} */
  private static class MatchImpl extends Pair<String, Double> implements Match {
	private MatchImpl(String first, Double second) {
	  super(first, second);
	}

	@Override public String match() {
	  return first;
	}

	@Override public double distance() {
	  return second;
	}

	@Override public String toString() {
	  return String.format("%s [%s]", first, second);
	}
  }
}
