package zyh.ml.regression;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import zyh.ml.data.IndexedSample;
import zyh.ml.data.Sample;
import zyh.ml.indexer.Indexer;
import zyh.ml.indexer.OnePassIndexer;

public class LRModel implements Serializable {

	/**
	 *
	 */
	private static final long serialVersionUID = -7469773052124361289L;

	private LogisticRegression logisticRegression;

	private Indexer indexer;

	private List<Double> weights;

	private LRModel() {}

	public String predict(Sample sample) {
		IndexedSample indexedSample = indexer.indexSample(sample);
		int labelIndex = logisticRegression.predict(indexedSample);
		String label = indexer.getLabels().get(labelIndex);
		return label;
	}

	public boolean trainModel(int iterations) {
		if (indexer != null && weights != null)
			return logisticRegression.fit(indexer.getIndexedSamples(), weights, iterations);
		return false;
	}

	public void discardTrainingData() {
		indexer = null;
		weights = null;
	}

	public static LRModel train(List<Sample> samples, List<Double> weights, int cutOff, int iterations) {
		LRModel model = new LRModel();
		OnePassIndexer indexer = new OnePassIndexer();

		indexer.collectFeatures(samples, cutOff);
		indexer.indexSamples(samples);

		model.indexer = indexer;
		model.weights = weights;
		model.logisticRegression = new LogisticRegression(
				model.indexer.numberOfLabels(),
				model.indexer.numberOfFeatures());

		if (!model.trainModel(iterations))
			return null;

		return model;
	}

	public static LRModel train(Indexer indexer, List<Double> weights, int iterations) {
		LRModel model = new LRModel();

		model.indexer = indexer;
		model.weights = weights;
		model.logisticRegression = new LogisticRegression(
				model.indexer.numberOfLabels(),
				model.indexer.numberOfFeatures());

		if (!model.trainModel(iterations))
			return null;

		return model;
	}

	public static LRModel train(List<Sample> samples, int cutOff, int iterations) {
		return train(samples, uniformWeights(samples.size()), cutOff, iterations);
	}

	public static LRModel train(Indexer indexer, int iterations) {
		return train(indexer, uniformWeights(indexer.getIndexedSamples().size()), iterations);
	}

	private static List<Double> uniformWeights(int sampleSize) {
		List<Double> weights = new ArrayList<>();

		final double w = 1.0 / sampleSize;

		for (int i = 0; i < sampleSize; i++)
			weights.add(w);

		return weights;
	}
}
