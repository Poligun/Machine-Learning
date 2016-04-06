package zyh.ml.indexer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import zyh.ml.data.IndexedSample;
import zyh.ml.data.Sample;
import zyh.ml.utils.TaskDispatcher;

public class OnePassIndexer extends Indexer {

	/**
	 *
	 */
	private static final long serialVersionUID = 3689149596907508818L;

	private int numberOfThreads = 1;

	public void collectFeatures(Collection<Sample> samples, int cutOff) {
		Map<String, Integer> binaryFeatures = new HashMap<>();
		Set<String> decimalFeatureSet = new HashSet<>();
		Set<String> labelSet = new HashSet<>();

		logger.log("Indexing features...");

		for (final Sample sample : samples) {
			for (final String binaryFeature : sample.getBinaryFeatures()) {
				if (!binaryFeatures.containsKey(binaryFeature))
					binaryFeatures.put(binaryFeature, 1);
				else
					binaryFeatures.put(binaryFeature, binaryFeatures.get(binaryFeature) + 1);
			}
			for (final String decimalFeatureName : sample.getDecimalFeatures().keySet())
				decimalFeatureSet.add(decimalFeatureName);
			labelSet.add(sample.getLabel());
		}

		Set<String> binaryFeatureSet = new HashSet<>();

		for (final Entry<String, Integer> entry : binaryFeatures.entrySet()) {
			if (entry.getValue() > cutOff)
				binaryFeatureSet.add(entry.getKey());
		}

		featureNameIndices.clear();

		featureNameIndices.put(BIAS_TERM, 0);
		for (final String decimalFeatureName : decimalFeatureSet)
			featureNameIndices.put(decimalFeatureName, featureNameIndices.size());
		for (final String binaryFeatureName : binaryFeatures.keySet())
			featureNameIndices.put(binaryFeatureName, featureNameIndices.size());

		labels = new ArrayList<>(labelSet);
	}
	public void indexSamples(List<Sample> samples) {
		logger.log("Indexing samples...");
		logger.tick();

		List<Thread> threads = new ArrayList<>();
		TaskDispatcher dispatcher = new TaskDispatcher(samples.size(), numberOfThreads);
		IndexedSample[] indexedSamples = new IndexedSample[samples.size()];

		for (int i = 0; i < numberOfThreads; i++) {
			final int threadIndex = i;
			threads.add(new Thread(new Runnable() {
				@Override
				public void run() {
					for (int i = dispatcher.begin(threadIndex); i < dispatcher.end(threadIndex); i++)
						indexedSamples[i] = indexSample(samples.get(i));
				}
			}));
		}

		for (final Thread thread : threads)
			thread.start();

		for (final Thread thread : threads) {
			try {
				thread.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}

		logger.tick();
		logger.logDuration();

		this.indexedSamples = Arrays.asList(indexedSamples);
	}

}
