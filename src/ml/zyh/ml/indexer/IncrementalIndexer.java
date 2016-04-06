package zyh.ml.indexer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import zyh.ml.data.IndexedSample;
import zyh.ml.data.Samplable;
import zyh.ml.data.Sample;

public class IncrementalIndexer extends Indexer implements Samplable {

	/**
	 *
	 */
	private static final long serialVersionUID = -7590106955664277112L;

	private IndexedSample indexedSample;

	private Map<String, Integer> binaryFeatures = new HashMap<>();

	private Set<String> decimalFeatureSet = new HashSet<>();

	public IncrementalIndexer() {
		featureNameIndices.put(BIAS_TERM, 0);
		indexedSamples = new ArrayList<>();
	}

	public void newSample() {
		indexedSample = new IndexedSample();
		indexedSample.features.put(0, 1.0);
	}

	public void addToSamples() {
		indexedSamples.add(indexedSample);
		discard();
	}

	public void discard() {
		indexedSample = null;
	}

	@Override
	public void addBinaryFeature(String name, String value) {
		String featureName = String.format("%s%s%s", name, Sample.SEPARATOR, value);
		if (!binaryFeatures.containsKey(featureName)) {
			binaryFeatures.put(featureName, 1);
			featureNameIndices.put(featureName, featureNameIndices.size());
		}
		else {
			binaryFeatures.put(featureName, binaryFeatures.get(featureName) + 1);
		}
		indexedSample.features.put(featureNameIndices.get(featureName), 1.0);
	}

	@Override
	public void addDecimalFeature(String name, double value) {
		if (!decimalFeatureSet.contains(name)) {
			decimalFeatureSet.add(name);
			featureNameIndices.put(name, featureNameIndices.size());
		}
		indexedSample.features.put(featureNameIndices.get(name), value);
	}

	@Override
	public void setLabel(String label) {
		for (int i = 0; i < labels.size(); i++) {
			if (labels.get(i).equals(label)) {
				indexedSample.label = i;
				return;
			}
		}
		labels.add(label);
		indexedSample.label = labels.size() - 1;
	}

	public void cutOff(int cutOff) {
		for (Entry<String, Integer> entry : binaryFeatures.entrySet()) {
			if (entry.getValue() >= cutOff)
				continue;
			int index = featureNameIndices.get(entry.getKey());
			for (IndexedSample indexedSample : indexedSamples) {
				indexedSample.features.remove(index);
			}
		}
	}
}
