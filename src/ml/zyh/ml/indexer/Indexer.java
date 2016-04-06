package zyh.ml.indexer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import zyh.ml.data.IndexedSample;
import zyh.ml.data.Sample;
import zyh.ml.utils.Logger;

public abstract class Indexer implements Serializable {

	/**
	 *
	 */
	private static final long serialVersionUID = -5859589069866903397L;

	protected static final String BIAS_TERM = "##___BIAS__TERM___##";

	protected Logger logger = new Logger(1);

	protected Map<String, Integer> featureNameIndices = new HashMap<>();

	protected List<String> labels = new ArrayList<>();

	protected List<IndexedSample> indexedSamples = new ArrayList<>();

	public List<String> getLabels() {
		return labels;
	}

	public int numberOfLabels() {
		return labels.size();
	}

	public int numberOfFeatures() {
		return featureNameIndices.size();
	}

	public List<IndexedSample> getIndexedSamples() {
		return indexedSamples;
	}

	public IndexedSample indexSample(Sample sample) {
		IndexedSample indexedSample = new IndexedSample();

		indexedSample.features.put(0, 1.0);

		for (final Entry<String, Double> entry : sample.getDecimalFeatures().entrySet()) {
			if (featureNameIndices.containsKey(entry.getKey()))
				indexedSample.features.put(featureNameIndices.get(entry.getKey()), entry.getValue());
		}

		for (final String binaryFeature : sample.getBinaryFeatures()) {
			if (featureNameIndices.containsKey(binaryFeature))
				indexedSample.features.put(featureNameIndices.get(binaryFeature), 1.0);
		}

		return indexedSample;
	}


}
