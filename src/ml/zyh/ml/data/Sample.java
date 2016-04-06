package zyh.ml.data;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class Sample implements Samplable {

	public static final String SEPARATOR = "=";

	private Map<String, String> binaryFeatureMap = new HashMap<>();

	private Set<String> binaryFeatures = new HashSet<>();

	private Map<String, Double> decimalFeatures = new HashMap<>();

	private String label;

	@Override
	public void addBinaryFeature(String name, String value) {
		if (binaryFeatureMap.containsKey(name)) {
			final String oldFeature = String.format("%s%s%s", name, SEPARATOR, binaryFeatureMap.get(name));
			binaryFeatures.remove(oldFeature);
		}
		binaryFeatureMap.put(name, value);
		binaryFeatures.add(String.format("%s%s%s", name, SEPARATOR, value));
	}

	@Override
	public void addDecimalFeature(String name, double value) {
		decimalFeatures.put(name, value);
	}

	public Set<String> getBinaryFeatures() {
		return binaryFeatures;
	}

	public Map<String, Double> getDecimalFeatures() {
		return decimalFeatures;
	}

	@Override
	public void setLabel(String label) {
		this.label = label;
	}

	public String getLabel() {
		return label;
	}

}
