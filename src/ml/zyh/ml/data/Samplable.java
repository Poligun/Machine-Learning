package zyh.ml.data;

public interface Samplable {

	public void addBinaryFeature(String name, String value);

	public void addDecimalFeature(String name, double value);

	public void setLabel(String label);
}
