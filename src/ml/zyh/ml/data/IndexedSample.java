package zyh.ml.data;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

public class IndexedSample implements Serializable {

	/**
	 *
	 */
	private static final long serialVersionUID = -8676433665150090059L;

	public Map<Integer, Double> features = new HashMap<>();

	public int label;

}
