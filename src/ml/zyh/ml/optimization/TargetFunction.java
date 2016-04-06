package zyh.ml.optimization;

import java.util.List;
import java.util.Map;

public interface TargetFunction {

	/**
	 * @return number of arguments this target function takes
	 */
	public int numberOfArguments();

	/**
	 * Initialize the arguments for the first iteration
	 * @param arguments store changes to this array
	 */
	public void initializeArguments(double[] arguments);

	/**
	 * Evaluate the target function
	 * @param arguments to be passed in for the evaluation
	 * @param gradient the implementation should write the gradient into this array
	 * @return the evaluated result
	 */
	public double evaluate(double[] arguments, double[] gradient);

	/**
	 * This method will be called before optimization starts
	 * @return a list of titles for each additional information about evaluation,
	 *         otherwise <b>null</b>.
	 */
	public List<String> additionalInfoTitles();

	/**
	 * Provide additional information of the last evaluation
	 * @param infoMap a map where the method will store informations
	 */
	public void setAdditionalInfo(Map<String, Double> infoMap);

	/**
	 * After optimization finished, the method is called for the original
	 * class to update its arguments
	 * @param arguments the optimized arguments
	 */
	public void updateArguments(double[] arguments);

}
