package zyh.ml.optimization;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;

import zyh.ml.utils.Logger;

public class StochasticGradientDescent {

	public enum Algorithm {
		BFGS,
		LimitedMemoryBFGS,
		TrustRegieonNewtonMethod
	}

	private Logger logger = new Logger(1);

	private Algorithm algorithm;

	private TargetFunction targetFunction;

	private int numberOfIterations;

	private boolean performGradientCheck = false;

	public void setPerformGradientCheck(boolean performGradientCheck) {
		this.performGradientCheck = performGradientCheck;
	}

	private boolean normalizeDirectionVector = false;

	public void setNormalizeDirectionVector(boolean normalizeDirectionVector) {
		this.normalizeDirectionVector = normalizeDirectionVector;
	}

	/* Two coefficients for the Backtracking Line Search */

	private double lambda = 0.6;

	private double C = 0.8;

	private double minimumStepSize = 0.00001;

	public StochasticGradientDescent(TargetFunction targetFunction, int numberOfIterations) {
		this(targetFunction, numberOfIterations, Algorithm.LimitedMemoryBFGS);
	}

	public StochasticGradientDescent(TargetFunction targetFunction, int numberOfIterations, Algorithm algorithm) {
		this.targetFunction = targetFunction;
		this.numberOfIterations = numberOfIterations;
		this.algorithm = algorithm;
	}

	public void run() throws Exception {
		if (performGradientCheck) {
			GradientCheck gradientCheck = new GradientCheck();
			logger.log("Checking gradient...");
			if (!gradientCheck.gradientCheck(targetFunction))
				throw new Exception("Target function doesn't pass the gradient check.");
			logger.log("Gradient check success...");
		}

		if (algorithm != Algorithm.BFGS && algorithm != Algorithm.LimitedMemoryBFGS)
			throw new Exception("Unsupported algorithm.");

		final int numberOfArguments = targetFunction.numberOfArguments();
		double[] arguments = new double[numberOfArguments];
		double[] gradient = new double[numberOfArguments];

		int historySize = (algorithm == Algorithm.LimitedMemoryBFGS) ? 10 : 0;
		FixedSizeList<double[]> argumentDiff = new FixedSizeList<>(historySize);
		FixedSizeList<double[]> gradientDiff = new FixedSizeList<>(historySize);
		FixedSizeList<Double> inversedRhos = new FixedSizeList<>(historySize);

		for (int i = 0; i < numberOfArguments; i++)
			arguments[i] = 1.0;
		targetFunction.initializeArguments(arguments);

		double result = targetFunction.evaluate(arguments, gradient);
		List<String> infoTitles = targetFunction.additionalInfoTitles();
		Map<String, Double> infoMap = new HashMap<>();
		String titles = "Iteration\tTarget Function\tGradient Norm";

		if (infoTitles != null) {
			for (String infoTitle : infoTitles) {
				titles += String.format("\t%s", infoTitle);
				infoMap.put(infoTitle, 0.0);
			}
		}

		logger.log("Stochastic Gradient Descent... MaxIter = %d", numberOfIterations);
		logger.log(titles);

		for (int i = 0; i < numberOfIterations; i++) {
			/* direction = InversedHessianMatrix * Gradient */

			double[] direction = inverseHessianMultiply(argumentDiff, gradientDiff, inversedRhos, gradient);

			if (normalizeDirectionVector) {
				double factor = 0.0;

				for (int j = 0; j < direction.length; j++)
					factor += direction[j] * direction[j];
				factor = 1.0 / Math.sqrt(factor);

				for (int j = 0; j < direction.length; j++)
					direction[j] *= factor;
			}

			/* Perform a line search on the direction */

			double slope = 0.0;

			for (int j = 0; j < gradient.length; j++)
				slope += gradient[j] * (-direction[j]);
			slope *= C;
			if (slope > 0)
				slope = -slope;

			double stepSize = 1.0;
			double[] newArguments = new double[arguments.length];
			double[] newGradient = new double[gradient.length];

			while (true) {
				for (int j = 0; j < arguments.length; j++)
					newArguments[j] = arguments[j] - stepSize * direction[j];
				double newResult = targetFunction.evaluate(newArguments, newGradient);
				if (newResult < result) { // + stepSize * slope) {
					result = newResult;
					break;
				}
				stepSize *= lambda;
				if (stepSize < minimumStepSize)
					break;
			}

			if (stepSize < minimumStepSize) {
				logger.log("Step size reaches 0 during line search.");
				break;
			}

			double gradNorm = 0.0;

			for (int j = 0; j < newGradient.length; j++)
				gradNorm += newGradient[j] * newGradient[j];
			gradNorm = Math.sqrt(gradNorm);

			String iterationResults = String.format("%d\t%f\t%f", i + 1, result, gradNorm);

			if (infoTitles != null) {
				targetFunction.setAdditionalInfo(infoMap);
				for (String infoTitle : infoTitles) {
					iterationResults += String.format("\t%f", infoMap.get(infoTitle));
				}
			}
			logger.log(iterationResults);

			if (gradNorm <= 0.000001) {
				logger.log("Norm of gradient reaches 0.");
				break;
			}

			double[] argDiff = new double[numberOfArguments];
			double[] gradDiff = new double[numberOfArguments];
			double rho = 0.0;

			for (int j = 0; j < numberOfArguments; j++) {
				argDiff[j] = newArguments[j] - arguments[j];
				gradDiff[j] = newGradient[j] - gradient[j];
			}

			for (int j = 0; j < numberOfArguments; j++)
				rho += argDiff[j] * gradDiff[j];
			inversedRhos.add(1.0 / rho);

			argumentDiff.add(argDiff);
			gradientDiff.add(gradDiff);

			arguments = newArguments;
			gradient = newGradient;
		}

		targetFunction.updateArguments(arguments);
	}

	/**
	 * Assuming the central Hessian matrix is identity, this function multiplies the inversed Hessian matrix
	 * with gradient to form the new direction, without having to store the matrix
	 * @param argumentDiff
	 * @param gradientDiff
	 * @param inversedRhos
	 * @param gradient
	 * @return
	 */
	private double[] inverseHessianMultiply(FixedSizeList<double[]> argumentDiff, FixedSizeList<double[]> gradientDiff,
			FixedSizeList<Double> inversedRhos, double[] gradient) {
		double[] direction = new double[gradient.length];
		double[] alphas = new double[argumentDiff.size()];
		ListIterator<double[]> argDiffIter = argumentDiff.getElements().listIterator();
		ListIterator<double[]> gradDiffIter = gradientDiff.getElements().listIterator();
		ListIterator<Double> rhoIter = inversedRhos.getElements().listIterator();

		System.arraycopy(gradient, 0, direction, 0, gradient.length);

		for (int i = argumentDiff.size() - 1; i >= 0; i--) {
			double alpha = 0.0;
			final double rho = rhoIter.next();
			final double[] argDiff = argDiffIter.next();
			final double[] gradDiff = gradDiffIter.next();

			for (int j = 0; j < direction.length; j++)
				alpha += rho * argDiff[j] * direction[j];

			for (int j = 0; j < direction.length; j++)
				direction[j] -= alpha * gradDiff[j];

			alphas[i] = alpha;
		}

		if (argumentDiff.size() > 0) {
			double numerator = 0.0;
			double denominator = 0.0;
			final double[] argDiff = argumentDiff.elements.getFirst();
			final double[] gradDiff = gradientDiff.elements.getFirst();

			for (int i = 0; i < argDiff.length; i++) {
				numerator += gradDiff[i] * argDiff[i];
				denominator += gradDiff[i] * gradDiff[i];
			}

			double H = numerator / denominator;

			for (int i = 0; i < direction.length; i++)
				direction[i] *= H;
		}

		argDiffIter = argumentDiff.getElements().listIterator(argumentDiff.size());
		gradDiffIter = gradientDiff.getElements().listIterator(gradientDiff.size());
		rhoIter = inversedRhos.getElements().listIterator(inversedRhos.size());

		for (int i = 0; i < argumentDiff.size(); i++) {
			final double alpha = alphas[i];
			final double rho = rhoIter.previous();
			final double[] argDiff = argDiffIter.previous();
			final double[] gradDiff = gradDiffIter.previous();
			double beta = 0.0;

			for (int j = 0; j < direction.length; j++)
				beta += rho * gradDiff[j] * direction[j];

			for (int j = 0; j < direction.length; j++)
				direction[j] += (alpha - beta) * argDiff[j];
		}

		return direction;
	}

	private static class FixedSizeList<T> {
		private int maximumSize;
		private LinkedList<T> elements = new LinkedList<>();

		public FixedSizeList(int maximumSize) {
			this.maximumSize = maximumSize;
		}

		public int size() {
			return elements.size();
		}

		public void add(T t) {
			elements.addFirst(t);
			if (maximumSize > 0 && elements.size() > maximumSize)
				elements.removeLast();
		}

		public List<T> getElements() {
			return elements;
		}
	}

}
