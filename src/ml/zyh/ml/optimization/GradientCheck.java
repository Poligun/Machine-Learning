package zyh.ml.optimization;

import java.util.Random;

import zyh.ml.utils.Logger;

public class GradientCheck {

	private Logger logger = new Logger(1);

	private int numberOfRounds = 10;

	/**
	 * Check whether the given function correctly computes the gradient
	 * @param targetFunction targetFunction to be tested
	 * @return <b>true</b> if computed gradients are correct, otherwise <b>false</b>
	 * @throws Exception
	 */
	public boolean gradientCheck(TargetFunction targetFunction) throws Exception {
		Random random = new Random();
		int numberOfArguments = targetFunction.numberOfArguments();
		double[] arguments = new double[numberOfArguments];
		double[] gradient = new double[numberOfArguments];
		double[] tempGradient = new double[numberOfArguments];
		double epsilon = 0.0001;

		for (int round = 0; round < numberOfRounds; round++) {
			for (int i = 0; i < arguments.length; i++) {
				arguments[i] = (round > 0) ? random.nextDouble() : 1.0;
			}

			double original = targetFunction.evaluate(arguments, gradient);

			for (int i = 0; i < arguments.length; i++) {
				double originalValue = arguments[i];

				arguments[i] = originalValue + epsilon;
				double plus = targetFunction.evaluate(arguments, tempGradient);

				arguments[i] = originalValue - epsilon;
				double minus = targetFunction.evaluate(arguments, tempGradient);

				arguments[i] = originalValue;

				double expectedGradient = (plus - minus) / (2 * epsilon);
				double delta = gradient[i] - expectedGradient;

				if (Math.abs(delta) > epsilon) {
					logger.log("Round #%d: For argument #%d", round + 1, i + 1);
					logger.log("Expected Gradient: %f, Got: %f", expectedGradient, gradient[i]);
					logger.log("Plus: %f, Original: %f, Minus: %f", plus, original, minus);
					logger.logArray("Args", arguments);
					return false;
				}
			}
		}
		return true;
	}

}
