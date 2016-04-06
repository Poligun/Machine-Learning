package zyh.ml.regression;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import zyh.ml.data.IndexedSample;
import zyh.ml.optimization.StochasticGradientDescent;
import zyh.ml.optimization.TargetFunction;
import zyh.ml.utils.Logger;
import zyh.ml.utils.TaskDispatcher;

/**
 * L2 Regularized Multinomial Logistic Regression
 * @author zhaoyuhan
 */
public class LogisticRegression implements TargetFunction, Serializable {

	/**
	 *
	 */
	private static final long serialVersionUID = 3666084436866079637L;

	private int numberOfClasses;

	private int numberOfFeatures;

	private int numberOfThreads = 10;

	private boolean usingL2Regularization = true;

	public void setUsingL2Regularization(boolean usingL2Regularization) {
		this.usingL2Regularization = usingL2Regularization;
	}

	private double regularizationCoefficient = 1.0;

	private double[] thetas = null;

	private Logger logger = new Logger(1);

	private int finishedThreads;
	private ThreadStatus threadStatus = ThreadStatus.ShouldWait;

	private double logLikelihood;
	private int correctLabels;
	private double[] gradient;

	private List<IndexedSample> trainingSamples;
	private List<Double> sampleWeights;

	private List<Thread> threads = new ArrayList<>();

	private Lock lock;
	private Condition workerCondition;
	private Condition masterCondition;

	public LogisticRegression(int numberOfClasses, int numberOfFeatures) {
		this.numberOfClasses = numberOfClasses;
		this.numberOfFeatures = numberOfFeatures;
	}

	private enum ThreadStatus {
		ShouldWait, ShouldStart, ShouldStop;
	}

	private class Worker implements Runnable {

		public Lock lock;
		public Condition workerCondition;
		public Condition masterCondition;
		public Object writeLock;
		public int beginIndex;
		public int endIndex;

		@Override
		public void run() {
			while (true) {
				lock.lock();

				try {
					while (threadStatus == ThreadStatus.ShouldWait)
						workerCondition.await();
					lock.unlock();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}

				if (threadStatus == ThreadStatus.ShouldStop)
					break;

				for (int i = beginIndex; i < endIndex; i++) {
					final IndexedSample sample = trainingSamples.get(i);
					final double[] probabilities = probabilityPredict(sample);
					final int predictedLabel = getPredictedLabel(probabilities);

					synchronized(writeLock) {
						if (predictedLabel == sample.label)
							correctLabels++;
						logLikelihood -= sampleWeights.get(i) * Math.log(probabilities[sample.label]);
						for (int j = numberOfClasses - 2; j >= 0; j--) {
							int startIndex = j * numberOfFeatures;
							double multiplier = -probabilities[j];

							if (sample.label == j)
								multiplier += 1.0;

							for (Entry<Integer, Double> entry : sample.features.entrySet())
								gradient[startIndex + entry.getKey()] -= sampleWeights.get(i) * entry.getValue() * multiplier;
						}
					}
				}

				lock.lock();
				if (++finishedThreads == numberOfThreads) {
					threadStatus = ThreadStatus.ShouldWait;
					masterCondition.signal();
				}
				lock.unlock();
			}
		}
	}

	@Override
	public int numberOfArguments() {
		return (numberOfClasses - 1) * numberOfFeatures;
	}

	@Override
	public void initializeArguments(double[] arguments) {
		if (thetas != null && thetas.length == arguments.length) {
			System.arraycopy(thetas, 0, arguments, 0, thetas.length);
			logger.log("Resuming training...");
		}
	}

	@Override
	public void updateArguments(double[] arguments) {
		lock.lock();
		threadStatus = ThreadStatus.ShouldStop;
		workerCondition.signalAll();
		lock.unlock();

		for (final Thread thread : threads) {
			try {
				thread.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		threads.clear();
		workerCondition = null;
		masterCondition = null;
		lock = null;

		thetas = arguments;
	}

	@Override
	public double evaluate(double[] arguments, double[] gradient) {
		if (threads.size() == 0) {
			TaskDispatcher dispatcher = new TaskDispatcher(trainingSamples.size(), numberOfThreads);
			Object writeLock = new Object();
			lock = new ReentrantLock();
			workerCondition = lock.newCondition();
			masterCondition = lock.newCondition();

			for (int i = 0; i < numberOfThreads; i++) {
				Worker worker = new Worker();

				worker.writeLock = writeLock;
				worker.lock = lock;
				worker.workerCondition = workerCondition;
				worker.masterCondition = masterCondition;
				worker.beginIndex = dispatcher.begin(i);
				worker.endIndex = dispatcher.end(i);

				threads.add(new Thread(worker));
			}

			for (final Thread thread : threads)
				thread.start();
		}

		lock.lock();
		logLikelihood = 0.0;
		correctLabels = 0;
		finishedThreads = 0;
		this.thetas = arguments;
		this.gradient = gradient;
		for (int i = 0; i < gradient.length; i++)
			gradient[i] = 0.0;
		threadStatus = ThreadStatus.ShouldStart;

		try {
			workerCondition.signalAll();
			while (finishedThreads < numberOfThreads)
				masterCondition.await();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		lock.unlock();

		for (int i = 0; i < gradient.length; i++)
			gradient[i] *=  trainingSamples.size();

		logLikelihood *= trainingSamples.size();

		if (usingL2Regularization) {
			double l2reg = 0.0;

			for (int i = 0; i < arguments.length; i++)
				l2reg += arguments[i] * arguments[i];

			for (int i = 0; i < gradient.length; i++)
				gradient[i] += 2 * regularizationCoefficient * arguments[i];

			logLikelihood += regularizationCoefficient * l2reg;
		}

		return logLikelihood;
	}

	@Override
	public List<String> additionalInfoTitles() {
		List<String> titles = new ArrayList<>();
		titles.add("Accuracy");
		return titles;
	}

	@Override
	public void setAdditionalInfo(Map<String, Double> infoMap) {
		infoMap.put("Accuracy", ((double) correctLabels) / trainingSamples.size());
	}

	public boolean fit(List<IndexedSample> samples, List<Double> weights, int numberOfIterations) {
		StochasticGradientDescent sgd = new StochasticGradientDescent(this, numberOfIterations,
				StochasticGradientDescent.Algorithm.BFGS);
		trainingSamples = samples;
		sampleWeights = weights;

		try {
			sgd.run();
		} catch (Exception e) {
			e.printStackTrace();
			logger.log("Stopping threads...");
			updateArguments(null);
			return false;
		}

		return true;
	}

	private int getPredictedLabel(double[] probabilities) {
		double maxProb = 0.0;
		int predictedLabel = 0;

		for (int i = 0; i < probabilities.length; i++) {
			if (probabilities[i] > maxProb) {
				maxProb = probabilities[i];
				predictedLabel = i;
			}
		}

		return predictedLabel;
	}

	public int predict(IndexedSample sample) {
		return getPredictedLabel(probabilityPredict(sample));
	}

	public double[] probabilityPredict(IndexedSample sample) {
		double[] probabilities = new double[numberOfClasses];

		for (int i = numberOfClasses - 2; i >= 0; i--) {
			probabilities[i] = Math.exp(multiply(i, sample));
		}
		probabilities[numberOfClasses - 1] = 1.0;

		double invertedSum = 0.0;

		for (int i = 0; i < numberOfClasses; i++)
			invertedSum += probabilities[i];
		invertedSum = 1.0 / invertedSum;

		for (int i = 0; i < numberOfClasses; i++)
			probabilities[i] *= invertedSum;

		return probabilities;
	}

	public double multiply(int thetaIndex, IndexedSample sample) {
		double sum = 0.0;
		int startIndex = thetaIndex * numberOfFeatures;

		for (Entry<Integer, Double> entry : sample.features.entrySet())
			sum += thetas[startIndex + entry.getKey()] * entry.getValue();

		return sum;
	}

}
