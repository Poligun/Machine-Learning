package zyh.ml.utils;

import java.util.ArrayList;
import java.util.List;

public class TaskDispatcher {

	private List<Integer> begins = new ArrayList<>();

	private List<Integer> ends = new ArrayList<>();

	public TaskDispatcher(int numberOfTasks, int numberOfWorkers) {
		int tasksPerWorker = numberOfTasks / numberOfWorkers;
		int extraTasks = numberOfTasks % numberOfWorkers;

		for (int i = 0; i < numberOfWorkers; i++) {
			begins.add(i * tasksPerWorker + Math.min(extraTasks, i));
			ends.add(begins.get(i) + tasksPerWorker);
		}
	}

	public int begin(int workerIndex) {
		return begins.get(workerIndex);
	}

	public int end(int workerIndex) {
		return ends.get(workerIndex);
	}
}
