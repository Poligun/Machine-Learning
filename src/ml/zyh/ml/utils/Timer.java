package zyh.ml.utils;

import java.util.ArrayList;
import java.util.List;

public class Timer {

	private List<Long> timeStamps = new ArrayList<>();

	public void tick() {
		timeStamps.add(System.currentTimeMillis());
	}

	public long duration() {
		if (timeStamps.size() < 2)
			return 0;
		else
			return timeStamps.get(timeStamps.size() - 1) - timeStamps.get(timeStamps.size() - 2);
	}

}
