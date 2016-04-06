package zyh.ml.utils;

import java.io.Serializable;

public class Logger extends Timer implements Serializable {

	/**
	 *
	 */
	private static final long serialVersionUID = -7940503665521069302L;

	private int verboseLevel;

	public void setVerboseLevel(int verboseLevel) {
		this.verboseLevel = verboseLevel;
	}

	public Logger(int verboseLevel) {
		super();
		this.verboseLevel = verboseLevel;
	}

	public void log(int requiredLevel, String format, Object... args) {
		if (verboseLevel >= requiredLevel)
			System.out.println(String.format(format, args));
	}

	public void log(String format, Object... args) {
		log(1, format, args);
	}

	public void log(String message) {
		log(1, message);
	}

	public void logArray(String title, double[] array) {
		System.out.print(title + ":");
		for (int i = 0; i < array.length; i++) {
			System.out.print(" ");
			System.out.print(array[i]);
		}
		System.out.println();
	}

	public void logDuration(int requiredLevel) {
		log(requiredLevel, "Finished within %d ms.", duration());
	}

	public void logDuration() {
		logDuration(1);
	}
}
