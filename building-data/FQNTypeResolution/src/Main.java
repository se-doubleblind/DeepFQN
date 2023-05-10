package fqntypeparser;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Scanner;

import fqntypeparser.FileUtil;


public class Main {
    public static void main(String[] args) {
		String[] libs = new String[]{"android", "com.google.gwt", "org.hibernate", "java", "org.joda.time", "com.thoughtworks.xstream"};
		String basePath = "/home/axy190020/research/deepfqn/data/fqndata/";
		String[] srcPaths = new String[]{"android", "gwt", "hibernate-orm", "jdk", "joda-time", "xstream"};
		String[] jarPaths = new String[]{"android-libcore-lib", "gwt-lib", "hibernate-orm-lib", "jdk-lib", "joda-time-lib", "xstream-lib"};

		String outBasePath = "/home/axy190020/research/deepfqn/data/fqndata/typedata/";

		for (int i = 0; i < libs.length; i++) {
			long start = System.currentTimeMillis();
			ProjectSequencesGenerator psg = new ProjectSequencesGenerator(basePath + srcPaths[i], basePath + jarPaths[i], false);
			File outDir = new File(outBasePath + srcPaths[i]);
			int n = 0;
			if (!outDir.exists())
				outDir.mkdirs();
			try {
				n = psg.generateSequences(false, libs[i], outDir.getAbsolutePath());
			} catch (Throwable t) {
				t.printStackTrace();
			}

			long end = System.currentTimeMillis();
			System.out.println("Finish parsing " + srcPaths[i] + " corpus in " + (end - start) / 1000 + ". Number of sequences: " + n);
		}
    }
}
