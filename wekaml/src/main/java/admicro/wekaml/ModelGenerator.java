package admicro.wekaml;

import java.awt.BorderLayout;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Reader;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.REPTree;
import weka.core.Debug;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.core.converters.TextDirectoryLoader;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

public class ModelGenerator {

	public Instances loadDataset(String path) {
		Instances dataset = null;
		try {
			dataset = DataSource.read(path);
			// Make the last attribute be the class
			if (dataset.classIndex() == -1) {
				dataset.setClassIndex(dataset.numAttributes() - 1);
			}
		} catch (Exception ex) {
			Logger.getLogger(ModelGenerator.class.getName()).log(Level.SEVERE, null, ex);
		}

		return dataset;
	}

	public Instances loadDatasetWithId(String path) {
		Instances dataset = null;
		try {
			// dataset = DataSource.read(path);
			CSVLoader loader = new CSVLoader();
			loader.setNominalAttributes("1,2");
			loader.setSource(new File(path));
			dataset = loader.getDataSet();

			// Make the last attribute be the class
			if (dataset.classIndex() == -1) {
				dataset.setClassIndex(dataset.numAttributes() - 1);
			}
		} catch (Exception ex) {
			Logger.getLogger(ModelGenerator.class.getName()).log(Level.SEVERE, null, ex);
		}

		return dataset;
	}

	public Classifier buildClassifierREPTree(Instances traindataset) {
		REPTree rt = new REPTree();

		rt.setBatchSize("64");
		// rt.setDebug(false);
		// rt.setDoNotCheckCapabilities(false);
		// rt.setInitialCount(0);
		// rt.setMaxDepth(-1);
		// rt.setMinNum(2);
		rt.setMinVarianceProp(0.001);
		// rt.setNoPruning(false);
		// rt.setNumDecimalPlaces(2);
		rt.setNumFolds(3);
		// rt.setSeed(1);
		// rt.setSpreadInitialCount(false);

		/*
		 * seed -- The seed used for randomizing the data (default = 1).
		 * 
		 * minNum -- The minimum total weight of the instances in a leaf
		 * (default = 2).
		 * 
		 * numFolds -- Determines the amount of data used for pruning. One fold
		 * is used for pruning, the rest for growing the rules (default = 3).
		 * 
		 * numDecimalPlaces -- The number of decimal places to be used for the
		 * output of numbers in the model (default = 2).
		 * 
		 * batchSize -- The preferred number of instances to process if batch
		 * prediction is being performed. More or fewer instances may be
		 * provided, but this gives implementations a chance to specify a
		 * preferred batch size (default = 100).
		 * 
		 * debug -- If set to true, classifier may output additional info to the
		 * console (default = false).
		 * 
		 * noPruning -- Whether pruning is performed (default = false).
		 * 
		 * spreadInitialCount -- Spread initial count across all values instead
		 * of using the count per value (default = false).
		 * 
		 * doNotCheckCapabilities -- If set, classifier capabilities are not
		 * checked before classifier is built (Use with caution to reduce
		 * runtime) (default = false).
		 * 
		 * maxDepth -- The maximum tree depth (-1 for no restriction) (default =
		 * -1).
		 * 
		 * minVarianceProp -- The minimum proportion of the variance on all the
		 * data that needs to be present at a node in order for splitting to be
		 * performed in regression trees (default = 0.001).
		 * 
		 * initialCount -- Initial class value count (default = 0).
		 */

		try {
			rt.buildClassifier(traindataset);

		} catch (Exception ex) {
			Logger.getLogger(ModelGenerator.class.getName()).log(Level.SEVERE, null, ex);
		}
		return rt;
	}

	public Classifier buildClassifierM5P(Instances traindataset) {
		M5P m5p = new M5P();

		m5p.setBatchSize("64");
		m5p.setBuildRegressionTree(true);
		// m5p.setDebug(false);
		// m5p.setDoNotCheckCapabilities(false);
		// m5p.setMinNumInstances(4);
		// m5p.setNumDecimalPlaces(2);
		// m5p.setSaveInstances(false);
		// m5p.setUnpruned(false);
		// m5p.setUseUnsmoothed(false);

		/*
		 * unpruned -- Whether unpruned tree/rules are to be generated (default
		 * = false).
		 * 
		 * numDecimalPlaces -- The number of decimal places to be used for the
		 * output of numbers in the model (default = 2).
		 * 
		 * batchSize -- The preferred number of instances to process if batch
		 * prediction is being performed. More or fewer instances may be
		 * provided, but this gives implementations a chance to specify a
		 * preferred batch size (default = 100).
		 * 
		 * debug -- If set to true, classifier may output additional info to the
		 * console (default = false).
		 * 
		 * useUnsmoothed -- Whether to use unsmoothed predictions (default =
		 * false).
		 * 
		 * saveInstances -- Whether to save instance data at each node in the
		 * tree for visualization purposes (default = false).
		 * 
		 * minNumInstances -- The minimum number of instances to allow at a leaf
		 * node (default = 4).
		 * 
		 * doNotCheckCapabilities -- If set, classifier capabilities are not
		 * checked before classifier is built (Use with caution to reduce
		 * runtime) (default = false).
		 * 
		 * buildRegressionTree -- Whether to generate a regression tree/rule
		 * instead of a model tree/rule (default = false).
		 * 
		 */

		try {
			m5p.buildClassifier(traindataset);

		} catch (Exception ex) {
			Logger.getLogger(ModelGenerator.class.getName()).log(Level.SEVERE, null, ex);
		}
		return m5p;
	}

	public Evaluation evaluateModel(Classifier model, Instances traindataset, Instances testdataset) {
		Evaluation eval = null;
		try {
			// Evaluate classifier with test dataset
			eval = new Evaluation(traindataset);
			eval.evaluateModel(model, testdataset);
		} catch (Exception ex) {
			Logger.getLogger(ModelGenerator.class.getName()).log(Level.SEVERE, null, ex);
		}
		return eval;
	}

	public void saveModel(Classifier model, String modelpath) {

		try {
			SerializationHelper.write(modelpath, model);
		} catch (Exception ex) {
			Logger.getLogger(ModelGenerator.class.getName()).log(Level.SEVERE, null, ex);
		}
	}

	public Classifier loadModelREPTree(String modelpath) {
		REPTree model = new REPTree();
		try {
			model = (REPTree) SerializationHelper.read(modelpath);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return model;
	}

	public Classifier loadModelM5P(String modelpath) {
		M5P model = new M5P();
		try {
			model = (M5P) SerializationHelper.read(modelpath);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return model;
	}

	public void savePredicted(Evaluation eval, String predictedPath) throws IOException {
		try (Writer writer = Files.newBufferedWriter(Paths.get(predictedPath + "/result_score.csv"));

				CSVWriter csvWriter = new CSVWriter(writer, CSVWriter.DEFAULT_SEPARATOR, CSVWriter.NO_QUOTE_CHARACTER,
						CSVWriter.DEFAULT_ESCAPE_CHARACTER, CSVWriter.DEFAULT_LINE_END);) {
			String[] headerRecord = { "Predicted", "Actual" };
			csvWriter.writeNext(headerRecord);

			ArrayList<Prediction> results = eval.predictions();
			for (int i = 0; i < results.size(); i++) {
				csvWriter.writeNext(new String[] { Double.toString(results.get(i).predicted()),
						Double.toString(results.get(i).actual()) });
			}
		}
	}

	public void savePredictedWithId(Instances testData, Evaluation eval, String predictedPath) throws IOException {
		try (Writer writer = Files.newBufferedWriter(Paths.get(predictedPath + "/result_Id_score.csv"));

				CSVWriter csvWriter = new CSVWriter(writer, CSVWriter.DEFAULT_SEPARATOR, CSVWriter.NO_QUOTE_CHARACTER,
						CSVWriter.DEFAULT_ESCAPE_CHARACTER, CSVWriter.DEFAULT_LINE_END);) {
			String[] headerRecord = { "newsId1", "newsId2", "Predicted", "Actual" };
			csvWriter.writeNext(headerRecord);

			ArrayList<Prediction> results = eval.predictions();
			for (int i = 0; i < results.size(); i++) {
				csvWriter.writeNext(new String[] { testData.get(i).toString(0), testData.get(i).toString(1),
						Double.toString(results.get(i).predicted()), Double.toString(results.get(i).actual()) });
			}
		}
	}

	public void saveSortedResult(String resultPath, HashMap<String, List<Instance>> sortedResult) throws IOException {
		// HashMap<String, List<Instance>> sortedResult =
		// sortResultByAttribute(resultPath+"/result_Id_score.csv",2);
		try (Writer writer = Files.newBufferedWriter(Paths.get(resultPath + "/result_Id_score_sorted.csv"));

				CSVWriter csvWriter = new CSVWriter(writer, CSVWriter.DEFAULT_SEPARATOR, CSVWriter.NO_QUOTE_CHARACTER,
						CSVWriter.DEFAULT_ESCAPE_CHARACTER, CSVWriter.DEFAULT_LINE_END);) {
			String[] headerRecord = { "newsId1", "newsId2", "Predicted", "Actual" };
			csvWriter.writeNext(headerRecord);

			for (String newsId1 : sortedResult.keySet()) {
				List<Instance> listInstance = sortedResult.get(newsId1);
				for (int i = 0; i < listInstance.size(); i++) {
					Instance record = listInstance.get(i);
					csvWriter.writeNext(
							new String[] { newsId1, record.toString(1), record.toString(2), record.toString(3) });
				}

			}
		}
	}

	public HashMap<String, List<Instance>> sortResultByAttribute(String resultPath, int attIndex) {
		Instances resultData = loadDatasetWithId(resultPath);
		resultData.sort(attIndex);
		HashMap<String, List<Instance>> sortedResult = new HashMap<>();
		for (int i = resultData.numInstances() - 1; i >= 0; i--) {
			Instance record = resultData.get(i);
			String newsId1 = record.toString(0);
			if (sortedResult.keySet().contains(newsId1)) {
				sortedResult.get(newsId1).add(record);
			} else {
				List<Instance> listInstance = new ArrayList<>();
				listInstance.add(record);
				sortedResult.put(newsId1, listInstance);
			}
		}
		return sortedResult;
	}

	public void saveEvaluation(Evaluation eval, HashMap<String, List<Instance>> sortedResult, String evalPath) {
		try (PrintWriter out = new PrintWriter(evalPath + "/eval.txt")) {
			out.println("Time: " + eval.totalCost() + "\n" + eval.toSummaryString());
			out.println("MRR: " + calculateMRR(sortedResult));
			out.println();
			try {
				// out.println(eval.toClassDetailsString());
				List<Double> eval2 = evaluation(evalPath);
				out.print("TP = " + eval2.get(0) + "\t");
				out.println("FP = " + eval2.get(1));
				out.print("FN = " + eval2.get(2) + "\t");
				out.println("TN = " + eval2.get(3));
				out.println("-------------------\n");

				out.println("Accuracy:\t" + eval2.get(4) + "\n");
				out.println("Class\t\t Precision\t\t\t Recall\t\t\t\t\t F1");
				out.println("1\t\t " + eval2.get(5) + "\t\t" + eval2.get(6) + "\t\t" + eval2.get(7));
				out.println("0\t\t " + eval2.get(8) + "\t\t" + eval2.get(9) + "\t\t" + eval2.get(10));
				out.println("AVG\t\t " + eval2.get(11) + "\t\t" + eval2.get(12) + "\t\t" + eval2.get(13));
			} catch (Exception e) {
				e.printStackTrace();
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public void saveEvaluationNoTime(String evalPath, HashMap<String, List<Instance>> sortedResult) {
		try (PrintWriter out = new PrintWriter(evalPath + "/eval.txt")) {
			try {
				// out.println(eval.toClassDetailsString());
				out.println("MRR: " + calculateMRR(sortedResult));
				out.println();
				List<Double> eval2 = evaluation(evalPath);
				out.print("TP = " + eval2.get(0) + "\t");
				out.println("FP = " + eval2.get(1));
				out.print("FN = " + eval2.get(2) + "\t");
				out.println("TN = " + eval2.get(3));
				out.println("-------------------\n");

				out.println("Accuracy:\t" + eval2.get(4) + "\n");
				out.println("Class\t\t Precision\t\t\t Recall\t\t\t\t\t F1");
				out.println("1\t\t " + eval2.get(5) + "\t\t" + eval2.get(6) + "\t\t" + eval2.get(7));
				out.println("0\t\t " + eval2.get(8) + "\t\t" + eval2.get(9) + "\t\t" + eval2.get(10));
				out.println("AVG\t\t " + eval2.get(11) + "\t\t" + eval2.get(12) + "\t\t" + eval2.get(13));
			} catch (Exception e) {
				e.printStackTrace();
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public void saveEvaluationTopK(String evalPath, HashMap<String, List<Instance>> sortedResult, int topK,
			double cut_off) throws IOException {
		try (Writer writer = Files.newBufferedWriter(Paths.get(evalPath + "/evalTopK.csv"));

				CSVWriter csvWriter = new CSVWriter(writer, CSVWriter.DEFAULT_SEPARATOR, CSVWriter.NO_QUOTE_CHARACTER,
						CSVWriter.DEFAULT_ESCAPE_CHARACTER, CSVWriter.DEFAULT_LINE_END);) {
			String[] headerRecord = { "newsId", "Predict Positive", "Actual Positive", "True Positive", "Precision",
					"Recall", "F1" };
			csvWriter.writeNext(headerRecord);

			List<List<String[]>> eval = evaluationTopK(sortedResult, topK, cut_off);
			List<String[]> evalTopK = eval.get(0);
			List<String[]> evalAvg = eval.get(1);

			String[] avg = evalAvg.get(0);
			csvWriter.writeNext(new String[] { "0", "0", "0", "0", avg[0], avg[1], avg[2] });

			for (int i = 0; i < evalTopK.size(); i++) {
				String[] records = evalTopK.get(i);
				csvWriter.writeNext(records);
			}

		}
	}

	public void saveNDCGTopK(String evalPath, HashMap<String, List<Instance>> sortedResultByPredictedScore,
			HashMap<String, List<Instance>> sortedResultByActualScore, int topK, double cut_off) throws IOException {
		try (Writer writer = Files.newBufferedWriter(Paths.get(evalPath + "/NDCGTopK.csv"));

				CSVWriter csvWriter = new CSVWriter(writer, CSVWriter.DEFAULT_SEPARATOR, CSVWriter.NO_QUOTE_CHARACTER,
						CSVWriter.DEFAULT_ESCAPE_CHARACTER, CSVWriter.DEFAULT_LINE_END);) {
			String[] headerRecord = { "newsId", "NDCG" };
			csvWriter.writeNext(headerRecord);

			List<String[]> listNDCGTopK = calculateNDCGTopK(sortedResultByPredictedScore, sortedResultByActualScore,
					topK, cut_off);

			for (int i = 0; i < listNDCGTopK.size(); i++) {
				String[] records = listNDCGTopK.get(i);
				csvWriter.writeNext(records);
			}
		}
	}

	public void showTree(REPTree model, String treePath) throws Exception {
		// display classifier
		final javax.swing.JFrame jf = new javax.swing.JFrame("Weka Classifier Tree Visualizer: REPTree");
		jf.setSize(500, 400);
		jf.getContentPane().setLayout(new BorderLayout());
		TreeVisualizer tv = new TreeVisualizer(null, model.graph(), new PlaceNode2());
		jf.getContentPane().add(tv, BorderLayout.CENTER);
		jf.addWindowListener(new java.awt.event.WindowAdapter() {
			public void windowClosing(java.awt.event.WindowEvent e) {
				jf.dispose();
			}
		});
		jf.setVisible(true);
		tv.fitToScreen();
		// save tree to file text
		try (PrintWriter out = new PrintWriter(treePath + "/tree.dot")) {
			out.println(model.graph());
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public void showTree(M5P model, String treePath) throws Exception {
		// display classifier
		final javax.swing.JFrame jf = new javax.swing.JFrame("Weka Classifier Tree Visualizer: M5P");
		jf.setSize(500, 400);
		jf.getContentPane().setLayout(new BorderLayout());
		TreeVisualizer tv = new TreeVisualizer(null, model.graph(), new PlaceNode2());
		jf.getContentPane().add(tv, BorderLayout.CENTER);
		jf.addWindowListener(new java.awt.event.WindowAdapter() {
			public void windowClosing(java.awt.event.WindowEvent e) {
				jf.dispose();
			}
		});
		jf.setVisible(true);
		tv.fitToScreen();
		// save tree to file text
		try (PrintWriter out = new PrintWriter(treePath + "/tree.dot")) {
			out.println(model.graph());
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public void convertScoreToLabel(String scorePath, String labelPath, double cut_off) {
		try (Reader reader = Files.newBufferedReader(Paths.get(scorePath + "/result_score.csv"));
				CSVReader csvReader = new CSVReader(reader);
				Writer writer = Files.newBufferedWriter(Paths.get(labelPath + "/result_label.csv"));
				CSVWriter csvWriter = new CSVWriter(writer, CSVWriter.DEFAULT_SEPARATOR, CSVWriter.NO_QUOTE_CHARACTER,
						CSVWriter.DEFAULT_ESCAPE_CHARACTER, CSVWriter.DEFAULT_LINE_END);) {
			// Reading Records One by One in a String array
			String[] nextRecord;
			nextRecord = csvReader.readNext();
			String[] headerRecord = { "Predicted", "Actual" };
			csvWriter.writeNext(headerRecord);
			while ((nextRecord = csvReader.readNext()) != null) {
				String predictedScore = nextRecord[0];
				String actualScore = nextRecord[1];
				double pScore = Double.parseDouble(predictedScore);
				double aScore = Double.parseDouble(actualScore);
				String predictedLabel = "";
				String actualLabel = "";
				if (pScore >= cut_off)
					predictedLabel = "1";
				else
					predictedLabel = "0";
				if (aScore >= cut_off)
					actualLabel = "1";
				else
					actualLabel = "0";
				csvWriter.writeNext(new String[] { predictedLabel, actualLabel });
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	public void convertScoreToLabelWithId(String scorePath, String labelPath, double cut_off) {
		try (Reader reader = Files.newBufferedReader(Paths.get(scorePath + "/result_Id_score.csv"));
				CSVReader csvReader = new CSVReader(reader);
				Writer writer = Files.newBufferedWriter(Paths.get(labelPath + "/result_Id_label.csv"));
				CSVWriter csvWriter = new CSVWriter(writer, CSVWriter.DEFAULT_SEPARATOR, CSVWriter.NO_QUOTE_CHARACTER,
						CSVWriter.DEFAULT_ESCAPE_CHARACTER, CSVWriter.DEFAULT_LINE_END);) {
			// Reading Records One by One in a String array
			String[] nextRecord;
			nextRecord = csvReader.readNext();
			String[] headerRecord = { "newsId1", "newsId2", "Predicted", "Actual" };
			csvWriter.writeNext(headerRecord);
			while ((nextRecord = csvReader.readNext()) != null) {
				String newsId1 = nextRecord[0];
				String newsId2 = nextRecord[1];
				String predictedScore = nextRecord[2];
				String actualScore = nextRecord[3];
				double pScore = Double.parseDouble(predictedScore);
				double aScore = Double.parseDouble(actualScore);
				String predictedLabel = "";
				String actualLabel = "";
				if (pScore >= cut_off)
					predictedLabel = "1";
				else
					predictedLabel = "0";
				if (aScore > 0)
					actualLabel = "1";
				else
					actualLabel = "0";
				csvWriter.writeNext(new String[] { newsId1, newsId2, predictedLabel, actualLabel });
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	public List<Double> evaluation(String resultPath) {
		List<Double> eval = new ArrayList<>();
		try {
			Reader reader = Files.newBufferedReader(Paths.get(resultPath + "/result_Id_label.csv"));
			CSVReader csvReader = new CSVReader(reader);
			// Reading All Records at once into a List<String[]>
			int TP = 0;
			int FP = 0;
			int TN = 0;
			int FN = 0;
			List<String[]> records = csvReader.readAll();
			for (String[] record : records) {
				String predicted = record[2];
				String actual = record[3];
				if (predicted.equals("1") && actual.equals("1"))
					TP = TP + 1;
				if (predicted.equals("1") && actual.equals("0"))
					FP = FP + 1;
				if (predicted.equals("0") && actual.equals("0"))
					TN = TN + 1;
				if (predicted.equals("0") && actual.equals("1"))
					FN = FN + 1;
			}
			double accuracy = (double) (TP + TN) / (TP + TN + FP + FN);
			double precisionP = (double) TP / (TP + FP);
			double recallP = (double) TP / (TP + FN);
			double f1P = (double) 2 * precisionP * recallP / (precisionP + recallP);

			double precisionN = (double) TN / (TN + FN);
			double recallN = (double) TN / (TN + FP);
			double f1N = (double) 2 * precisionN * recallN / (precisionN + recallN);

			double precisionAvg = (precisionP + precisionN) / 2;
			double recallAvg = (recallP + recallN) / 2;
			double f1Avg = (f1P + f1N) / 2;

			eval.add((double) TP);
			eval.add((double) FP);
			eval.add((double) FN);
			eval.add((double) TN);

			eval.add(accuracy);
			eval.add(precisionP);
			eval.add(recallP);
			eval.add(f1P);
			eval.add(precisionN);
			eval.add(recallN);
			eval.add(f1N);

			eval.add(precisionAvg);
			eval.add(recallAvg);
			eval.add(f1Avg);

		} catch (IOException e) {
			e.printStackTrace();
		}
		return eval;
	}

	public double calculateMRR(HashMap<String, List<Instance>> sortedResult) {
		double mrrScore = 0;
		for (String newsId1 : sortedResult.keySet()) {
			List<Instance> records = sortedResult.get(newsId1);
			double rr = 0;
			for (int i = 0; i < records.size(); i++) {
				Instance record = records.get(i);
				double actualScore = record.value(3);
				if (actualScore > 0) {
					rr = (double) 1 / (i + 1);
					break;
				}
			}
			mrrScore = mrrScore + rr;
		}
		mrrScore = mrrScore / sortedResult.size();
		return mrrScore;
	}

	public List<String[]> calculateNDCGTopK(HashMap<String, List<Instance>> sortedResultByPredictedScore,
			HashMap<String, List<Instance>> sortedResultByActualScore, int topK, double cut_off) throws IOException {
		List<String[]> listNDCGTopK = new ArrayList<>();

		for (String newsId1 : sortedResultByPredictedScore.keySet()) {
			List<Instance> recordsDCG = sortedResultByPredictedScore.get(newsId1);
			List<Instance> recordsIDCG = sortedResultByActualScore.get(newsId1);
			int maxIndex = topK;
			double dcgp = 0;
			double idcgp = 0;
			double ndcgp = 0;

			if (topK > recordsDCG.size())
				maxIndex = recordsDCG.size();

			// Tinh DCGp
			for (int i = 0; i < maxIndex; i++) {
				Instance record = recordsDCG.get(i);
				double reli = record.value(3);
				double dcg = reli / (Math.log(i + 2) / Math.log(2));
				dcgp = dcgp + dcg;
			}
			// Tinh iDCGp
			for (int i = 0; i < maxIndex; i++) {
				Instance record = recordsIDCG.get(i);
				double reli = record.value(3);
				double idcg = reli / (Math.log(i + 2) / Math.log(2));
				idcgp = idcgp + idcg;
			}
			// Tinh nDCGp
			if (dcgp == 0)
				ndcgp = 0;
			else
				ndcgp = dcgp / idcgp;
			listNDCGTopK.add(new String[] { newsId1, Double.toString(ndcgp) });
		}
		return listNDCGTopK;
	}

	public List<List<String[]>> evaluationTopK(HashMap<String, List<Instance>> sortedResult, int topK, double cut_off)
			throws IOException {
		List<String[]> listEvalTopK = new ArrayList<>();
		List<String[]> listEvalAvg = new ArrayList<>();

		double avgPrecision = 0;
		double avgRecall = 0;
		double avgF1 = 0;
		int nPositive = 0;

		for (String newsId1 : sortedResult.keySet()) {
			List<Instance> records = sortedResult.get(newsId1);
			int predictPositive = 0;
			int maxIndex = topK;
			if (topK > records.size())
				maxIndex = records.size();
			int actualPositive = 0;
			int truePositive = 0;

			for (int i = 0; i < maxIndex; i++) {
				Instance record = records.get(i);
				double predictScore = record.value(2);
				if (predictScore >= cut_off)
					predictPositive = predictPositive + 1;
			}

			for (int i = 0; i < records.size(); i++) {
				Instance record = records.get(i);
				double actualScore = record.value(3);
				if (actualScore > 0)
					actualPositive = actualPositive + 1;
				if (actualScore > 0 && i < predictPositive)
					truePositive = truePositive + 1;
			}

			double precisionTopK = 0;
			if (predictPositive > 0)
				precisionTopK = (double) truePositive / predictPositive;

			double recallTopK = 0;
			if (actualPositive > 0)
				recallTopK = (double) truePositive / actualPositive;

			double f1TopK = 0;
			if (precisionTopK > 0 && recallTopK > 0)
				f1TopK = 2 * precisionTopK * recallTopK / (precisionTopK + recallTopK);

			String[] evalTopK = new String[] { newsId1, Double.toString(predictPositive),
					Double.toString(actualPositive), Double.toString(truePositive), Double.toString(precisionTopK),
					Double.toString(recallTopK), Double.toString(f1TopK) };
			listEvalTopK.add(evalTopK);

			if (actualPositive > 0 || predictPositive > 0) {
				avgPrecision = avgPrecision + precisionTopK;
				avgRecall = avgRecall + recallTopK;
				avgF1 = avgF1 + f1TopK;
				nPositive = nPositive + 1;
			}
		}

		avgPrecision = avgPrecision / nPositive;
		avgRecall = avgRecall / nPositive;
		avgF1 = avgF1 / nPositive;

		listEvalAvg.add(
				new String[] { Double.toString(avgPrecision), Double.toString(avgRecall), Double.toString(avgF1) });

		List<List<String[]>> listEval = new ArrayList<>();
		listEval.add(listEvalTopK);
		listEval.add(listEvalAvg);
		return listEval;
	}

	public void createTrainAndTestDataset(String dataPath, String outPath) throws Exception {
		long startTime;
		long endTime;
		long totalTime;

		Instances originalData = loadDatasetWithId(dataPath);

		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedData = prep.Numeric2Nominal(originalData, "1,2");
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// divide dataset to train dataset 60% and test dataset 40%
		int trainSize = (int) Math.round(preprocessedData.numInstances() * 0.6);
		int testSize = preprocessedData.numInstances() - trainSize;

		preprocessedData.randomize(new Debug.Random(1));

		System.out.println("Deviding dataset ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Instances traindataset = new Instances(preprocessedData, 0, trainSize);
		Instances testdataset = new Instances(preprocessedData, trainSize, testSize);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		CSVSaver saverTrain = new CSVSaver();
		saverTrain.setInstances(traindataset);
		saverTrain.setFile(new File(outPath + "/Train_graph_ne_1992.csv"));
		saverTrain.setNoHeaderRow(false);
		saverTrain.writeBatch();

		CSVSaver saverTest = new CSVSaver();
		saverTest.setInstances(testdataset);
		saverTest.setFile(new File(outPath + "/Test_graph_ne_1992.csv"));
		saverTest.setNoHeaderRow(false);
		saverTest.writeBatch();
	}

	/**
	 * JUST FOR TESTING CODE
	 * 
	 * @throws IOException
	 */
	public static void testL2R() throws IOException {
		// Danh gia Learning to rank
		ModelGenerator mg = new ModelGenerator();
		// String RESULTPATH =
		// "C:/Users/ADMIN/Desktop/Demo/result_L2R/04_09_2018/dataset1/event";
		// String RESULTPATH =
		// "C:/Users/ADMIN/Desktop/Demo/result_L2R/04_09_2018/dataset1/topic";
		// String RESULTPATH =
		// "C:/Users/ADMIN/Desktop/Demo/result_L2R/04_09_2018/dataset1/ne";

		// String RESULTPATH =
		// "C:/Users/ADMIN/Desktop/Demo/result_L2R/04_09_2018/dataset2/event";
		// String RESULTPATH =
		// "C:/Users/ADMIN/Desktop/Demo/result_L2R/04_09_2018/dataset2/topic";
		// String RESULTPATH =
		// "C:/Users/ADMIN/Desktop/Demo/result_L2R/04_09_2018/dataset2/ne";

		// String RESULTPATH =
		// "C:/Users/ADMIN/Desktop/Demo/result_L2R/04_09_2018/dataset3/event";
		// String RESULTPATH =
		// "C:/Users/ADMIN/Desktop/Demo/result_L2R/04_09_2018/dataset3/topic";
		String RESULTPATH = "C:/Users/ADMIN/Desktop/Demo/result_L2R/04_09_2018/dataset3/ne";
		mg.convertScoreToLabelWithId(RESULTPATH, RESULTPATH, 0);

		// Sort result
		HashMap<String, List<Instance>> sortedResultByPredictedScore = mg
				.sortResultByAttribute(RESULTPATH + "/result_Id_score.csv", 2);
		mg.saveSortedResult(RESULTPATH, sortedResultByPredictedScore);
		HashMap<String, List<Instance>> sortedResultByActualScore = mg
				.sortResultByAttribute(RESULTPATH + "/result_Id_score.csv", 3);

		mg.saveEvaluationNoTime(RESULTPATH, sortedResultByPredictedScore);
		mg.saveEvaluationTopK(RESULTPATH, sortedResultByPredictedScore, 2, 0);
		mg.saveNDCGTopK(RESULTPATH, sortedResultByPredictedScore, sortedResultByActualScore, 2, 0);
		// ----//
	}

	/**
	 * JUST FOR TESTING CODE
	 * 
	 * @throws IOException
	 */
	public static void testTFIDF() throws IOException {
		// Danh gia TF-IDF
		ModelGenerator mg = new ModelGenerator();
		String RESULTPATH = "C:/Users/ADMIN/Desktop/Demo/result_tfidf/04_09_2018/dataset1/TFIDF1/event";
		// String RESULTPATH =
		// "C:/Users/ADMIN/Desktop/Demo/result_tfidf/04_09_2018/dataset1/TFIDF1/topic";
		// String RESULTPATH =
		// "C:/Users/ADMIN/Desktop/Demo/result_tfidf/04_09_2018/dataset1/TFIDF1/ne";

		// String RESULTPATH =
		// "C:/Users/ADMIN/Desktop/Demo/result_tfidf/04_09_2018/dataset2/TFIDF2/event";
		// String RESULTPATH =
		// "C:/Users/ADMIN/Desktop/Demo/result_tfidf/04_09_2018/dataset2/TFIDF2/topic";
		// String RESULTPATH =
		// "C:/Users/ADMIN/Desktop/Demo/result_tfidf/04_09_2018/dataset2/TFIDF2/ne";

		// String RESULTPATH =
		// "C:/Users/ADMIN/Desktop/Demo/result_tfidf/04_09_2018/dataset3/TFIDF3/event";
		// String RESULTPATH =
		// "C:/Users/ADMIN/Desktop/Demo/result_tfidf/04_09_2018/dataset3/TFIDF3/topic";
		// String RESULTPATH =
		// "C:/Users/ADMIN/Desktop/Demo/result_tfidf/04_09_2018/dataset3/TFIDF3/ne";

		mg.convertScoreToLabelWithId(RESULTPATH, RESULTPATH, 0);

		// Sort result
		HashMap<String, List<Instance>> sortedResultByPredictedScore = mg
				.sortResultByAttribute(RESULTPATH + "/result_Id_score.csv", 2);
		mg.saveSortedResult(RESULTPATH, sortedResultByPredictedScore);
		HashMap<String, List<Instance>> sortedResultByActualScore = mg
				.sortResultByAttribute(RESULTPATH + "/result_Id_score.csv", 3);

		mg.saveEvaluationNoTime(RESULTPATH, sortedResultByPredictedScore);
		mg.saveEvaluationTopK(RESULTPATH, sortedResultByPredictedScore, 2, 0);
		mg.saveNDCGTopK(RESULTPATH, sortedResultByPredictedScore, sortedResultByActualScore, 2, 0);
		// ----//
	}

	/**
	 * JUST FOR TESTING CODE
	 * 
	 * @throws IOException
	 */
	public static void testNDCG() throws IOException {
		ModelGenerator mg = new ModelGenerator();
		HashMap<String, List<Instance>> sortedResultByPredictedScore = mg
				.sortResultByAttribute("C:/Users/ADMIN/Desktop/testNDCG.csv", 2);
		HashMap<String, List<Instance>> sortedResultByActualScore = mg
				.sortResultByAttribute("C:/Users/ADMIN/Desktop/testNDCG.csv", 3);
		List<String[]> listResult = mg.calculateNDCGTopK(sortedResultByPredictedScore, sortedResultByActualScore, 10,
				0);
		for (int i = 0; i < listResult.size(); i++) {
			System.out.println(listResult.get(i)[0] + "-->" + listResult.get(i)[1]);
		}

	}

	public static void main(String[] args) throws Exception {
		ModelGenerator mg = new ModelGenerator();
		System.out.println("Start");
		// Chia du lieu Train va Test
		// String dataPath =
		// "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset3/features_ne_1992.csv";
		// String outPath =
		// "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset3/Train_Test";
		// mg.createTrainAndTestDataset(dataPath, outPath);
		// ----//

		// Test MRR
		// HashMap<String, List<Instance>> sortedResult =
		// mg.sortResultByAttribute("C:/Users/ADMIN/Desktop/testMRR.csv", 2);
		// System.out.println(mg.calculateMRR(sortedResult));
		// ----//

		// testL2R();

		// testNDCG();

		System.out.println("(((o(*ﾟ▽ﾟ*)o)))");

	}

}
