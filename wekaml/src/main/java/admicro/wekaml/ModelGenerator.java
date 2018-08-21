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
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.REPTree;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.SerializationHelper;
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

	public Classifier buildClassifier(Instances traindataset) {
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
		rt.setNumFolds(10);
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
			SerializationHelper.write(modelpath+"/REPTree_model.bin", model);
		} catch (Exception ex) {
			Logger.getLogger(ModelGenerator.class.getName()).log(Level.SEVERE, null, ex);
		}
	}

	public Classifier loadModel(String modelpath) {
		REPTree model = new REPTree();
		try {
			model = (REPTree) SerializationHelper.read(modelpath+"/REPTree_model.bin");
		} catch (Exception e) {
			e.printStackTrace();
		}
		return model;
	}

	public void savePredicted(Evaluation eval, String predictedPath) throws IOException {
		try (Writer writer = Files.newBufferedWriter(Paths.get(predictedPath+"/result_score.csv"));

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

	public void saveEvaluation(Evaluation eval, String evalPath) {
		try (PrintWriter out = new PrintWriter(evalPath+"/eval.txt")) {
			out.println("Time: " + eval.totalCost() + "\n" + eval.toSummaryString());
			try {
				// out.println(eval.toClassDetailsString());
				List<Double> eval2 = evaluation(evalPath);
				out.print("TP = "+eval2.get(0)+"\t");
				out.println("FP = "+eval2.get(1));
				out.print("FN = "+eval2.get(2)+"\t");
				out.println("TN = "+eval2.get(3));
				out.println("-------------------\n");
				
				out.println("Accuracy:\t"+eval2.get(4)+"\n");
				out.println("Class\t\t Precision\t\t\t Recall\t\t\t\t\t F1");
				out.println("1\t\t "+eval2.get(5)+"\t\t"+eval2.get(6)+"\t\t"+eval2.get(7));
				out.println("0\t\t "+eval2.get(8)+"\t\t"+eval2.get(9)+"\t\t"+eval2.get(10));
				out.println("AVG\t\t "+eval2.get(11)+"\t\t"+eval2.get(12)+"\t\t"+eval2.get(13));
			} catch (Exception e) {
				e.printStackTrace();
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
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
		try (PrintWriter out = new PrintWriter(treePath+"/tree.dot")) {
			out.println(model.graph());
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public void convertScoreToLabel(String scorePath, String labelPath, double cut_off) {
		try (
				Reader reader = Files.newBufferedReader(Paths.get(scorePath+"/result_score.csv"));				
				CSVReader csvReader = new CSVReader(reader);
				Writer writer = Files.newBufferedWriter(Paths.get(labelPath+"/result_label.csv"));
				CSVWriter csvWriter = new CSVWriter(writer, CSVWriter.DEFAULT_SEPARATOR, CSVWriter.NO_QUOTE_CHARACTER,
						CSVWriter.DEFAULT_ESCAPE_CHARACTER, CSVWriter.DEFAULT_LINE_END);				
		) {
			// Reading Records One by One in a String array
			String[] nextRecord;
			nextRecord = csvReader.readNext();
			String[] headerRecord = {"Predicted","Actual"};
			csvWriter.writeNext(headerRecord);
			while ((nextRecord = csvReader.readNext()) != null) {				
				String predictedScore = nextRecord[0];
				String actualScore = nextRecord[1];
				double pScore = Double.parseDouble(predictedScore);
				double aScore = Double.parseDouble(actualScore);
				String predictedLabel = "";
				String actualLabel = "";
				if(pScore>=cut_off)				
					predictedLabel = "1";
				else predictedLabel = "0";
				if(aScore>=cut_off)				
					actualLabel = "1";
				else actualLabel = "0";
				csvWriter.writeNext(new String[] { predictedLabel, actualLabel });
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
	
	public List<Double> evaluation(String resultPath)
	{
		List<Double> eval = new ArrayList<>();
		try {
			Reader reader = Files.newBufferedReader(Paths.get(resultPath+"/result_label.csv"));
			CSVReader csvReader = new CSVReader(reader);
			// Reading All Records at once into a List<String[]>
			int TP = 0;
			int FP = 0;
			int TN = 0;
			int FN = 0;
			List<String[]> records = csvReader.readAll();
			for (String[] record : records) {
			   String predicted = record[0];
			   String actual = record[1];
			   if(predicted.equals("1") && actual.equals("1"))			   
				   TP = TP + 1;			   
			   if(predicted.equals("1") && actual.equals("0"))			   
				   FP = FP + 1;
			   if(predicted.equals("0") && actual.equals("0"))			   
				   TN = TN + 1;
			   if(predicted.equals("0") && actual.equals("1"))			   
				   FN = FN + 1;
			}
			double accuracy = (double) (TP+TN)/(TP+TN+FP+FN);
			double precisionP = (double) TP/(TP+FP);
			double recallP = (double) TP/(TP+FN);
			double f1P = (double) 2*precisionP*recallP/(precisionP+recallP);
			
			double precisionN = (double) TN/(TN+FN);
			double recallN = (double) TN/(TN+FP);
			double f1N = (double) 2*precisionN*recallN/(precisionN+recallN);
			
			double precisionAvg = (precisionP+precisionN)/2;
			double recallAvg = (recallP+recallN)/2;
			double f1Avg = (f1P+f1N)/2;
			
			eval.add((double)TP);
			eval.add((double)FP);
			eval.add((double)FN);
			eval.add((double)TN);
			
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
	
	public void createTrainAndTestDataset(String dataPath, String outPath) throws Exception
	{
		long startTime;
		long endTime;
		long totalTime;
		
		Instances originalData = loadDataset(dataPath);
		// divide dataset to train dataset 60% and test dataset 40%
		int trainSize = (int) Math.round(originalData.numInstances() * 0.6);
		int testSize = originalData.numInstances() - trainSize;
		
		originalData.randomize(new Debug.Random(1));

		System.out.println("Deviding dataset ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		Instances traindataset = new Instances(originalData, 0, trainSize);
		Instances testdataset = new Instances(originalData, trainSize, testSize);
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");
		
		CSVSaver saverTrain = new CSVSaver();		
		saverTrain.setInstances(traindataset);
		saverTrain.setFile(new File(outPath+"/Train.csv"));
		saverTrain.setNoHeaderRow(false);
		saverTrain.writeBatch();
		
		CSVSaver saverTest = new CSVSaver();		
		saverTest.setInstances(testdataset);
		saverTest.setFile(new File(outPath+"/Test.csv"));
		saverTest.setNoHeaderRow(false);
		saverTest.writeBatch();
	}
	
	public static void main(String[] args) throws Exception {

		ModelGenerator mg = new ModelGenerator();
		TextDirectoryLoader loader = new TextDirectoryLoader();
		String[] option = new String[2];
		option[0] = "-dir";
		option[1] = "E:/Now/relevant_news/test";
		loader.setOptions(option);
		System.out.println(loader.getDataSet());

	}

}
