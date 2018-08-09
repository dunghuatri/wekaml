package admicro.wekaml;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.opencsv.CSVWriter;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.REPTree;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.TextDirectoryLoader;

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
		 rt.setMinVarianceProp(0.8);
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
			SerializationHelper.write(modelpath, model);
		} catch (Exception ex) {
			Logger.getLogger(ModelGenerator.class.getName()).log(Level.SEVERE, null, ex);
		}
	}
	
	public Classifier loadModel(String modelpath)
	{
		REPTree model = new REPTree();
		try {
			model = (REPTree) SerializationHelper.read(modelpath);
		} catch (Exception e) {			
			e.printStackTrace();
		}
		return model;
	}

	public void savePredicted(Evaluation eval, String predictedPath) throws IOException {
		try (Writer writer = Files.newBufferedWriter(Paths.get(predictedPath));

				CSVWriter csvWriter = new CSVWriter(writer, CSVWriter.DEFAULT_SEPARATOR, CSVWriter.NO_QUOTE_CHARACTER,
						CSVWriter.DEFAULT_ESCAPE_CHARACTER, CSVWriter.DEFAULT_LINE_END);) {
			String[] headerRecord = { "Predicted", "Actual" };
			csvWriter.writeNext(headerRecord);

			ArrayList<Prediction> results = eval.predictions();
			for (int i = 0; i < results.size(); i++) {
				csvWriter.writeNext(
						new String[] { Double.toString(results.get(i).predicted()) , Double.toString(results.get(i).actual()) });
			}
		}
	}
	
	public void saveEvaluation(Evaluation eval, String evalPath)
	{
		try (PrintWriter out = new PrintWriter(evalPath)) {			
		    out.println("Time: "+eval.totalCost()+"\n"+eval.toSummaryString());
		} catch (FileNotFoundException e) {			
			e.printStackTrace();
		}
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
