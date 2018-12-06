/**
 * 
 */
package admicro.wekaml;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import guru.nidi.graphviz.attribute.Color;
import guru.nidi.graphviz.attribute.Style;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.Graph;
import guru.nidi.graphviz.model.MutableGraph;
import guru.nidi.graphviz.parse.Parser;

import static guru.nidi.graphviz.model.Factory.*;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.REPTree;
import weka.core.Debug;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

/**
 * @author Hua Tri Dung
 *
 */

public class StartWeka extends BaseClass{

	// private static final String DATASETPATH = "data/iris.2D.arff";
	// private static final String DATASETPATH = "data/heart.csv";
	private static String DATASETPATH = "C:/Users/ADMIN/Desktop/Demo/data/features_graph_event.csv";
	// private static String DATASETPATH =
	// "C:/Users/ADMIN/Desktop/Demo/data/features_graph_topic.csv";
	// private static String DATASETPATH =
	// "C:/Users/ADMIN/Desktop/Demo/data/features_graph_ne.csv";
	// private static String MODElPATH = "model/REPTree_model_data1_ne.bin";
	private static String MODElPATH = "result/model.bin";
	private static String RESULTPATH = "result";
	private static String EVALPATH = "result";
	private static String TREEPATH = "result";
	private double cut_off = 0.1;
	private List<Integer> topK = new ArrayList<>();

	public StartWeka() {
		super();
		topK.add(1);
		topK.add(3);
		topK.add(4);
		topK.add(5);
		topK.add(10);
	}

	/**
	 * IN PROCESS
	 * 
	 * @param dataPath
	 * @param folds
	 * @throws Exception
	 */
	public void crossValidationREPTree(String dataPath, int folds) throws Exception {
		// Measure time
		long startTime;
		long endTime;
		long totalTime;

		// Load dataset
		ModelGenerator mg = new ModelGenerator();
		Instances originalData = mg.loadDataset(dataPath);

		int seed = 1;

		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedData = prep.removeFeatures(originalData, "1-2");
		preprocessedData.randomize(new Debug.Random(seed));
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Normalize dataset
		Filter filter = new Normalize();
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		filter.setInputFormat(preprocessedData);
		Instances datasetnor = Filter.useFilter(preprocessedData, filter);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Stratify
		if (datasetnor.classAttribute().isNominal())
			datasetnor.stratify(folds);

		// List result summary
		HashMap<Integer, List<Double>> listResultTopKCV = new HashMap<>();
		for (int topK : this.topK) {
			List<Double> listResult = new ArrayList<>();
			listResultTopKCV.put(topK, listResult);
		}

		// Perform cross-validation
		for (int n = 0; n < folds; n++) {
			System.out.println("Fold " + n);
			// Get the folds
			Instances trainData = datasetnor.trainCV(folds, n);
			Instances testData = datasetnor.testCV(folds, n);

			// Build classifier with train dataset
			System.out.println("Building model ...");
			startTime = System.currentTimeMillis();
			// -------------------------------------------//
			REPTree trainedModel = (REPTree) mg.buildClassifierREPTree(trainData);
			// -------------------------------------------//
			endTime = System.currentTimeMillis();
			totalTime = endTime - startTime;
			System.out.println("done " + totalTime / 1000 + " s");

			// Save model
			System.out.println("Saving model ...");
			startTime = System.currentTimeMillis();
			// -------------------------------------------//
			mg.saveModel(trainedModel, MODElPATH);
			// -------------------------------------------//
			endTime = System.currentTimeMillis();
			totalTime = endTime - startTime;
			System.out.println("done " + totalTime / 1000 + " s");

			// Show tree
			System.out.println("Showing tree ...");
			startTime = System.currentTimeMillis();
			// -------------------------------------------//
			// mg.showTree(trainedModel, TREEPATH);
			// createDotGraph();
			// -------------------------------------------//
			endTime = System.currentTimeMillis();
			totalTime = endTime - startTime;
			System.out.println("done " + totalTime / 1000 + " s");

			// Load model
			System.out.println("Loading model ...");
			startTime = System.currentTimeMillis();
			// -------------------------------------------//
			REPTree loadedModel = (REPTree) mg.loadModelREPTree(MODElPATH);
			// -------------------------------------------//
			endTime = System.currentTimeMillis();
			totalTime = endTime - startTime;
			System.out.println("done " + totalTime / 1000 + " s");

			// Evaluate classifier with test dataset
			System.out.println("Evaluating ...");
			startTime = System.currentTimeMillis();
			// -------------------------------------------//
			Evaluation eval = mg.evaluateModel(loadedModel, trainData, testData);
			// -------------------------------------------//
			endTime = System.currentTimeMillis();
			totalTime = endTime - startTime;
			System.out.println("done " + totalTime / 1000 + " s");

			// Save predicted results
			System.out.println("Saving results ...");
			startTime = System.currentTimeMillis();
			// -------------------------------------------//
			mg.savePredictedWithId(testData, eval,
					RESULTPATH + "/Fold_" + Integer.toString(n) + "_result_Id_score.csv");
			mg.convertScoreToLabelWithId(RESULTPATH + "/Fold_" + Integer.toString(n) + "_result_Id_score.csv",
					RESULTPATH + "/Fold_" + Integer.toString(n) + "_result_Id_label.csv", cut_off);
			// -------------------------------------------//
			endTime = System.currentTimeMillis();
			totalTime = endTime - startTime;
			System.out.println("done " + totalTime / 1000 + " s");

			// Save evaluation
			System.out.println("Saving evaluation ...");
			startTime = System.currentTimeMillis();
			// -------------------------------------------//
			// Sort result
			HashMap<String, List<Instance>> sortedResultByPredictedScore = mg
					.sortResultByAttribute(EVALPATH + "/Fold_" + Integer.toString(n) + "_result_Id_score.csv", 2);
			mg.saveSortedResult(EVALPATH + "/Fold_" + Integer.toString(n) + "_result_Id_score_sorted.csv",
					sortedResultByPredictedScore);
			HashMap<String, List<Instance>> sortedResultByActualScore = mg
					.sortResultByAttribute(EVALPATH + "/Fold_" + Integer.toString(n) + "_result_Id_score.csv", 3);
			HashMap<Integer, List<String>> listResultTopK = new HashMap<>();
			for (int topK : this.topK) {
				List<String> listResult = new ArrayList<>();
				listResultTopK.put(topK, listResult);
				mg.saveEvaluationTopK(
						EVALPATH + "/Fold_" + Integer.toString(n) + "_evalTopK_" + Integer.toString(topK) + ".csv",
						sortedResultByPredictedScore, topK, cut_off, listResultTopK, 2, 3);
				mg.saveNDCGTopK(
						EVALPATH + "/Fold_" + Integer.toString(n) + "_NDCGTopK" + Integer.toString(topK) + ".csv",
						sortedResultByPredictedScore, sortedResultByActualScore, topK, cut_off, 3, listResultTopK);
				mg.saveEvaluation(eval, sortedResultByPredictedScore,
						EVALPATH + "/Fold_" + Integer.toString(n) + "_eval_" + Integer.toString(topK) + ".txt",
						RESULTPATH + "/Fold_" + Integer.toString(n) + "_result_Id_label.csv", topK, listResultTopK, 2, 3);
			}
			mg.saveEvaluationSumary(EVALPATH + "/Fold_" + Integer.toString(n) + "_evalSumary.csv", listResultTopK);
			mg.evaluationSumaryCV(listResultTopK, listResultTopKCV);
			// -------------------------------------------//
			endTime = System.currentTimeMillis();
			totalTime = endTime - startTime;
			System.out.println("done " + totalTime / 1000 + " s");
		}
		mg.saveEvaluationSumaryCV(EVALPATH + "/evalSumaryCV.csv", listResultTopKCV, folds);
	}

	public void trainREPTreeLeaveOneFeatureOut(String trainDataPath, int featureIndex, String modelPath)
			throws Exception {
		/* Measure time */
		long startTime;
		long endTime;
		long totalTime;

		ModelGenerator mg = new ModelGenerator();
		Instances trainData = mg.loadDataset(trainDataPath);

		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedData = prep.removeFeatures(trainData, "1,2," + Integer.toString(featureIndex));
		preprocessedData.randomize(new Debug.Random(1));
		// Instances preprocessedData = trainData;
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		Filter filter = new Normalize();

		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		filter.setInputFormat(preprocessedData);
		Instances datasetnor = Filter.useFilter(preprocessedData, filter);
		// Instances datasetnor = preprocessedData;
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// build classifier with train dataset
		System.out.println("Building model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		REPTree trainedModel = (REPTree) mg.buildClassifierREPTree(datasetnor);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save model
		System.out.println("Saving model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.saveModel(trainedModel, modelPath + "/model.bin");
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Show tree
		System.out.println("Showing tree ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.showTree(trainedModel, modelPath);
		// createDotGraph(modelPath);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");
	}

	public void trainM5PLeaveOneFeatureOut(String trainDataPath, int featureIndex, String modelPath) throws Exception {
		/* Measure time */
		long startTime;
		long endTime;
		long totalTime;

		ModelGenerator mg = new ModelGenerator();
		Instances trainData = mg.loadDataset(trainDataPath);

		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedData = prep.removeFeatures(trainData, "1,2," + Integer.toString(featureIndex));
		preprocessedData.randomize(new Debug.Random(1));
		// Instances preprocessedData = trainData;
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		Filter filter = new Normalize();

		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		filter.setInputFormat(preprocessedData);
		Instances datasetnor = Filter.useFilter(preprocessedData, filter);
		// Instances datasetnor = preprocessedData;
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// build classifier with train dataset
		System.out.println("Building model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		M5P trainedModel = (M5P) mg.buildClassifierM5P(datasetnor);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save model
		System.out.println("Saving model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.saveModel(trainedModel, modelPath + "/model.bin");
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Show tree
		System.out.println("Showing tree ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.showTree(trainedModel, modelPath);
		// createDotGraph(modelPath);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");
	}

	public void trainSVMLeaveOneFeatureOut(String trainDataPath, String modelPath, String featureIndex) throws Exception {
		/* Measure time */
		long startTime;
		long endTime;
		long totalTime;

		ModelGenerator mg = new ModelGenerator();
		Instances trainData = mg.loadDataset(trainDataPath);

		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedData = prep.removeFeatures(trainData, "1,2," + featureIndex);
		// preprocessedData = prep.Numeric2Nominal(preprocessedData, "last");
		preprocessedData.randomize(new Debug.Random(1));
		// Instances preprocessedData = trainData;
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		Filter filter = new Normalize();

		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		filter.setInputFormat(preprocessedData);
		Instances datasetnor = Filter.useFilter(preprocessedData, filter);
		// Instances datasetnor = preprocessedData;
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// build classifier with train dataset
		System.out.println("Building model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		SMOreg trainedModel = (SMOreg) mg.buildClassifierSVM(datasetnor);
		// SMO trainedModel = (SMO) mg.buildClassifierSVM(datasetnor);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save model
		System.out.println("Saving model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.saveModel(trainedModel, modelPath);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Show tree
		// System.out.println("Showing tree ...");
		// startTime = System.currentTimeMillis();
		// -------------------------------------------//
		// mg.showTree(trainedModel, TREEPATH);
		// createDotGraph(TREEPATH);
		// -------------------------------------------//
		// endTime = System.currentTimeMillis();
		// totalTime = endTime - startTime;
		// System.out.println("done " + totalTime / 1000 + " s");
	}
	
	public void trainSVCLeaveOneFeatureOut(String trainDataPath, String modelPath, String featureIndex) throws Exception {
		/* Measure time */
		long startTime;
		long endTime;
		long totalTime;

		ModelGenerator mg = new ModelGenerator();
		Instances trainData = mg.loadDataset(trainDataPath);

		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedData = prep.removeFeatures(trainData, "1,2," + featureIndex);
		preprocessedData = prep.Numeric2Nominal(preprocessedData, "last");
		preprocessedData.randomize(new Debug.Random(1));
		// Instances preprocessedData = trainData;
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		Filter filter = new Normalize();

		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		filter.setInputFormat(preprocessedData);
//		Instances datasetnor = Filter.useFilter(preprocessedData, filter);
		 Instances datasetnor = preprocessedData;
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// build classifier with train dataset
		System.out.println("Building model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		SMO trainedModel = (SMO) mg.buildClassifierSVC(datasetnor);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save model
		System.out.println("Saving model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.saveModel(trainedModel, modelPath);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");
	}

	public void trainREPTree(String trainDataPath) throws Exception {
		/* Measure time */
		long startTime;
		long endTime;
		long totalTime;

		ModelGenerator mg = new ModelGenerator();
		Instances trainData = mg.loadDataset(trainDataPath);

		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedData = prep.removeFeatures(trainData, "1-2");
		preprocessedData.randomize(new Debug.Random(1));
		// Instances preprocessedData = trainData;
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		Filter filter = new Normalize();

		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		filter.setInputFormat(preprocessedData);
		Instances datasetnor = Filter.useFilter(preprocessedData, filter);
		// Instances datasetnor = preprocessedData;
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// build classifier with train dataset
		System.out.println("Building model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		REPTree trainedModel = (REPTree) mg.buildClassifierREPTree(datasetnor);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save model
		System.out.println("Saving model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.saveModel(trainedModel, MODElPATH);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Show tree
		// System.out.println("Showing tree ...");
		// startTime = System.currentTimeMillis();
		// -------------------------------------------//
		// mg.showTree(trainedModel, TREEPATH);
		// createDotGraph(TREEPATH);
		// -------------------------------------------//
		// endTime = System.currentTimeMillis();
		// totalTime = endTime - startTime;
		// System.out.println("done " + totalTime / 1000 + " s");
	}

	public void trainM5P(String trainDataPath) throws Exception {
		/* Measure time */
		long startTime;
		long endTime;
		long totalTime;

		ModelGenerator mg = new ModelGenerator();
		Instances trainData = mg.loadDataset(trainDataPath);

		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedData = prep.removeFeatures(trainData, "1-2");
		preprocessedData.randomize(new Debug.Random(1));
		// Instances preprocessedData = trainData;
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		Filter filter = new Normalize();

		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		filter.setInputFormat(preprocessedData);
		Instances datasetnor = Filter.useFilter(preprocessedData, filter);
		// Instances datasetnor = preprocessedData;
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// build classifier with train dataset
		System.out.println("Building model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		M5P trainedModel = (M5P) mg.buildClassifierM5P(datasetnor);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save model
		System.out.println("Saving model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.saveModel(trainedModel, MODElPATH);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Show tree
		// System.out.println("Showing tree ...");
		// startTime = System.currentTimeMillis();
		// -------------------------------------------//
		// mg.showTree(trainedModel, TREEPATH);
		// createDotGraph(TREEPATH);
		// -------------------------------------------//
		// endTime = System.currentTimeMillis();
		// totalTime = endTime - startTime;
		// System.out.println("done " + totalTime / 1000 + " s");
	}

	public void trainSVM2(SMOreg trainModel, String trainDataPath) throws Exception {
		BaseClass svm = new BaseClass(trainModel);
		svm.train(trainDataPath, MODElPATH);
	}
	public void trainSVM(String trainDataPath) throws Exception {
		/* Measure time */
		long startTime;
		long endTime;
		long totalTime;

		ModelGenerator mg = new ModelGenerator();
		Instances trainData = mg.loadDataset(trainDataPath);

		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedData = prep.removeFeatures(trainData, "1-2");
		preprocessedData.randomize(new Debug.Random(1));
		// Instances preprocessedData = trainData;
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		Filter filter = new Normalize();

		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		filter.setInputFormat(preprocessedData);
		Instances datasetnor = Filter.useFilter(preprocessedData, filter);
		// Instances datasetnor = preprocessedData;
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// build classifier with train dataset
		System.out.println("Building model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		SMOreg trainedModel = (SMOreg) mg.buildClassifierSVM(datasetnor);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save model
		System.out.println("Saving model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.saveModel(trainedModel, MODElPATH);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Show tree
		// System.out.println("Showing tree ...");
		// startTime = System.currentTimeMillis();
		// -------------------------------------------//
		// mg.showTree(trainedModel, TREEPATH);
		// createDotGraph(TREEPATH);
		// -------------------------------------------//
		// endTime = System.currentTimeMillis();
		// totalTime = endTime - startTime;
		// System.out.println("done " + totalTime / 1000 + " s");
	}

	public void testREPTreeLeaveOneFeatureOut(String modelPath, String trainDataPath, String testDataPath,
			int featureIndex) throws Exception {
		/* Measure time */
		long startTime;
		long endTime;
		long totalTime;

		ModelGenerator mg = new ModelGenerator();
		Instances trainData = mg.loadDataset(trainDataPath);
		// Instances testData = mg.loadDataset(testDataPath);
		Instances testData = mg.loadDatasetWithId(testDataPath);

		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedTrainData = prep.removeFeatures(trainData, "1,2," + Integer.toString(featureIndex));
		Instances preprocessedTestData = prep.removeFeatures(testData, "1,2," + Integer.toString(featureIndex));

		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		Filter filterTrain = new Normalize();
		Filter filterTest = new Normalize();

		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		filterTrain.setInputFormat(preprocessedTrainData);
		Instances datasetnorTrain = Filter.useFilter(preprocessedTrainData, filterTrain);
		filterTest.setInputFormat(preprocessedTestData);
		Instances datasetnorTest = Filter.useFilter(preprocessedTestData, filterTest);

		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Load model
		System.out.println("Loading model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		REPTree loadedModel = (REPTree) mg.loadModelREPTree(modelPath + "/model.bin");
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Evaluate classifier with test dataset
		System.out.println("Evaluating ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Evaluation eval = mg.evaluateModel(loadedModel, datasetnorTrain, datasetnorTest);
		// System.out.println("Evaluation: " + eval.toSummaryString("", true));
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save predicted results
		System.out.println("Saving results ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.savePredictedWithId(testData, eval, modelPath + "/result_Id_score.csv");
		mg.convertScoreToLabelWithId(modelPath + "/result_Id_score.csv", modelPath + "/result_Id_label.csv", cut_off);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save evaluation
		System.out.println("Saving evaluation ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		// Sort result
		HashMap<String, List<Instance>> sortedResultByPredictedScore = mg
				.sortResultByAttribute(modelPath + "/result_Id_score.csv", 2);
		mg.saveSortedResult(modelPath + "/result_Id_score_sorted.csv", sortedResultByPredictedScore);
		HashMap<String, List<Instance>> sortedResultByActualScore = mg
				.sortResultByAttribute(modelPath + "/result_Id_score.csv", 3);
		HashMap<Integer, List<String>> listResultTopK = new HashMap<>();
		for (int topK : this.topK) {
			List<String> listResult = new ArrayList<>();
			listResultTopK.put(topK, listResult);
			mg.saveEvaluationTopK(modelPath + "/evalTopK_" + Integer.toString(topK) + ".csv",
					sortedResultByPredictedScore, topK, cut_off, listResultTopK, 2, 3);
			mg.saveNDCGTopK(modelPath + "/NDCGTopK" + Integer.toString(topK) + ".csv", sortedResultByPredictedScore,
					sortedResultByActualScore, topK, cut_off, 3, listResultTopK);
			mg.saveEvaluation(eval, sortedResultByPredictedScore,
					modelPath + "/eval_" + Integer.toString(topK) + ".txt", modelPath + "/result_Id_label.csv", topK,
					listResultTopK, 2, 3);
		}
		mg.saveEvaluationSumary(modelPath + "/evalSumary.csv", listResultTopK);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");
	}

	public void testM5PLeaveOneFeatureOut(String modelPath, String trainDataPath, String testDataPath, int featureIndex)
			throws Exception {
		/* Measure time */
		long startTime;
		long endTime;
		long totalTime;

		ModelGenerator mg = new ModelGenerator();
		Instances trainData = mg.loadDataset(trainDataPath);
		// Instances testData = mg.loadDataset(testDataPath);
		Instances testData = mg.loadDatasetWithId(testDataPath);

		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedTrainData = prep.removeFeatures(trainData, "1,2," + Integer.toString(featureIndex));
		Instances preprocessedTestData = prep.removeFeatures(testData, "1,2," + Integer.toString(featureIndex));

		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		Filter filterTrain = new Normalize();
		Filter filterTest = new Normalize();

		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		filterTrain.setInputFormat(preprocessedTrainData);
		Instances datasetnorTrain = Filter.useFilter(preprocessedTrainData, filterTrain);
		filterTest.setInputFormat(preprocessedTestData);
		Instances datasetnorTest = Filter.useFilter(preprocessedTestData, filterTest);

		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Load model
		System.out.println("Loading model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		M5P loadedModel = (M5P) mg.loadModelM5P(modelPath + "/model.bin");
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Evaluate classifier with test dataset
		System.out.println("Evaluating ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Evaluation eval = mg.evaluateModel(loadedModel, datasetnorTrain, datasetnorTest);
		// System.out.println("Evaluation: " + eval.toSummaryString("", true));
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save predicted results
		System.out.println("Saving results ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.savePredictedWithId(testData, eval, modelPath + "/result_Id_score.csv");
		mg.convertScoreToLabelWithId(modelPath + "/result_Id_score.csv", modelPath + "/result_Id_label.csv", cut_off);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save evaluation
		System.out.println("Saving evaluation ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		// Sort result
		HashMap<String, List<Instance>> sortedResultByPredictedScore = mg
				.sortResultByAttribute(modelPath + "/result_Id_score.csv", 2);
		mg.saveSortedResult(modelPath + "/result_Id_score_sorted.csv", sortedResultByPredictedScore);
		HashMap<String, List<Instance>> sortedResultByActualScore = mg
				.sortResultByAttribute(modelPath + "/result_Id_score.csv", 3);
		HashMap<Integer, List<String>> listResultTopK = new HashMap<>();
		for (int topK : this.topK) {
			List<String> listResult = new ArrayList<>();
			listResultTopK.put(topK, listResult);
			mg.saveEvaluationTopK(modelPath + "/evalTopK_" + Integer.toString(topK) + ".csv",
					sortedResultByPredictedScore, topK, cut_off, listResultTopK, 2, 3);
			mg.saveNDCGTopK(modelPath + "/NDCGTopK" + Integer.toString(topK) + ".csv", sortedResultByPredictedScore,
					sortedResultByActualScore, topK, cut_off, 3, listResultTopK);
			mg.saveEvaluation(eval, sortedResultByPredictedScore,
					modelPath + "/eval_" + Integer.toString(topK) + ".txt", modelPath + "/result_Id_label.csv", topK,
					listResultTopK, 2, 3);
		}
		mg.saveEvaluationSumary(modelPath + "/evalSumary.csv", listResultTopK);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");
	}

	public void testSVMLeaveOneFeatureOut(String modelPath, String trainDataPath, String testDataPath,
			String featureIndex) throws Exception {
		/* Measure time */
		long startTime;
		long endTime;
		long totalTime;

		ModelGenerator mg = new ModelGenerator();
		Instances trainData = mg.loadDataset(trainDataPath);
		// Instances testData = mg.loadDataset(testDataPath);
		Instances testData = mg.loadDatasetWithId(testDataPath);

		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedTrainData = prep.removeFeatures(trainData, "1,2," + featureIndex);
		Instances preprocessedTestData = prep.removeFeatures(testData, "1,2," + featureIndex);
		// preprocessedTrainData = prep.Numeric2Nominal(preprocessedTrainData,
		// "last");
		// preprocessedTestData = prep.Numeric2Nominal(preprocessedTestData,
		// "last");

		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		Filter filterTrain = new Normalize();
		Filter filterTest = new Normalize();

		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		filterTrain.setInputFormat(preprocessedTrainData);
		Instances datasetnorTrain = Filter.useFilter(preprocessedTrainData, filterTrain);
		filterTest.setInputFormat(preprocessedTestData);
		Instances datasetnorTest = Filter.useFilter(preprocessedTestData, filterTest);

		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Load model
		System.out.println("Loading model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		SMOreg loadedModel = (SMOreg) mg.loadModelSVM(modelPath);
		// SMO loadedModel = (SMO) mg.loadModelSVM(modelPath);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Evaluate classifier with test dataset
		System.out.println("Evaluating ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Evaluation eval = mg.evaluateModel(loadedModel, datasetnorTrain, datasetnorTest);
		// System.out.println("Evaluation: " + eval.toSummaryString("", true));
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save predicted results
		System.out.println("Saving results ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.savePredictedWithId(testData, eval, RESULTPATH + "/result_Id_score.csv");
		mg.convertScoreToLabelWithId(RESULTPATH + "/result_Id_score.csv", RESULTPATH + "/result_Id_label.csv", cut_off);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save evaluation
		System.out.println("Saving evaluation ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		// Sort result
		HashMap<String, List<Instance>> sortedResultByPredictedScore = mg
				.sortResultByAttribute(EVALPATH + "/result_Id_score.csv", 2);
		mg.saveSortedResult(EVALPATH + "/result_Id_score_sorted.csv", sortedResultByPredictedScore);
		// Filter wrong results
		HashMap<String, List<Instance>> wrongResults = mg.filterWrongResults(sortedResultByPredictedScore, 10);
		mg.saveSortedResult(EVALPATH + "/wrong_Results.csv", wrongResults);
		HashMap<String, List<Instance>> sortedResultByActualScore = mg
				.sortResultByAttribute(EVALPATH + "/result_Id_score.csv", 3);

		HashMap<Integer, List<String>> listResultTopK = new HashMap<>();
		for (int topK : this.topK) {
			List<String> listResult = new ArrayList<>();
			listResultTopK.put(topK, listResult);
			mg.saveEvaluationTopK(EVALPATH + "/evalTopK_" + Integer.toString(topK) + ".csv",
					sortedResultByPredictedScore, topK, cut_off, listResultTopK, 2, 3);
			mg.saveNDCGTopK(EVALPATH + "/NDCGTopK" + Integer.toString(topK) + ".csv", sortedResultByPredictedScore,
					sortedResultByActualScore, topK, cut_off, 3, listResultTopK);
			mg.saveEvaluation(eval, sortedResultByPredictedScore, EVALPATH + "/eval_" + Integer.toString(topK) + ".txt",
					RESULTPATH + "/result_Id_label.csv", topK, listResultTopK, 2, 3);
		}
		mg.saveEvaluationSumary(EVALPATH + "/evalSumary.csv", listResultTopK);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");
	}
	
	public void testSVCLeaveOneFeatureOut(String modelPath, String trainDataPath, String testDataPath,
			String featureIndex) throws Exception {
		/* Measure time */
		long startTime;
		long endTime;
		long totalTime;

		ModelGenerator mg = new ModelGenerator();
		Instances trainData = mg.loadDataset(trainDataPath);
		// Instances testData = mg.loadDataset(testDataPath);
		Instances testData = mg.loadDatasetWithId(testDataPath);

		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedTrainData = prep.removeFeatures(trainData, "1,2," + featureIndex);
		Instances preprocessedTestData = prep.removeFeatures(testData, "1,2," + featureIndex);
		preprocessedTrainData = prep.Numeric2Nominal(preprocessedTrainData,"last");
		preprocessedTestData = prep.Numeric2Nominal(preprocessedTestData,"last");

		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		Filter filterTrain = new Normalize();
		Filter filterTest = new Normalize();

		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		filterTrain.setInputFormat(preprocessedTrainData);
//		Instances datasetnorTrain = Filter.useFilter(preprocessedTrainData, filterTrain);
		Instances datasetnorTrain = preprocessedTrainData;
		filterTest.setInputFormat(preprocessedTestData);
//		Instances datasetnorTest = Filter.useFilter(preprocessedTestData, filterTest);
		Instances datasetnorTest = preprocessedTestData;
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Load model
		System.out.println("Loading model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		SMO loadedModel = (SMO) mg.loadModelSVC(modelPath);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Evaluate classifier with test dataset
		System.out.println("Evaluating ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Evaluation eval = mg.evaluateModel(loadedModel, datasetnorTrain, datasetnorTest);
		// System.out.println("Evaluation: " + eval.toSummaryString("", true));
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save predicted results
		System.out.println("Saving results ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.savePredictedWithId(testData, eval, RESULTPATH + "/result_Id_score.csv");
		mg.convertScoreToLabelWithId(RESULTPATH + "/result_Id_score.csv", RESULTPATH + "/result_Id_label.csv", cut_off);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		
		// Save evaluation
		System.out.println("Saving evaluation ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.saveEvaluationSVC(eval, EVALPATH + "/eval.txt", RESULTPATH + "/result_Id_label.csv");
		mg.saveEvaluationSVC2(EVALPATH + "/eval_brief.txt", RESULTPATH + "/result_Id_label.csv");
		/*// Sort result
		HashMap<String, List<Instance>> sortedResultByPredictedScore = mg
				.sortResultByAttribute(EVALPATH + "/result_Id_score.csv", 2);
		mg.saveSortedResult(EVALPATH + "/result_Id_score_sorted.csv", sortedResultByPredictedScore);
		// Filter wrong results
		HashMap<String, List<Instance>> wrongResults = mg.filterWrongResults(sortedResultByPredictedScore, 10);
		mg.saveSortedResult(EVALPATH + "/wrong_Results.csv", wrongResults);
		HashMap<String, List<Instance>> sortedResultByActualScore = mg
				.sortResultByAttribute(EVALPATH + "/result_Id_score.csv", 3);

		HashMap<Integer, List<String>> listResultTopK = new HashMap<>();
		for (int topK : this.topK) {
			List<String> listResult = new ArrayList<>();
			listResultTopK.put(topK, listResult);
			mg.saveEvaluationTopK(EVALPATH + "/evalTopK_" + Integer.toString(topK) + ".csv",
					sortedResultByPredictedScore, topK, cut_off, listResultTopK);
			mg.saveNDCGTopK(EVALPATH + "/NDCGTopK" + Integer.toString(topK) + ".csv", sortedResultByPredictedScore,
					sortedResultByActualScore, topK, cut_off, listResultTopK);
			mg.saveEvaluation(eval, sortedResultByPredictedScore, EVALPATH + "/eval_" + Integer.toString(topK) + ".txt",
					RESULTPATH + "/result_Id_label.csv", topK, listResultTopK);
		}
		mg.saveEvaluationSumary(EVALPATH + "/evalSumary.csv", listResultTopK);*/		
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");
	}
	
	public void testSvcAndSvrLeaveOneFeatureOut(String modelSvcPath, String resultSvcPath, String evalSvcPath, String modelSvrPath, String resultSvrPath, String evalSvrPath, String trainDataPath, String testDataPath,
			String featureIndex) throws Exception {
		/* Measure time */
		long startTime;
		long endTime;
		long totalTime;

		ModelGenerator mg = new ModelGenerator();
		Instances trainDataSvc = mg.loadDataset(trainDataPath);
		// Instances testData = mg.loadDataset(testDataPath);
		Instances testDataSvc = mg.loadDatasetWithId(testDataPath);

		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedTrainDataSvc = prep.removeFeatures(trainDataSvc, "1,2," + featureIndex);
		Instances preprocessedTestDataSvc = prep.removeFeatures(testDataSvc, "1,2," + featureIndex);
		preprocessedTrainDataSvc = prep.Numeric2Nominal(preprocessedTrainDataSvc,"last");
		preprocessedTestDataSvc = prep.Numeric2Nominal(preprocessedTestDataSvc,"last");
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		Filter filterTrainSvc = new Normalize();
		Filter filterTestSvc = new Normalize();

		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		filterTrainSvc.setInputFormat(preprocessedTrainDataSvc);
		Instances datasetnorTrainSvc = Filter.useFilter(preprocessedTrainDataSvc, filterTrainSvc);
		filterTestSvc.setInputFormat(preprocessedTestDataSvc);
		Instances datasetnorTestSvc = Filter.useFilter(preprocessedTestDataSvc, filterTestSvc);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Load model
		System.out.println("Loading model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		SMO loadedModelSvc = (SMO) mg.loadModelSVC(modelSvcPath);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Evaluate classifier with test dataset
		System.out.println("Evaluating ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Evaluation evalSvc = mg.evaluateModel(loadedModelSvc, datasetnorTrainSvc, datasetnorTestSvc);
		// System.out.println("Evaluation: " + eval.toSummaryString("", true));
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save predicted results
		System.out.println("Saving results ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.savePredictedWithId(testDataSvc, evalSvc, resultSvcPath + "/result_Id_score.csv");
		mg.convertScoreToLabelWithId(resultSvcPath + "/result_Id_score.csv", resultSvcPath + "/result_Id_label.csv", cut_off);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");
		
		// Save evaluation
		System.out.println("Saving evaluation ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.saveEvaluationSVC(evalSvc, evalSvcPath + "/eval.txt", resultSvcPath + "/result_Id_label.csv");
		mg.saveEvaluationSVC2(evalSvcPath + "/eval_brief.txt", resultSvcPath + "/result_Id_label.csv");				
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");
		
		//=========================================================//
		Instances trainDataSvr = mg.loadDataset(trainDataPath);
		// Instances testData = mg.loadDataset(testDataPath);
		Instances testDataSvr = mg.loadDatasetWithId(testDataPath);

		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Preporcess prepSvr = new Preporcess();
		Instances preprocessedTrainDataSvr = prep.removeFeatures(trainDataSvr, "1,2," + featureIndex);
		Instances preprocessedTestDataSvr = prep.removeFeatures(testDataSvr, "1,2," + featureIndex);
		// preprocessedTrainData = prep.Numeric2Nominal(preprocessedTrainData,
		// "last");
		// preprocessedTestData = prep.Numeric2Nominal(preprocessedTestData,
		// "last");
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		Filter filterTrainSvr = new Normalize();
		Filter filterTestSvr = new Normalize();

		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		filterTrainSvr.setInputFormat(preprocessedTrainDataSvr);
		Instances datasetnorTrainSvr = Filter.useFilter(preprocessedTrainDataSvr, filterTrainSvr);
		filterTestSvr.setInputFormat(preprocessedTestDataSvr);
		Instances datasetnorTestSvr = Filter.useFilter(preprocessedTestDataSvr, filterTestSvr);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Load model
		System.out.println("Loading model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		SMOreg loadedModelSvr = (SMOreg) mg.loadModelSVM(modelSvrPath);
		// SMO loadedModel = (SMO) mg.loadModelSVM(modelPath);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Evaluate classifier with test dataset
		System.out.println("Evaluating ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Evaluation evalSvr = mg.evaluateModel(loadedModelSvr, datasetnorTrainSvr, datasetnorTestSvr);
		// System.out.println("Evaluation: " + eval.toSummaryString("", true));
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save predicted results
		System.out.println("Saving results ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.savePredictedWithId(testDataSvr, evalSvr, resultSvrPath + "/result_Id_score.csv");
		mg.convertScoreToLabelWithId(resultSvrPath + "/result_Id_score.csv", resultSvrPath + "/result_Id_label.csv", cut_off);
		mg.combineAndSaveResult(resultSvrPath + "/result_Id_score.csv", resultSvcPath + "/result_Id_label.csv", 2, 2, 3,resultSvrPath + "/result_Id_score_label.csv");
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save evaluation
		System.out.println("Saving evaluation ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		// Sort result
		HashMap<String, List<Instance>> sortedResultByPredictedScore = mg
				.sortResultByAttributes(resultSvrPath + "/result_Id_score_label.csv", 2, 3);
		mg.saveSortedResultSvcSvr(evalSvrPath + "/result_Id_score_sorted.csv", sortedResultByPredictedScore);		
		// Filter wrong results
		HashMap<String, List<Instance>> wrongResults = mg.filterWrongResults(sortedResultByPredictedScore, 10);
		mg.saveSortedResult(evalSvrPath + "/wrong_Results.csv", wrongResults);
		HashMap<String, List<Instance>> sortedResultByActualScore = mg
				.sortResultByAttribute(resultSvrPath + "/result_Id_score_label.csv", 4);

		HashMap<Integer, List<String>> listResultTopK = new HashMap<>();
		for (int topK : this.topK) {
			List<String> listResult = new ArrayList<>();
			listResultTopK.put(topK, listResult);
			mg.saveEvaluationTopK(evalSvrPath + "/evalTopK_" + Integer.toString(topK) + ".csv",
					sortedResultByPredictedScore, topK, cut_off, listResultTopK, 3, 4);
			mg.saveNDCGTopK(evalSvrPath + "/NDCGTopK" + Integer.toString(topK) + ".csv", sortedResultByPredictedScore,
					sortedResultByActualScore, topK, cut_off, 4, listResultTopK);
			mg.saveEvaluation(evalSvr, sortedResultByPredictedScore, evalSvrPath + "/eval_" + Integer.toString(topK) + ".txt",
					resultSvrPath + "/result_Id_label.csv", topK, listResultTopK, 2, 4);
		}
		mg.saveEvaluationSumary(evalSvrPath + "/evalSumary.csv", listResultTopK);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");
	}
	
	public void testSvcAndTfidfLeaveOneFeatureOut(String modelSvcPath, String resultSvcPath, String evalSvcPath, String trainDataPath, String testDataPath,	String featureIndex) throws Exception {
		/* Measure time */
		long startTime;
		long endTime;
		long totalTime;

		ModelGenerator mg = new ModelGenerator();
		Instances trainDataSvc = mg.loadDataset(trainDataPath);
		// Instances testData = mg.loadDataset(testDataPath);
		Instances testDataSvc = mg.loadDatasetWithId(testDataPath);

		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedTrainDataSvc = prep.removeFeatures(trainDataSvc, "1,2," + featureIndex);
		Instances preprocessedTestDataSvc = prep.removeFeatures(testDataSvc, "1,2," + featureIndex);
		preprocessedTrainDataSvc = prep.Numeric2Nominal(preprocessedTrainDataSvc,"last");
		preprocessedTestDataSvc = prep.Numeric2Nominal(preprocessedTestDataSvc,"last");
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		Filter filterTrainSvc = new Normalize();
		Filter filterTestSvc = new Normalize();

		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		filterTrainSvc.setInputFormat(preprocessedTrainDataSvc);
//		Instances datasetnorTrainSvc = Filter.useFilter(preprocessedTrainDataSvc, filterTrainSvc);
		Instances datasetnorTrainSvc = preprocessedTrainDataSvc;
		filterTestSvc.setInputFormat(preprocessedTestDataSvc);
//		Instances datasetnorTestSvc = Filter.useFilter(preprocessedTestDataSvc, filterTestSvc);
		Instances datasetnorTestSvc = preprocessedTestDataSvc;
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Load model
		System.out.println("Loading model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		SMO loadedModelSvc = (SMO) mg.loadModelSVC(modelSvcPath);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Evaluate classifier with test dataset
		System.out.println("Evaluating ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Evaluation evalSvc = mg.evaluateModel(loadedModelSvc, datasetnorTrainSvc, datasetnorTestSvc);
		// System.out.println("Evaluation: " + eval.toSummaryString("", true));
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save predicted results
		System.out.println("Saving results ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.savePredictedWithId(testDataSvc, evalSvc, resultSvcPath + "/result_Id_score.csv");
		mg.convertScoreToLabelWithId(resultSvcPath + "/result_Id_score.csv", resultSvcPath + "/result_Id_label.csv", cut_off);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");
		
		// Save evaluation
		System.out.println("Saving evaluation ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.saveEvaluationSVC(evalSvc, evalSvcPath + "/eval.txt", resultSvcPath + "/result_Id_label.csv");
		mg.saveEvaluationSVC2(evalSvcPath + "/eval_brief.txt", resultSvcPath + "/result_Id_label.csv");				
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");
		
		//=========================================================//
		
		// Save predicted results
		System.out.println("Saving results ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//		
		mg.combineAndSaveResult(testDataPath, resultSvcPath + "/result_Id_label.csv", 16, 2, 3, resultSvcPath + "/result_Id_score_label.csv");
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save evaluation
		System.out.println("Saving evaluation ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		// Sort result
		HashMap<String, List<Instance>> sortedResultByPredictedScore = mg
				.sortResultByAttributes(resultSvcPath + "/result_Id_score_label.csv", 2, 3);
		mg.saveSortedResultSvcSvr(evalSvcPath + "/result_Id_score_sorted.csv", sortedResultByPredictedScore);		
		// Filter wrong results
		HashMap<String, List<Instance>> wrongResults = mg.filterWrongResults(sortedResultByPredictedScore, 10);
		mg.saveSortedResult(resultSvcPath + "/wrong_Results.csv", wrongResults);
		HashMap<String, List<Instance>> sortedResultByActualScore = mg
				.sortResultByAttribute(resultSvcPath + "/result_Id_score_label.csv", 4);

		HashMap<Integer, List<String>> listResultTopK = new HashMap<>();
		for (int topK : this.topK) {
			List<String> listResult = new ArrayList<>();
			listResultTopK.put(topK, listResult);
			mg.saveEvaluationTopK(evalSvcPath + "/evalTopK_" + Integer.toString(topK) + ".csv",
					sortedResultByPredictedScore, topK, cut_off, listResultTopK, 3, 4);
			mg.saveNDCGTopK(evalSvcPath + "/NDCGTopK" + Integer.toString(topK) + ".csv", sortedResultByPredictedScore,
					sortedResultByActualScore, topK, cut_off, 4, listResultTopK);			
			mg.saveEvaluationNoTime(sortedResultByPredictedScore, evalSvcPath + "/eval_" + Integer.toString(topK) + ".txt",
					resultSvcPath + "/result_Id_label.csv", topK, listResultTopK, 2, 4);
		}
		mg.saveEvaluationSumary(evalSvcPath + "/evalSumary.csv", listResultTopK);		
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");
	}

	public void testREPTree(String modelPath, String trainDataPath, String testDataPath) throws Exception {
		/* Measure time */
		long startTime;
		long endTime;
		long totalTime;

		ModelGenerator mg = new ModelGenerator();
		Instances trainData = mg.loadDataset(trainDataPath);
		// Instances testData = mg.loadDataset(testDataPath);
		Instances testData = mg.loadDatasetWithId(testDataPath);

		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedTrainData = prep.removeFeatures(trainData, "1-2");
		Instances preprocessedTestData = prep.removeFeatures(testData, "1-2");

		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		Filter filterTrain = new Normalize();
		Filter filterTest = new Normalize();

		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		filterTrain.setInputFormat(preprocessedTrainData);
		Instances datasetnorTrain = Filter.useFilter(preprocessedTrainData, filterTrain);
		filterTest.setInputFormat(preprocessedTestData);
		Instances datasetnorTest = Filter.useFilter(preprocessedTestData, filterTest);

		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Load model
		System.out.println("Loading model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		REPTree loadedModel = (REPTree) mg.loadModelREPTree(modelPath);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Evaluate classifier with test dataset
		System.out.println("Evaluating ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Evaluation eval = mg.evaluateModel(loadedModel, datasetnorTrain, datasetnorTest);
		// System.out.println("Evaluation: " + eval.toSummaryString("", true));
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save predicted results
		System.out.println("Saving results ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.savePredictedWithId(testData, eval, RESULTPATH + "/result_Id_score.csv");
		mg.convertScoreToLabelWithId(RESULTPATH + "/result_Id_score.csv", RESULTPATH + "/result_Id_label.csv", cut_off);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save evaluation
		System.out.println("Saving evaluation ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//		
		// Sort result
		HashMap<String, List<Instance>> sortedResultByPredictedScore = mg
				.sortResultByAttribute(EVALPATH + "/result_Id_score.csv", 2);
		mg.saveSortedResult(EVALPATH + "/result_Id_score_sorted.csv", sortedResultByPredictedScore);
		// Filter wrong results
		HashMap<String, List<Instance>> wrongResults = mg.filterWrongResults(sortedResultByPredictedScore, 10);
		mg.saveSortedResult(EVALPATH + "/wrong_Results.csv", wrongResults);
		HashMap<String, List<Instance>> sortedResultByActualScore = mg
				.sortResultByAttribute(EVALPATH + "/result_Id_score.csv", 3);
		HashMap<Integer, List<String>> listResultTopK = new HashMap<>();
		for (int topK : this.topK) {
			List<String> listResult = new ArrayList<>();
			listResultTopK.put(topK, listResult);
			mg.saveEvaluationTopK(EVALPATH + "/evalTopK_" + Integer.toString(topK) + ".csv",
					sortedResultByPredictedScore, topK, cut_off, listResultTopK, 2, 3);
			mg.saveNDCGTopK(EVALPATH + "/NDCGTopK" + Integer.toString(topK) + ".csv", sortedResultByPredictedScore,
					sortedResultByActualScore, topK, cut_off, 3, listResultTopK);
			mg.saveEvaluation(eval, sortedResultByPredictedScore, EVALPATH + "/eval_" + Integer.toString(topK) + ".txt",
					RESULTPATH + "/result_Id_label.csv", topK, listResultTopK, 2, 3);
		}
		mg.saveEvaluationSumary(EVALPATH + "/evalSumary.csv", listResultTopK);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");
	}

	public void testM5P(String modelPath, String trainDataPath, String testDataPath) throws Exception {
		/* Measure time */
		long startTime;
		long endTime;
		long totalTime;

		ModelGenerator mg = new ModelGenerator();
		Instances trainData = mg.loadDataset(trainDataPath);
		// Instances testData = mg.loadDataset(testDataPath);
		Instances testData = mg.loadDatasetWithId(testDataPath);

		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedTrainData = prep.removeFeatures(trainData, "1-2");
		Instances preprocessedTestData = prep.removeFeatures(testData, "1-2");

		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		Filter filterTrain = new Normalize();
		Filter filterTest = new Normalize();

		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		filterTrain.setInputFormat(preprocessedTrainData);
		Instances datasetnorTrain = Filter.useFilter(preprocessedTrainData, filterTrain);
		filterTest.setInputFormat(preprocessedTestData);
		Instances datasetnorTest = Filter.useFilter(preprocessedTestData, filterTest);

		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Load model
		System.out.println("Loading model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		M5P loadedModel = (M5P) mg.loadModelM5P(modelPath);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Evaluate classifier with test dataset
		System.out.println("Evaluating ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Evaluation eval = mg.evaluateModel(loadedModel, datasetnorTrain, datasetnorTest);
		// System.out.println("Evaluation: " + eval.toSummaryString("", true));
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save predicted results
		System.out.println("Saving results ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.savePredictedWithId(testData, eval, RESULTPATH + "/result_Id_score.csv");
		mg.convertScoreToLabelWithId(RESULTPATH + "/result_Id_score.csv", RESULTPATH + "/result_Id_label.csv", cut_off);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save evaluation
		System.out.println("Saving evaluation ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		// Sort result
		HashMap<String, List<Instance>> sortedResultByPredictedScore = mg
				.sortResultByAttribute(EVALPATH + "/result_Id_score.csv", 2);
		mg.saveSortedResult(EVALPATH + "/result_Id_score_sorted.csv", sortedResultByPredictedScore);
		// Filter wrong results
		HashMap<String, List<Instance>> wrongResults = mg.filterWrongResults(sortedResultByPredictedScore, 10);
		mg.saveSortedResult(EVALPATH + "/wrong_Results.csv", wrongResults);
		HashMap<String, List<Instance>> sortedResultByActualScore = mg
				.sortResultByAttribute(EVALPATH + "/result_Id_score.csv", 3);

		HashMap<Integer, List<String>> listResultTopK = new HashMap<>();
		for (int topK : this.topK) {
			List<String> listResult = new ArrayList<>();
			listResultTopK.put(topK, listResult);
			mg.saveEvaluationTopK(EVALPATH + "/evalTopK_" + Integer.toString(topK) + ".csv",
					sortedResultByPredictedScore, topK, cut_off, listResultTopK, 2, 3);
			mg.saveNDCGTopK(EVALPATH + "/NDCGTopK" + Integer.toString(topK) + ".csv", sortedResultByPredictedScore,
					sortedResultByActualScore, topK, cut_off, 3, listResultTopK);
			mg.saveEvaluation(eval, sortedResultByPredictedScore, EVALPATH + "/eval_" + Integer.toString(topK) + ".txt",
					RESULTPATH + "/result_Id_label.csv", topK, listResultTopK, 2, 3);
		}
		mg.saveEvaluationSumary(EVALPATH + "/evalSumary.csv", listResultTopK);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");
	}

	public void testSVM(String modelPath, String trainDataPath, String testDataPath) throws Exception {
		/* Measure time */
		long startTime;
		long endTime;
		long totalTime;

		ModelGenerator mg = new ModelGenerator();
		Instances trainData = mg.loadDataset(trainDataPath);
		// Instances testData = mg.loadDataset(testDataPath);
		Instances testData = mg.loadDatasetWithId(testDataPath);

		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedTrainData = prep.removeFeatures(trainData, "1-2");
		Instances preprocessedTestData = prep.removeFeatures(testData, "1-2");

		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		Filter filterTrain = new Normalize();
		Filter filterTest = new Normalize();

		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		filterTrain.setInputFormat(preprocessedTrainData);
		Instances datasetnorTrain = Filter.useFilter(preprocessedTrainData, filterTrain);
		filterTest.setInputFormat(preprocessedTestData);
		Instances datasetnorTest = Filter.useFilter(preprocessedTestData, filterTest);

		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Load model
		System.out.println("Loading model ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		SMOreg loadedModel = (SMOreg) mg.loadModelSVM(modelPath);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Evaluate classifier with test dataset
		System.out.println("Evaluating ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		Evaluation eval = mg.evaluateModel(loadedModel, datasetnorTrain, datasetnorTest);
		// System.out.println("Evaluation: " + eval.toSummaryString("", true));
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save predicted results
		System.out.println("Saving results ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		mg.savePredictedWithId(testData, eval, RESULTPATH + "/result_Id_score.csv");
		mg.convertScoreToLabelWithId(RESULTPATH + "/result_Id_score.csv", RESULTPATH + "/result_Id_label.csv", cut_off);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");

		// Save evaluation
		System.out.println("Saving evaluation ...");
		startTime = System.currentTimeMillis();
		// -------------------------------------------//
		// Sort result
		HashMap<String, List<Instance>> sortedResultByPredictedScore = mg
				.sortResultByAttribute(EVALPATH + "/result_Id_score.csv", 2);
		mg.saveSortedResult(EVALPATH + "/result_Id_score_sorted.csv", sortedResultByPredictedScore);
		// Filter wrong results
		HashMap<String, List<Instance>> wrongResults = mg.filterWrongResults(sortedResultByPredictedScore, 10);
		mg.saveSortedResult(EVALPATH + "/wrong_Results.csv", wrongResults);
		HashMap<String, List<Instance>> sortedResultByActualScore = mg
				.sortResultByAttribute(EVALPATH + "/result_Id_score.csv", 3);

		HashMap<Integer, List<String>> listResultTopK = new HashMap<>();
		for (int topK : this.topK) {
			List<String> listResult = new ArrayList<>();
			listResultTopK.put(topK, listResult);
			mg.saveEvaluationTopK(EVALPATH + "/evalTopK_" + Integer.toString(topK) + ".csv",
					sortedResultByPredictedScore, topK, cut_off, listResultTopK, 2, 3);
			mg.saveNDCGTopK(EVALPATH + "/NDCGTopK" + Integer.toString(topK) + ".csv", sortedResultByPredictedScore,
					sortedResultByActualScore, topK, cut_off, 3, listResultTopK);
			mg.saveEvaluation(eval, sortedResultByPredictedScore, EVALPATH + "/eval_" + Integer.toString(topK) + ".txt",
					RESULTPATH + "/result_Id_label.csv", topK, listResultTopK, 2, 3);
		}
		mg.saveEvaluationSumary(EVALPATH + "/evalSumary.csv", listResultTopK);
		// -------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.println("done " + totalTime / 1000 + " s");
	}

	public static void runREPTree() throws Exception {
		StartWeka wk = new StartWeka();
		System.out.println("Training REPTree...");

		List<String> listTrainPath = new ArrayList<>();
		List<String> listTestPath = new ArrayList<>();

		// Train Dataset1
		listTrainPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_01_11_2018/Train/dataset1/d1_features_event_3110_NEW.csv");
		listTrainPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_01_11_2018/Train/dataset1/d1_features_topic_3110_NEW.csv");
		listTrainPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_01_11_2018/Train/dataset1/d1_features_ne_3110_NEW.csv");

		// Train Dataset2
		// listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset2/d2_features_graph_event.csv");
		// listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset2/d2_features_graph_topic.csv");
		// listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset2/d2_features_graph_ne.csv");

		// Train Dataset3
		// listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset3/d3_features_graph_event.csv");
		// listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset3/d3_features_graph_topic.csv");
		// listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset3/d3_features_graph_ne.csv");

		// Test
		listTestPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_01_11_2018/Test/features_082017_ts_event_NEW.csv");
		listTestPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_01_11_2018/Test/features_082017_ts_topic_NEW.csv");
		listTestPath
				.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_01_11_2018/Test/features_082017_ts_ne_NEW.csv");
		// Test Pos
		// listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/Positive/features_082017_pos_event.csv");
		// listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/Positive/features_082017_pos_topic.csv");
		// listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/Positive/features_082017_pos_ne.csv");

		for (int i = 0; i < listTrainPath.size(); i++) {
			String trainDataPath = listTrainPath.get(i);
			String testDataPath = listTestPath.get(i);
			switch (i) {
			case 0:
				RESULTPATH = "result/event";
				MODElPATH = "result/event/model.bin";
				EVALPATH = RESULTPATH;
				TREEPATH = RESULTPATH;
				System.out.println("----Event----");
				break;
			case 1:
				RESULTPATH = "result/topic";
				MODElPATH = "result/topic/model.bin";
				EVALPATH = RESULTPATH;
				TREEPATH = RESULTPATH;
				System.out.println("----Topic----");
				break;
			case 2:
				RESULTPATH = "result/ne";
				MODElPATH = "result/ne/model.bin";
				EVALPATH = RESULTPATH;
				TREEPATH = RESULTPATH;
				System.out.println("----NE----");
				break;
			default:
				break;
			}

			wk.trainREPTree(trainDataPath);
			System.out.println("Testing REPTree...");
			wk.testREPTree(MODElPATH, trainDataPath, testDataPath);
		}
	}

	public static void runM5P() throws Exception {
		StartWeka wk = new StartWeka();
		System.out.println("Training M5P...");

		List<String> listTrainPath = new ArrayList<>();
		List<String> listTestPath = new ArrayList<>();

		// Train Dataset1
		// listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset1/d1_features_label_event.csv");
		// listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset1/d1_features_label_topic.csv");
		// listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset1/d1_features_label_ne.csv");

		// Train Dataset2
		// listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset2/d2_features_graph_event.csv");
		// listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset2/d2_features_graph_topic.csv");
		// listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset2/d2_features_graph_ne.csv");

		// Train Dataset3
		listTrainPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset3/d3_features_graph_event.csv");
		listTrainPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset3/d3_features_graph_topic.csv");
		listTrainPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset3/d3_features_graph_ne.csv");

		// Test
		// listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/features_082017_ts_event.csv");
		// listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/features_082017_ts_topic.csv");
		// listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/features_082017_ts_ne.csv");

		// Test Pos
		listTestPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/Positive/features_082017_pos_event.csv");
		listTestPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/Positive/features_082017_pos_topic.csv");
		listTestPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/Positive/features_082017_pos_ne.csv");

		for (int i = 0; i < listTrainPath.size(); i++) {
			String trainDataPath = listTrainPath.get(i);
			String testDataPath = listTestPath.get(i);
			switch (i) {
			case 0:
				RESULTPATH = "result/event";
				MODElPATH = "result/event/model.bin";
				EVALPATH = RESULTPATH;
				TREEPATH = RESULTPATH;
				System.out.println("----Event----");
				break;
			case 1:
				RESULTPATH = "result/topic";
				MODElPATH = "result/topic/model.bin";
				EVALPATH = RESULTPATH;
				TREEPATH = RESULTPATH;
				System.out.println("----Topic----");
				break;
			case 2:
				RESULTPATH = "result/ne";
				MODElPATH = "result/ne/model.bin";
				EVALPATH = RESULTPATH;
				TREEPATH = RESULTPATH;
				System.out.println("----NE----");
				break;
			default:
				break;
			}

			wk.trainM5P(trainDataPath);
			System.out.println("Testing M5P...");
			wk.testM5P(MODElPATH, trainDataPath, testDataPath);
		}

	}

	public static void runSVM() throws Exception {
		StartWeka wk = new StartWeka();		

		List<String> listTrainPath = new ArrayList<>();
		List<String> listTestPath = new ArrayList<>();

		// Train Dataset1		
		 listTrainPath.add("/home/lana/code/PostClickRecommendation/src/main/resources/output/train_data2/d1_features_event_2.csv");
//		 listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Train/dataset1/d1_features_topic.csv");
//		 listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Train/dataset1/d1_features_ne.csv");

		// Test
		listTestPath.add("/home/lana/code/PostClickRecommendation/src/main/resources/output/test_positive/features_082017_pos_event.csv");
//		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Test_label0/testset_features_topic.csv");
//		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Test_label0/testset_features_ne.csv");

		// Test Positive
//		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Test_label0/Positive/testset_features_pos_event.csv");
//		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Test_label0/Positive/testset_features_pos_topic.csv");
//		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Test_label0/Positive/testset_features_pos_ne.csv");
		
		for (int i = 0; i < listTrainPath.size(); i++) {
			String trainDataPath = listTrainPath.get(i);
			String testDataPath = listTestPath.get(i);
			switch (i) {
			case 0:
				RESULTPATH = "result/event";
				MODElPATH = "result/event/model.bin";
				EVALPATH = RESULTPATH;
				TREEPATH = RESULTPATH;
				System.out.println("----Event----");
				break;
			case 1:
				RESULTPATH = "result/topic";
				MODElPATH = "result/topic/model.bin";
				EVALPATH = RESULTPATH;
				TREEPATH = RESULTPATH;
				System.out.println("----Topic----");
				break;
			case 2:
				RESULTPATH = "result/ne";
				MODElPATH = "result/ne/model.bin";
				EVALPATH = RESULTPATH;
				TREEPATH = RESULTPATH;
				System.out.println("----NE----");
				break;
			default:
				break;
			}
			
			System.out.println("Training SVM...");
			SMOreg trainedModel = new SMOreg();
//			wk.trainSVM2(trainedModel,trainDataPath);
			wk.trainSVM(trainDataPath);
			System.out.println("Testing SVM...");
			wk.testSVMLeaveOneFeatureOut(MODElPATH, trainDataPath, testDataPath, "");
		}

	}
	
	public static void runSVC() throws Exception {
		StartWeka wk = new StartWeka();		

		List<String> listTrainPath = new ArrayList<>();
		List<String> listTestPath = new ArrayList<>();

		// Train Dataset1
		listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_12_2018/Test/dataset1/0.05/event/Train_balance_event.csv");
		listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_12_2018/Test/dataset1/0.05/topic/Train_balance_topic.csv");
		listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_12_2018/Test/dataset1/0.05/ne/Train_balance_ne.csv");		
//		listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Train/dataset1/d1_features_event.csv");
//		listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Train/dataset1/d1_features_topic.csv");
//		listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Train/dataset1/d1_features_ne.csv");

		// Test
		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_12_2018/Test/dataset1/0.05/event/Test_balance_event.csv");
		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_12_2018/Test/dataset1/0.05/topic/Test_balance_topic.csv");
		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_12_2018/Test/dataset1/0.05/ne/Test_balance_ne.csv");
//		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Test_SVC/features_test_event_2.csv");		
//		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Test_SVC/features_test_topic_2.csv");
//		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Test_SVC/features_test_ne_2.csv");
		
		for (int i = 0; i < listTrainPath.size(); i++) {
			String trainDataPath = listTrainPath.get(i);
			String testDataPath = listTestPath.get(i);
			switch (i) {
			case 0:
				RESULTPATH = "result/event";
				MODElPATH = "result/event/model.bin";
				EVALPATH = RESULTPATH;
				TREEPATH = RESULTPATH;
				System.out.println("----Event----");
				break;
			case 1:
				RESULTPATH = "result/topic";
				MODElPATH = "result/topic/model.bin";
				EVALPATH = RESULTPATH;
				TREEPATH = RESULTPATH;
				System.out.println("----Topic----");
				break;
			case 2:
				RESULTPATH = "result/ne";
				MODElPATH = "result/ne/model.bin";
				EVALPATH = RESULTPATH;
				TREEPATH = RESULTPATH;
				System.out.println("----NE----");
				break;
			default:
				break;
			}
			System.out.println("-----------");
			System.out.println("Training SVC...");
			wk.trainSVCLeaveOneFeatureOut(trainDataPath, MODElPATH, "");
			System.out.println("-----------");
			System.out.println("Testing SVC...");
			wk.testSVCLeaveOneFeatureOut(MODElPATH, trainDataPath, testDataPath, "");
		}

	}
	
	public static void runSvcAndSvr() throws Exception {
		StartWeka wk = new StartWeka();		

		List<String> listTrainPath = new ArrayList<>();
		List<String> listTestPath = new ArrayList<>();
		String modelSvcPath = "";
		String resultSvcPath = "";
		String evalSvcPath = "";
		String modelSvrPath = "";
		String resultSvrPath = "";
		String evalSvrPath = "";

		// Train Dataset1		
		listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Train/dataset1/d1_features_event.csv");
		listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Train/dataset1/d1_features_topic.csv");
		listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Train/dataset1/d1_features_ne.csv");

		// Test Positive
		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Test_label0/Positive/testset_features_pos_event.csv");		
		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Test_label0/Positive/testset_features_pos_topic.csv");
		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Test_label0/Positive/testset_features_pos_ne.csv");
		
		for (int i = 0; i < listTrainPath.size(); i++) {
			String trainDataPath = listTrainPath.get(i);
			String testDataPath = listTestPath.get(i);
			
			switch (i) {
			case 0:
				resultSvcPath = "result/SVC/event";
				resultSvrPath = "result/SVR/event";
				modelSvcPath = "result/SVC/event/model.bin";
				modelSvrPath = "result/SVR/event/model.bin";
				evalSvcPath = resultSvcPath;
				evalSvrPath = resultSvrPath;				
				System.out.println("----Event----");
				break;
			case 1:
				resultSvcPath = "result/SVC/topic";
				resultSvrPath = "result/SVR/topic";
				modelSvcPath = "result/SVC/topic/model.bin";
				modelSvrPath = "result/SVR/topic/model.bin";
				evalSvcPath = resultSvcPath;
				evalSvrPath = resultSvrPath;
				System.out.println("----Topic----");
				break;
			case 2:
				resultSvcPath = "result/SVC/ne";
				resultSvrPath = "result/SVR/ne";
				modelSvcPath = "result/SVC/ne/model.bin";
				modelSvrPath = "result/SVR/ne/model.bin";
				evalSvcPath = resultSvcPath;
				evalSvrPath = resultSvrPath;
				System.out.println("----NE----");
				break;
			default:
				break;
			}
			
			System.out.println("-----------");
			System.out.println("Training SVC...");
			wk.trainSVCLeaveOneFeatureOut(trainDataPath, modelSvcPath, "");
			System.out.println("-----------");
			System.out.println("Training SVR...");
			wk.trainSVMLeaveOneFeatureOut(trainDataPath, modelSvrPath, "");
			
			System.out.println("-----------");
			System.out.println("Testing SVC and SVR...");			
			wk.testSvcAndSvrLeaveOneFeatureOut(modelSvcPath, resultSvcPath, evalSvcPath, modelSvrPath, resultSvrPath, evalSvrPath, trainDataPath, testDataPath, "");
		}

	}
	
	public static void runSvcAndTfidf() throws Exception {
		StartWeka wk = new StartWeka();		

		List<String> listTrainPath = new ArrayList<>();
		List<String> listTestPath = new ArrayList<>();
		String modelSvcPath = "";
		String resultSvcPath = "";
		String evalSvcPath = "";		

		// Train Dataset1		
		listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Train/dataset1/d1_features_event.csv");
		listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Train/dataset1/d1_features_topic.csv");
		listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Train/dataset1/d1_features_ne.csv");
		
		// Test Positive
//		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Test_label0/testset_features_event.csv");		
//		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Test_label0/testset_features_topic.csv");
//		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Test_label0/testset_features_ne.csv");

		// Test Positive
		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Test_label0/Positive/testset_features_pos_event.csv");		
		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Test_label0/Positive/testset_features_pos_topic.csv");
		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Test_label0/Positive/testset_features_pos_ne.csv");
		
		for (int i = 0; i < listTrainPath.size(); i++) {
			String trainDataPath = listTrainPath.get(i);
			String testDataPath = listTestPath.get(i);
			
			switch (i) {
			case 0:
				resultSvcPath = "result/SVC/event";
				modelSvcPath = "result/SVC/event/model.bin";
				evalSvcPath = resultSvcPath;
				System.out.println("----Event----");
				break;
			case 1:
				resultSvcPath = "result/SVC/topic";
				modelSvcPath = "result/SVC/topic/model.bin";
				evalSvcPath = resultSvcPath;
				System.out.println("----Topic----");
				break;
			case 2:
				resultSvcPath = "result/SVC/ne";
				modelSvcPath = "result/SVC/ne/model.bin";
				evalSvcPath = resultSvcPath;
				System.out.println("----NE----");
				break;
			default:
				break;
			}
			
			System.out.println("-----------");
			System.out.println("Training SVC...");
			wk.trainSVCLeaveOneFeatureOut(trainDataPath, modelSvcPath, "");
			
			System.out.println("-----------");
			System.out.println("Testing SVC and Tfidf...");			
			wk.testSvcAndTfidfLeaveOneFeatureOut(modelSvcPath, resultSvcPath, evalSvcPath, trainDataPath, testDataPath, "");
		}

	}

	public static void runSvmLofo() throws Exception {
		StartWeka wk = new StartWeka();
		System.out.println("Run total...");

		List<String> listTrainDataPath = new ArrayList<>();
		List<String> listTestDataPath = new ArrayList<>();
		List<String> listDatasetName = new ArrayList<>();
		List<String> listCriteria = new ArrayList<>();
		List<String> listFeature = new ArrayList<>();
		List<String> listFeatureIndex = new ArrayList<>();
		List<String> listTailName = new ArrayList<>();

		// Add train path Dataset1
//		listTrainDataPath.add(
//				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset1/d1_features_label_event.csv");
//		listTrainDataPath.add(
//				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset1/d1_features_label_topic.csv");
		listTrainDataPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_07_11_2018/Train/dataset1/d1_features_ne_2.csv");

		// Add test path
		// listTestDataPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/features_082017_ts_event.csv");
		// listTestDataPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/features_082017_ts_topic.csv");
//		 listTestDataPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_07_11_2018/Test/features_082017_ts_ne.csv");

		// Add test path (positive)
//		listTestDataPath.add(
//				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/Positive/features_082017_pos_event.csv");
//		listTestDataPath.add(
//				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/Positive/features_082017_pos_topic.csv");
		listTestDataPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_07_11_2018/Test/Positive/features_082017_pos_ne.csv");

		// Add dataset name
		listDatasetName.add("dataset1");

		// Add criteria name
//		listCriteria.add("event");
//		listCriteria.add("topic");
		listCriteria.add("ne");

		// Add feature name
		//
//		listFeature.add("no_keyword");
//		listFeature.add("no_cosineTF");
//		listFeature.add("no_jaccardBody");
//		listFeature.add("no_jaccardTitle");
//		listFeature.add("no_bm25");
//		listFeature.add("no_lm");
//		listFeature.add("no_ib");
//		listFeature.add("no_avgSim");
//		listFeature.add("no_sumOfMax");
//		listFeature.add("no_maxSim");
//		listFeature.add("no_minSim");
//		listFeature.add("no_jaccardSim");
//		listFeature.add("no_timeSpan");
//		listFeature.add("no_LDASim");
//		listFeature.add("TFIDF");
		
		listFeature.add("keyword");
		listFeature.add("cosineTF");
		listFeature.add("jaccardBody");
		listFeature.add("jaccardTitle");
		listFeature.add("bm25");
		listFeature.add("lm");
		listFeature.add("ib");
		listFeature.add("avgSim");
		listFeature.add("sumOfMax");
		listFeature.add("maxSim");
		listFeature.add("minSim");
		listFeature.add("jaccardSim");
		listFeature.add("timeSpan");
		listFeature.add("LDASim");
		listFeature.add("TFIDF");

		// Add feature name
		listFeatureIndex.add("4,5,6,7,8,9,10,11,12,13,14,15,16,17");
		listFeatureIndex.add("3,5,6,7,8,9,10,11,12,13,14,15,16,17");
		listFeatureIndex.add("3,4,6,7,8,9,10,11,12,13,14,15,16,17");
		listFeatureIndex.add("3,4,5,7,8,9,10,11,12,13,14,15,16,17");
		listFeatureIndex.add("3,4,5,6,8,9,10,11,12,13,14,15,16,17");
		listFeatureIndex.add("3,4,5,6,7,9,10,11,12,13,14,15,16,17");
		listFeatureIndex.add("3,4,5,6,7,8,10,11,12,13,14,15,16,17");
		listFeatureIndex.add("3,4,5,6,7,8,9,11,12,13,14,15,16,17");
		listFeatureIndex.add("3,4,5,6,7,8,9,10,12,13,14,15,16,17");
		listFeatureIndex.add("3,4,5,6,7,8,9,10,11,13,14,15,16,17");
		listFeatureIndex.add("3,4,5,6,7,8,9,10,11,12,14,15,16,17");
		listFeatureIndex.add("3,4,5,6,7,8,9,10,11,12,13,15,16,17");
		listFeatureIndex.add("3,4,5,6,7,8,9,10,11,12,13,14,16,17");
		listFeatureIndex.add("3,4,5,6,7,8,9,10,11,12,13,14,15,17");
		listFeatureIndex.add("3,4,5,6,7,8,9,10,11,12,13,14,15,16");

		for (String dataset : listDatasetName)
			for (String criteria : listCriteria)
				for (String feature : listFeature)
					listTailName.add("_" + dataset + "_" + criteria + "_" + feature);

		int index = 0;
		for (int i = 0; i < listTrainDataPath.size(); i++) {
			for (int j = 0; j < listFeatureIndex.size(); j++) {
				
				switch (i) {
				case 2:
					RESULTPATH = "result/event/"+listTailName.get(index);
					MODElPATH = "result/event/"+listTailName.get(index)+"/model.bin";
					EVALPATH = RESULTPATH;
					TREEPATH = RESULTPATH;
					System.out.println("----Event----");
					break;
				case 1:
					RESULTPATH = "result/topic/"+listTailName.get(index);
					MODElPATH = "result/topic/"+listTailName.get(index)+"/model.bin";
					EVALPATH = RESULTPATH;
					TREEPATH = RESULTPATH;
					System.out.println("----Topic----");
					break;
				case 0:
					RESULTPATH = "result/ne/"+listTailName.get(index);
					MODElPATH = "result/ne/"+listTailName.get(index)+"/model.bin";
					EVALPATH = RESULTPATH;
					TREEPATH = RESULTPATH;
					System.out.println("----NE----");
					break;
				default:
					break;
				}
				
				System.out.println("----------------------------------------------------");
				System.out.println("Feature: " + listFeature.get(j) + " listTrainDataPath=" + i);
				System.out.println("Training SVM...");
				wk.trainSVMLeaveOneFeatureOut(listTrainDataPath.get(i), MODElPATH, listFeatureIndex.get(j));				
				System.out.println("Testing SVM...");
				wk.testSVMLeaveOneFeatureOut(MODElPATH, listTrainDataPath.get(i), listTestDataPath.get(i), listFeatureIndex.get(j));				
				index++;
			}
		}

	}
	
	public static void runSvcLofo() throws Exception {
		StartWeka wk = new StartWeka();
		System.out.println("Run total...");

		List<String> listTrainDataPath = new ArrayList<>();
		List<String> listTestDataPath = new ArrayList<>();
		List<String> listDatasetName = new ArrayList<>();
		List<String> listCriteria = new ArrayList<>();
		List<String> listFeature = new ArrayList<>();
		List<String> listFeatureIndex = new ArrayList<>();
		List<String> listTailName = new ArrayList<>();

		// Add train path Dataset1
//		listTrainDataPath.add(
//				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset1/d1_features_label_event.csv");
//		listTrainDataPath.add(
//				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset1/d1_features_label_topic.csv");
		listTrainDataPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_07_11_2018/Train/dataset1/d1_features_ne_2.csv");

		// Add test path
		// listTestDataPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/features_082017_ts_event.csv");
		// listTestDataPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/features_082017_ts_topic.csv");
//		 listTestDataPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_07_11_2018/Test/features_082017_ts_ne.csv");

		// Add test path (positive)
//		listTestDataPath.add(
//				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/Positive/features_082017_pos_event.csv");
//		listTestDataPath.add(
//				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/Positive/features_082017_pos_topic.csv");
		listTestDataPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_07_11_2018/Test/Positive/features_082017_pos_ne.csv");

		// Add dataset name
		listDatasetName.add("dataset1");

		// Add criteria name
//		listCriteria.add("event");
//		listCriteria.add("topic");
		listCriteria.add("ne");

		// Add feature name		
		listFeature.add("keyword");
		listFeature.add("cosineTF");
		listFeature.add("jaccardBody");
		listFeature.add("jaccardTitle");
		listFeature.add("bm25");
		listFeature.add("lm");
		listFeature.add("ib");
		listFeature.add("avgSim");
		listFeature.add("sumOfMax");
		listFeature.add("maxSim");
		listFeature.add("minSim");
		listFeature.add("jaccardSim");
		listFeature.add("timeSpan");
		listFeature.add("LDASim");
		listFeature.add("TFIDF");

		// Add feature name
		listFeatureIndex.add("4,5,6,7,8,9,10,11,12,13,14,15,16,17");
		listFeatureIndex.add("3,5,6,7,8,9,10,11,12,13,14,15,16,17");
		listFeatureIndex.add("3,4,6,7,8,9,10,11,12,13,14,15,16,17");
		listFeatureIndex.add("3,4,5,7,8,9,10,11,12,13,14,15,16,17");
		listFeatureIndex.add("3,4,5,6,8,9,10,11,12,13,14,15,16,17");
		listFeatureIndex.add("3,4,5,6,7,9,10,11,12,13,14,15,16,17");
		listFeatureIndex.add("3,4,5,6,7,8,10,11,12,13,14,15,16,17");
		listFeatureIndex.add("3,4,5,6,7,8,9,11,12,13,14,15,16,17");
		listFeatureIndex.add("3,4,5,6,7,8,9,10,12,13,14,15,16,17");
		listFeatureIndex.add("3,4,5,6,7,8,9,10,11,13,14,15,16,17");
		listFeatureIndex.add("3,4,5,6,7,8,9,10,11,12,14,15,16,17");
		listFeatureIndex.add("3,4,5,6,7,8,9,10,11,12,13,15,16,17");
		listFeatureIndex.add("3,4,5,6,7,8,9,10,11,12,13,14,16,17");
		listFeatureIndex.add("3,4,5,6,7,8,9,10,11,12,13,14,15,17");
		listFeatureIndex.add("3,4,5,6,7,8,9,10,11,12,13,14,15,16");

		for (String dataset : listDatasetName)
			for (String criteria : listCriteria)
				for (String feature : listFeature)
					listTailName.add("_" + dataset + "_" + criteria + "_" + feature);

		int index = 0;
		for (int i = 0; i < listTrainDataPath.size(); i++) {
			for (int j = 0; j < listFeatureIndex.size(); j++) {
				
				switch (i) {
				case 2:
					RESULTPATH = "result/event/"+listTailName.get(index);
					MODElPATH = "result/event/"+listTailName.get(index)+"/model.bin";
					EVALPATH = RESULTPATH;
					TREEPATH = RESULTPATH;
					System.out.println("----Event----");
					break;
				case 1:
					RESULTPATH = "result/topic/"+listTailName.get(index);
					MODElPATH = "result/topic/"+listTailName.get(index)+"/model.bin";
					EVALPATH = RESULTPATH;
					TREEPATH = RESULTPATH;
					System.out.println("----Topic----");
					break;
				case 0:
					RESULTPATH = "result/ne/"+listTailName.get(index);
					MODElPATH = "result/ne/"+listTailName.get(index)+"/model.bin";
					EVALPATH = RESULTPATH;
					TREEPATH = RESULTPATH;
					System.out.println("----NE----");
					break;
				default:
					break;
				}
				
				System.out.println("----------------------------------------------------");
				System.out.println("Feature: " + listFeature.get(j) + " listTrainDataPath=" + i);
				System.out.println("Training SVM...");
				wk.trainSVCLeaveOneFeatureOut(listTrainDataPath.get(i), MODElPATH, listFeatureIndex.get(j));				
				System.out.println("Testing SVM...");
				wk.testSVCLeaveOneFeatureOut(MODElPATH, listTrainDataPath.get(i), listTestDataPath.get(i), listFeatureIndex.get(j));				
				index++;
			}
		}

	}

	public static void runCVREPTree() throws Exception {
		StartWeka wk = new StartWeka();
		System.out.println("Cross validation REPTree...");

		// String dataPath =
		// "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset1/d1_features_label_event.csv";
		String dataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset1/d1_features_label_event.csv";
		// String dataPath =
		// "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset1/d1_features_label_event.csv";
		wk.crossValidationREPTree(dataPath, 10);
	}

	public static void runCVREPTree2() throws Exception {
		StartWeka wk = new StartWeka();
		System.out.println("Training REPTree...");

		List<String> listTrainPath = new ArrayList<>();
		List<String> listTestPath = new ArrayList<>();
		int folds = 10;

		// Train Dataset1
		listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset1/CV10");

		// Test
		listTestPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset1/CV10");

		for (int i = 0; i < listTrainPath.size(); i++) {
			for (int n = 0; n < folds; n++) {
				String trainDataPath = listTrainPath.get(i);
				String testDataPath = listTestPath.get(i);
				switch (i) {
				case 0:
					RESULTPATH = "result/event/Fold_" + n;
					MODElPATH = "result/event/Fold_" + n + "/model.bin";
					EVALPATH = RESULTPATH;
					TREEPATH = RESULTPATH;
					System.out.println("----Event----");
					break;
				case 1:
					RESULTPATH = "result/topic";
					MODElPATH = "result/topic/model.bin";
					EVALPATH = RESULTPATH;
					TREEPATH = RESULTPATH;
					System.out.println("----Topic----");
					break;
				case 2:
					RESULTPATH = "result/ne";
					MODElPATH = "result/ne/model.bin";
					EVALPATH = RESULTPATH;
					TREEPATH = RESULTPATH;
					System.out.println("----NE----");
					break;
				default:
					break;
				}

				wk.trainREPTree(trainDataPath + "/Fold_" + n + "/d1_features_label_event.csv");
				System.out.println("Testing REPTree...");
				wk.testREPTree(MODElPATH, trainDataPath + "/Fold_" + n + "/d1_features_label_event.csv",
						testDataPath + "/Fold_" + n + "/features_082017_ts_event.csv");
			}
		}
	}

	public static void runTotal() throws Exception {
		StartWeka wk = new StartWeka();
		System.out.println("Run total...");

		List<String> listTrainDataPath = new ArrayList<>();
		List<String> listTestDataPath = new ArrayList<>();
		List<String> listDatasetName = new ArrayList<>();
		List<String> listCriteria = new ArrayList<>();
		List<String> listFeature = new ArrayList<>();
		List<String> listTailName = new ArrayList<>();

		// Add train path Dataset1
		// listTrainDataPath.add(
		// "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset1/d1_features_label_event.csv");
		// listTrainDataPath.add(
		// "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset1/d1_features_label_topic.csv");
		// listTrainDataPath.add(
		// "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset1/d1_features_label_ne.csv");

		// Add train path Dataset2
		// listTrainDataPath.add(
		// "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset2/d2_features_graph_event.csv");
		// listTrainDataPath.add(
		// "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset2/d2_features_graph_topic.csv");
		// listTrainDataPath.add(
		// "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset2/d2_features_graph_ne.csv");

		// Add train path Dataset3
		listTrainDataPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset3/d3_features_graph_event.csv");
		listTrainDataPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset3/d3_features_graph_topic.csv");
		listTrainDataPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Train/dataset3/d3_features_graph_ne.csv");

		// Add test path
		// listTestDataPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/features_082017_ts_event.csv");
		// listTestDataPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/features_082017_ts_topic.csv");
		// listTestDataPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/features_082017_ts_ne.csv");

		// Add test path (positive)
		listTestDataPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/Positive/features_082017_pos_event.csv");
		listTestDataPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/Positive/features_082017_pos_topic.csv");
		listTestDataPath.add(
				"C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_24_10_2018/Test/Positive/features_082017_pos_ne.csv");

		// Add dataset name
		// listDatasetName.add("dataset1");
		// listDatasetName.add("dataset2");
		listDatasetName.add("dataset3");

		// Add criteria name
		listCriteria.add("event");
		listCriteria.add("topic");
		listCriteria.add("ne");

		// Add feature name
		listFeature.add("no_keyword");
		listFeature.add("no_cosineTF");
		listFeature.add("no_jaccardBody");
		listFeature.add("no_jaccardTitle");
		listFeature.add("no_bm25");
		listFeature.add("no_lm");
		listFeature.add("no_ib");
		listFeature.add("no_avgSim");
		listFeature.add("no_sumOfMax");
		listFeature.add("no_maxSim");
		listFeature.add("no_minSim");
		listFeature.add("no_jaccardSim");
		listFeature.add("no_timeSpan");
		listFeature.add("no_LDASim");

		for (String dataset : listDatasetName)
			for (String criteria : listCriteria)
				for (String feature : listFeature)
					listTailName.add("_" + dataset + "_" + criteria + "_" + feature);

		int index = 0;

		for (int featureIndex = 3; featureIndex <= 16; featureIndex++) {
			for (int i = 0; i < listTrainDataPath.size(); i++) {
				System.out.println("----------------------------------------------------");
				System.out.println("FeatureIndex=" + featureIndex + " listTrainDataPath=" + i);
				System.out.println("Training REPTree...");
				// wk.trainREPTreeLeaveOneFeatureOut(listTrainDataPath.get(i),featureIndex,
				// "result/"+listTailName.get(index));
				wk.trainM5PLeaveOneFeatureOut(listTrainDataPath.get(i), featureIndex,
						"result/" + listTailName.get(index));
				System.out.println("Testing REPTree...");
				// wk.testREPTreeLeaveOneFeatureOut("result/"+listTailName.get(index),listTrainDataPath.get(i),
				// listTestDataPath.get(i),featureIndex);
				wk.testM5PLeaveOneFeatureOut("result/" + listTailName.get(index), listTrainDataPath.get(i),
						listTestDataPath.get(i), featureIndex);
				index++;
			}
		}

	}

	public static void createDotGraph(String treePath) throws IOException {
		MutableGraph g = Parser.read(new FileInputStream(treePath + "/tree.dot"));
		Graphviz.fromGraph(g).width(6000).render(Format.PNG).toFile(new File("result/tree1.png"));

		g.graphAttrs().add(Color.WHITE.gradient(Color.rgb("888888")).background().angle(90)).nodeAttrs()
				.add(Color.WHITE.fill()).nodes()
				.forEach(node -> node.add(Color.hsv(.7, .3, 1.0), Style.lineWidth(2).and(Style.FILLED)));

		Graphviz.fromGraph(g).width(6000).render(Format.PNG).toFile(new File("result/tree2.png"));
	}

	public static void main(String[] args) throws Exception {
		// DATASETPATH = args[0];
		// MODElPATH = args[1];
		// RESULTPATH = args[2];
		// EVALPATH = args[3];
		// TREEPATH = args[4];
		// runREPTree();
		// runM5P();
//		runSVC();
		runSVM();
//		runScó vcAndSvr();
//		runSvcAndTfidf();
//		runSvmLofo();
		// runCVREPTree2();
		// runTotal();
		System.out.println("Done!");
	}

}
