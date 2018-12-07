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

import com.sun.org.apache.regexp.internal.RE;
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
			REPTree loadedModel = (REPTree) mg.loadModel(MODElPATH);
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

	public void trainLOFO(REPTree trainModel, String trainDataPath, String modelPath, String featureIndex) throws Exception {
		BaseClass baseClass = new BaseClass(trainModel);
		baseClass.train(trainDataPath, modelPath, featureIndex);
	}
	public void trainLOFO(M5P trainModel, String trainDataPath, String modelPath, String featureIndex) throws Exception {
		BaseClass baseClass = new BaseClass(trainModel);
		baseClass.train(trainDataPath, modelPath, featureIndex);
	}
	public void trainLOFO(SMOreg trainModel, String trainDataPath, String modelPath, String featureIndex) throws Exception {
		BaseClass baseClass = new BaseClass(trainModel);
		baseClass.train(trainDataPath, modelPath, featureIndex);
	}
	public void trainLOFO(SMO trainModel, String trainDataPath, String modelPath, String featureIndex) throws Exception {
		BaseClass baseClass = new BaseClass(trainModel);
		baseClass.train(trainDataPath, modelPath, featureIndex);
	}

	public void trainModel(REPTree trainModel, String modelPath, String trainDataPath) throws Exception {
		BaseClass baseClass = new BaseClass(trainModel);
		baseClass.train(trainDataPath, modelPath, null);
	}

	public void trainModel(M5P trainModel, String modelPath, String trainDataPath) throws Exception {
		BaseClass baseClass = new BaseClass(trainModel);
		baseClass.train(trainDataPath, modelPath, null);
	}

	public void trainModel(SMOreg trainModel, String modelPath, String trainDataPath) throws Exception {
		BaseClass svm = new BaseClass(trainModel);
		svm.train(trainDataPath, modelPath, null);
	}

	public void testLOFO(REPTree trainModel, String modelPath, String trainDataPath, String testDataPath,
						 String evalPath, String resultPath, String featureIndex) throws Exception {
		BaseClass svm = new BaseClass(trainModel);
		svm.test(modelPath, trainDataPath,testDataPath,evalPath, resultPath, cut_off,topK, featureIndex);
	}
	public void testLOFO(M5P trainModel, String modelPath, String trainDataPath, String testDataPath,
						 String evalPath, String resultPath, String featureIndex) throws Exception {
		BaseClass svm = new BaseClass(trainModel);
		svm.test(modelPath, trainDataPath,testDataPath,evalPath, resultPath, cut_off,topK, featureIndex);
	}
	public void testLOFO(SMO trainModel, String modelPath, String trainDataPath, String testDataPath,
						 String evalPath, String resultPath, String featureIndex) throws Exception {
		BaseClass svm = new BaseClass(trainModel);
		svm.test(modelPath, trainDataPath,testDataPath,evalPath, resultPath, cut_off,topK, featureIndex);
	}
	public void testLOFO(SMOreg trainModel, String modelPath, String trainDataPath, String testDataPath,
						 String evalPath, String resultPath, String featureIndex) throws Exception {
		BaseClass svm = new BaseClass(trainModel);
		svm.test(modelPath, trainDataPath, testDataPath,evalPath, resultPath, cut_off,topK, featureIndex);
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

	public void testModel(REPTree trainModel, String modelPath, String trainDataPath, String testDataPath,
							String EVALPATH, String RESULTPATH, String featureIndex) throws Exception {
		BaseClass baseClass = new BaseClass(trainModel);
		baseClass.test(modelPath, trainDataPath, testDataPath, EVALPATH, RESULTPATH, cut_off, topK, featureIndex);
	}

	public void testModel(M5P trainModel, String modelPath, String trainDataPath, String testDataPath,
						String EVALPATH, String RESULTPATH, String featureIndex) throws Exception {
		BaseClass baseClass = new BaseClass(trainModel);
		baseClass.test(modelPath, trainDataPath, testDataPath, EVALPATH, RESULTPATH, cut_off, topK, featureIndex);
	}

	public void testModel(SMOreg trainModel, String modelPath, String trainDataPath, String testDataPath,
						String EVALPATH, String RESULTPATH, String featureIndex) throws Exception {
		BaseClass baseClass = new BaseClass(trainModel);
		baseClass.test(modelPath, trainDataPath, testDataPath, EVALPATH, RESULTPATH, cut_off, topK, featureIndex);
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
			REPTree trainModel = new REPTree();
			wk.trainModel(trainModel, MODElPATH, trainDataPath);
			System.out.println("Testing REPTree...");
			wk.testModel(trainModel, MODElPATH, trainDataPath, testDataPath, EVALPATH, RESULTPATH, "");
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

			M5P trainModel = new M5P();
			wk.trainModel(trainModel, MODElPATH, trainDataPath);
			System.out.println("Testing M5P...");
			wk.testModel(trainModel, MODElPATH, trainDataPath, testDataPath, EVALPATH, RESULTPATH, null);
		}

	}

	public static void runSVM() throws Exception {
		StartWeka wk = new StartWeka();		

		List<String> listTrainPath = new ArrayList<>();
		List<String> listTestPath = new ArrayList<>();

		// Train Dataset1		
		 listTrainPath.add("/home/lana/Downloads/Train/dataset1/d1_features_event.csv");
//		 listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Train/dataset1/d1_features_topic.csv");
//		 listTrainPath.add("C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Train/dataset1/d1_features_ne.csv");

		// Test
		listTestPath.add("/home/lana/Downloads/Test_label0/testset_features_event.csv");
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
			SMOreg trainModel = new SMOreg();
			wk.trainModel(trainModel, MODElPATH, trainDataPath);
			System.out.println("Testing SVM...");
			wk.testModel(trainModel, MODElPATH, trainDataPath, testDataPath, EVALPATH, RESULTPATH, null);
//			wk.testLOFO(trainModel, MODElPATH, trainDataPath, testDataPath, EVALPATH, RESULTPATH, "");
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
			SMO trainModel = new SMO();
			wk.trainLOFO(trainModel, trainDataPath, MODElPATH, null);
			System.out.println("-----------");
			System.out.println("Testing SVC...");
			wk.testLOFO(trainModel, MODElPATH, trainDataPath, testDataPath,
					EVALPATH, RESULTPATH, "");
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
			SMO trainModelSVC = new SMO();
			SMOreg trainModelSVR = new SMOreg();
			System.out.println("-----------");
			System.out.println("Training SVC...");
			wk.trainLOFO(trainModelSVC, trainDataPath, modelSvcPath, null);
			System.out.println("-----------");
			System.out.println("Training SVR...");
			wk.trainLOFO(trainModelSVR, trainDataPath, modelSvrPath, null);
			
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
			SMO trainModel = new SMO();
			wk.trainLOFO(trainModel, trainDataPath, modelSvcPath, null);
			
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
		int ignoredIndex = 3;
		for(int i = 3; i <= listFeature.size(); i++){
			String str = "";
			for(int j = i; j < 18; j++){
				if(ignoredIndex == i){
					continue;
				}
				if(ignoredIndex == 17){
					str += j;
				}else{
					str += j +",";
				}
			}
			ignoredIndex++;
		}

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
				SMOreg trainModel = new SMOreg();
				wk.trainLOFO(trainModel, listTrainDataPath.get(i), MODElPATH, listFeatureIndex.get(j));
				System.out.println("Testing SVM...");
				wk.testLOFO(trainModel, MODElPATH, listTrainDataPath.get(i), listTestDataPath.get(i),
						EVALPATH, RESULTPATH, listFeatureIndex.get(j));
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
		int ignoredIndex = 3;
		for(int i = 3; i <= listFeature.size(); i++){
			String str = "";
			for(int j = i; j < 18; j++){
				if(ignoredIndex == i){
					continue;
				}
				if(ignoredIndex == 17){
					str += j;
				}else{
					str += j +",";
				}
			}
			ignoredIndex++;
		}

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
				SMOreg trainModel = new SMOreg();
				wk.trainLOFO(trainModel, listTrainDataPath.get(i), MODElPATH, listFeatureIndex.get(j));
				System.out.println("Testing SVM...");
				wk.testLOFO(trainModel, MODElPATH, listTrainDataPath.get(i), listTestDataPath.get(i),
						EVALPATH, RESULTPATH, listFeatureIndex.get(j));
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
				REPTree trainModel = new REPTree();
//				wk.trainREPTree(trainDataPath + "/Fold_" + n + "/d1_features_label_event.csv");
				System.out.println("Testing REPTree...");

				wk.testModel(trainModel, MODElPATH, trainDataPath + "/Fold_" + n + "/d1_features_label_event.csv",
						testDataPath + "/Fold_" + n + "/features_082017_ts_event.csv", EVALPATH, RESULTPATH, null);
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
				M5P trainModel = new M5P();
				wk.trainLOFO(trainModel, listTrainDataPath.get(i),
						"result/" + listTailName.get(index), String.valueOf(featureIndex));
				System.out.println("Testing REPTree...");
				// wk.testREPTreeLeaveOneFeatureOut("result/"+listTailName.get(index),listTrainDataPath.get(i),
				// listTestDataPath.get(i),featureIndex);
				wk.testLOFO(trainModel, "result/" + listTailName.get(index), listTrainDataPath.get(i), listTestDataPath.get(i),
						EVALPATH, RESULTPATH, String.valueOf(featureIndex));
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
//		runREPTree();
		runSVM();
//		runScó vcAndSvr();
//		runSvcAndTfidf();
//		runSvmLofo();
		// runCVREPTree2();
		// runTotal();
		System.out.println("Done!");
	}

}
