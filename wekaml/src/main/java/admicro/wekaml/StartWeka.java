/**
 * 
 */
package admicro.wekaml;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
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

public class StartWeka {

//	private static final String DATASETPATH = "data/iris.2D.arff";
//	private static final String DATASETPATH = "data/heart.csv";
	private static String DATASETPATH = "C:/Users/ADMIN/Desktop/Demo/data/features_graph_event.csv";
//	private static String DATASETPATH = "C:/Users/ADMIN/Desktop/Demo/data/features_graph_topic.csv";
//	private static String DATASETPATH = "C:/Users/ADMIN/Desktop/Demo/data/features_graph_ne.csv";
//	private static String MODElPATH = "model/REPTree_model_ne.bin";
	private static String MODElPATH = "model/M5P_model_ne.bin";
	private static String RESULTPATH = "result";
	private static String EVALPATH = "result";
	private static String TREEPATH = "result";
	private double cut_off = 0;
	private int topK = 2;
	
	public static void runWeka() throws Exception
	{
		/*Measure time*/
		long startTime;
		long endTime;
		long totalTime;
		
		ModelGenerator mg = new ModelGenerator();		
		Instances originalData = mg.loadDataset(DATASETPATH);
		
		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		Preporcess prep = new Preporcess();
//		Instances preprocessedData = prep.Numeric2Nominal(originalData,"last");
		Instances preprocessedData = prep.removeFeatures(originalData, "1-2");		
//		Instances preprocessedData = originalData;
		
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");

		Filter filter = new Normalize();

		// divide dataset to train dataset 80% and test dataset 20%
		int trainSize = (int) Math.round(preprocessedData.numInstances() * 0.6);
		int testSize = preprocessedData.numInstances() - trainSize;
		
		
		preprocessedData.randomize(new Debug.Random(1));// if you comment this line the
												// accuracy of the model will be
												// droped down
		
		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		filter.setInputFormat(preprocessedData);		
		Instances datasetnor = Filter.useFilter(preprocessedData, filter);
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");

		System.out.println("Deviding dataset ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		Instances traindataset = new Instances(datasetnor, 0, trainSize);
		Instances testdataset = new Instances(datasetnor, trainSize, testSize);
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");

		// build classifier with train dataset
		System.out.println("Building model ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		REPTree trainedModel = (REPTree) mg.buildClassifierREPTree(traindataset);
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");
		
		// Save model
		System.out.println("Saving model ...");		
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		mg.saveModel(trainedModel, MODElPATH);
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");
		
		// Load model
		System.out.println("Loading model ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		REPTree loadedModel = (REPTree) mg.loadModelREPTree(MODElPATH);
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");

		// Evaluate classifier with test dataset
		System.out.println("Evaluating ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		Evaluation eval = mg.evaluateModel(loadedModel, traindataset, testdataset);
//		System.out.println("Evaluation: " + eval.toSummaryString("", true));
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");

		// Save predicted results
		System.out.println("Saving results ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		mg.savePredicted(eval, RESULTPATH);
		mg.convertScoreToLabel(RESULTPATH, RESULTPATH, 0.6);
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");

		// Save evaluation
		System.out.println("Saving evaluation ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		//Sort result
		HashMap<String, List<Instance>> sortedResult = mg.sortResultByAttribute(EVALPATH + "/result_Id_score.csv", 2);
		mg.saveSortedResult(EVALPATH, sortedResult);
		mg.saveEvaluation(eval, sortedResult, EVALPATH,RESULTPATH+"/result_Id_label.csv");
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");
		
		// Show tree
		System.out.println("Showing tree ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		mg.showTree(loadedModel,TREEPATH);
		createDotGraph();
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");
	}
	
	public void trainREPTree(String trainDataPath) throws Exception
	{
		/*Measure time*/
		long startTime;
		long endTime;
		long totalTime;
		
		ModelGenerator mg = new ModelGenerator();		
		Instances trainData = mg.loadDataset(trainDataPath);
		
		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedData = prep.removeFeatures(trainData, "1-2");
		preprocessedData.randomize(new Debug.Random(1));
//		Instances preprocessedData = trainData;
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");
		
		Filter filter = new Normalize();
		
		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		filter.setInputFormat(preprocessedData);		
		Instances datasetnor = Filter.useFilter(preprocessedData, filter);
//		Instances datasetnor = preprocessedData;
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");

		// build classifier with train dataset
		System.out.println("Building model ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		REPTree trainedModel = (REPTree) mg.buildClassifierREPTree(datasetnor);
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");
		
		// Save model
		System.out.println("Saving model ...");		
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		mg.saveModel(trainedModel, MODElPATH);		
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");
				
		// Show tree
		System.out.println("Showing tree ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		mg.showTree(trainedModel,TREEPATH);
		createDotGraph();
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");
	}
	
	public void trainM5P(String trainDataPath) throws Exception
	{
		/*Measure time*/
		long startTime;
		long endTime;
		long totalTime;
		
		ModelGenerator mg = new ModelGenerator();		
		Instances trainData = mg.loadDataset(trainDataPath);
		
		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedData = prep.removeFeatures(trainData, "1-2");
		preprocessedData.randomize(new Debug.Random(1));
//		Instances preprocessedData = trainData;
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");
		
		Filter filter = new Normalize();
		
		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		filter.setInputFormat(preprocessedData);		
		Instances datasetnor = Filter.useFilter(preprocessedData, filter);
//		Instances datasetnor = preprocessedData;
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");

		// build classifier with train dataset
		System.out.println("Building model ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		M5P trainedModel = (M5P) mg.buildClassifierM5P(datasetnor);
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");
		
		// Save model
		System.out.println("Saving model ...");		
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		mg.saveModel(trainedModel, MODElPATH);		
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");
				
		// Show tree
		System.out.println("Showing tree ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		mg.showTree(trainedModel,TREEPATH);
		createDotGraph();
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");
	}
	
	public void testREPTree(String modelPath, String trainDataPath, String testDataPath) throws Exception
	{
		/*Measure time*/
		long startTime;
		long endTime;
		long totalTime;
		
		ModelGenerator mg = new ModelGenerator();		
		Instances trainData = mg.loadDataset(trainDataPath);
//		Instances testData = mg.loadDataset(testDataPath);
		Instances testData = mg.loadDatasetWithId(testDataPath);
		
		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedTrainData = prep.removeFeatures(trainData, "1-2");
		Instances preprocessedTestData = prep.removeFeatures(testData, "1-2");
		
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");
		
		Filter filterTrain = new Normalize();
		Filter filterTest = new Normalize();
		
		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		filterTrain.setInputFormat(preprocessedTrainData);		
		Instances datasetnorTrain = Filter.useFilter(preprocessedTrainData, filterTrain);
		filterTest.setInputFormat(preprocessedTestData);
		Instances datasetnorTest = Filter.useFilter(preprocessedTestData, filterTest);
		
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");
				
		// Load model
		System.out.println("Loading model ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		REPTree loadedModel = (REPTree) mg.loadModelREPTree(modelPath);
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");

		// Evaluate classifier with test dataset
		System.out.println("Evaluating ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		Evaluation eval = mg.evaluateModel(loadedModel, datasetnorTrain, datasetnorTest);
//		System.out.println("Evaluation: " + eval.toSummaryString("", true));
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");

		// Save predicted results
		System.out.println("Saving results ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		mg.savePredictedWithId(testData, eval, RESULTPATH+"/result_Id_score.csv");	
		mg.convertScoreToLabelWithId(RESULTPATH+"/result_Id_score.csv", RESULTPATH+"/result_Id_label.csv", cut_off);
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");

		// Save evaluation
		System.out.println("Saving evaluation ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//		
		//Sort result
		HashMap<String, List<Instance>> sortedResultByPredictedScore = mg.sortResultByAttribute(EVALPATH + "/result_Id_score.csv", 2);
		mg.saveSortedResult(EVALPATH+"/result_Id_score_sorted.csv", sortedResultByPredictedScore);
		HashMap<String, List<Instance>> sortedResultByActualScore = mg.sortResultByAttribute(EVALPATH + "/result_Id_score.csv", 3);
		
		mg.saveEvaluation(eval, sortedResultByPredictedScore, EVALPATH+"/eval.txt", RESULTPATH+"/result_Id_label.csv");
		for(int k=2;k<=20;k=k+2)
		{
			topK = k;
			mg.saveEvaluationTopK(EVALPATH+"/evalTopK_"+Integer.toString(topK)+".csv", sortedResultByPredictedScore,topK,cut_off);
			mg.saveNDCGTopK(EVALPATH+"/NDCGTopK"+Integer.toString(topK)+".csv", sortedResultByPredictedScore, sortedResultByActualScore, topK, cut_off);
		}		
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");
	}
	
	public void testM5P(String modelPath, String trainDataPath, String testDataPath) throws Exception
	{
		/*Measure time*/
		long startTime;
		long endTime;
		long totalTime;
		
		ModelGenerator mg = new ModelGenerator();		
		Instances trainData = mg.loadDataset(trainDataPath);
//		Instances testData = mg.loadDataset(testDataPath);
		Instances testData = mg.loadDatasetWithId(testDataPath);
		
		// Preprocess data
		System.out.println("Preprocessing ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		Preporcess prep = new Preporcess();
		Instances preprocessedTrainData = prep.removeFeatures(trainData, "1-2");
		Instances preprocessedTestData = prep.removeFeatures(testData, "1-2");
		
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");
		
		Filter filterTrain = new Normalize();
		Filter filterTest = new Normalize();
		
		// Normalize dataset
		System.out.println("Normalizing ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		filterTrain.setInputFormat(preprocessedTrainData);		
		Instances datasetnorTrain = Filter.useFilter(preprocessedTrainData, filterTrain);
		filterTest.setInputFormat(preprocessedTestData);
		Instances datasetnorTest = Filter.useFilter(preprocessedTestData, filterTest);
		
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");
				
		// Load model
		System.out.println("Loading model ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		M5P loadedModel = (M5P) mg.loadModelM5P(modelPath);
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");

		// Evaluate classifier with test dataset
		System.out.println("Evaluating ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		Evaluation eval = mg.evaluateModel(loadedModel, datasetnorTrain, datasetnorTest);
//		System.out.println("Evaluation: " + eval.toSummaryString("", true));
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");

		// Save predicted results
		System.out.println("Saving results ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		mg.savePredictedWithId(testData, eval, RESULTPATH+"/result_Id_score.csv");	
		mg.convertScoreToLabelWithId(RESULTPATH+"/result_Id_score.csv", RESULTPATH+"/result_Id_label.csv", cut_off);
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");

		// Save evaluation
		System.out.println("Saving evaluation ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//		
		//Sort result
		HashMap<String, List<Instance>> sortedResultByPredictedScore = mg.sortResultByAttribute(EVALPATH + "/result_Id_score.csv", 2);
		mg.saveSortedResult(EVALPATH+"/result_Id_score_sorted.csv", sortedResultByPredictedScore);
		HashMap<String, List<Instance>> sortedResultByActualScore = mg.sortResultByAttribute(EVALPATH + "/result_Id_score.csv", 3);
		
		mg.saveEvaluation(eval, sortedResultByPredictedScore, EVALPATH+"/eval.txt", RESULTPATH+"/result_Id_label.csv");
		for(int k=2;k<=20;k=k+2)
		{
			topK = k;
			mg.saveEvaluationTopK(EVALPATH+"/evalTopK_"+Integer.toString(topK)+".csv", sortedResultByPredictedScore,topK,cut_off);
			mg.saveNDCGTopK(EVALPATH+"/NDCGTopK"+Integer.toString(topK)+".csv", sortedResultByPredictedScore, sortedResultByActualScore, topK, cut_off);
		}	
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");
	}
		
	public static void runREPTree() throws Exception
	{
		StartWeka wk = new StartWeka();
		System.out.println("Training REPTree...");
//		String trainDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset1/Train_Test_9_1/Train_event_9.csv";
//		String testDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset1/Test_full_matrix/features_event_fullMatrix.csv";
//		String trainDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset1/Train_Test_9_1/Train_topic_9.csv";
//		String testDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset1/Test_full_matrix/features_topic_fullMatrix.csv";
		String trainDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset1/Train_Test_9_1/Train_ne_9.csv";
		String testDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset1/Test_full_matrix/features_ne_fullMatrix.csv";
		
//		String trainDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset2/Train_Test/Train_graph_event_3000.csv";
//		String testDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset2/Train_Test/Test_graph_event_3000.csv";
//		String trainDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset2/Train_Test/Train_graph_topic_3000.csv";
//		String testDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset2/Train_Test/Test_graph_topic_3000.csv";
//		String trainDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset2/Train_Test/Train_graph_ne_3000.csv";
//		String testDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset2/Train_Test/Test_graph_ne_3000.csv";
		
//		String trainDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset3/Train_Test/Train_graph_event_1992.csv";
//		String testDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset3/Train_Test/Test_graph_event_1992.csv";
//		String trainDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset3/Train_Test/Train_graph_topic_1992.csv";
//		String testDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset3/Train_Test/Test_graph_topic_1992.csv";
//		String trainDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset3/Train_Test/Train_graph_ne_1992.csv";
//		String testDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset3/Train_Test/Test_graph_ne_1992.csv";
		
		wk.trainREPTree(trainDataPath);
		System.out.println("Testing REPTree...");
		wk.testREPTree(MODElPATH, trainDataPath, testDataPath);
	}
	
	public static void runM5P() throws Exception
	{
		StartWeka wk = new StartWeka();
		System.out.println("Training M5P...");
//		String trainDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset1/Train_Test_9_1/Train_event_9.csv";
//		String testDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset1/Test_full_matrix/features_event_fullMatrix.csv";
//		String trainDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset1/Train_Test_9_1/Train_topic_9.csv";
//		String testDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset1/Test_full_matrix/features_topic_fullMatrix.csv";
		String trainDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset1/Train_Test_9_1/Train_ne_9.csv";
		String testDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset1/Test_full_matrix/features_ne_fullMatrix.csv";
		
//		String trainDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset2/Train_Test/Train_graph_event_3000.csv";
//		String testDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset2/Train_Test/Test_graph_event_3000.csv";
//		String trainDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset2/Train_Test/Train_graph_topic_3000.csv";
//		String testDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset2/Train_Test/Test_graph_topic_3000.csv";
//		String trainDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset2/Train_Test/Train_graph_ne_3000.csv";
//		String testDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset2/Train_Test/Test_graph_ne_3000.csv";
		
//		String trainDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset3/Train_Test/Train_graph_event_1992.csv";
//		String testDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset3/Train_Test/Test_graph_event_1992.csv";
//		String trainDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset3/Train_Test/Train_graph_topic_1992.csv";
//		String testDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset3/Train_Test/Test_graph_topic_1992.csv";
//		String trainDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset3/Train_Test/Train_graph_ne_1992.csv";
//		String testDataPath = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_04_09_2018/dataset3/Train_Test/Test_graph_ne_1992.csv";
		
		wk.trainM5P(trainDataPath);
		System.out.println("Testing M5P...");
		wk.testM5P(MODElPATH, trainDataPath, testDataPath);
	}
	
	public static void createDotGraph() throws IOException
	{
		MutableGraph g = Parser.read(new FileInputStream(TREEPATH+"/tree.dot"));
		Graphviz.fromGraph(g).width(6000).render(Format.PNG).toFile(new File("result/tree1.png"));

		g.graphAttrs()
		        .add(Color.WHITE.gradient(Color.rgb("888888")).background().angle(90))
		        .nodeAttrs().add(Color.WHITE.fill())
		        .nodes().forEach(node -> node.add(Color.hsv(.7, .3, 1.0), Style.lineWidth(2).and(Style.FILLED)));
		
		
		Graphviz.fromGraph(g).width(6000).render(Format.PNG).toFile(new File("result/tree2.png"));
	}

	public static void main(String[] args) throws Exception {
//		DATASETPATH = args[0];
//		MODElPATH = args[1];
//		RESULTPATH = args[2];
//		EVALPATH = args[3];
//		TREEPATH = args[4];
//		runWeka();
//		runREPTree();
		runM5P();
		System.out.println("Done!");
	}

}
