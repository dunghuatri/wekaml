/**
 * 
 */
package admicro.wekaml;

import java.io.File;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.REPTree;
import weka.core.Debug;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

/**
 * @author Hua Tri Dung
 *
 */

public class StartWeka {

//	public static final String DATASETPATH = "data/iris.2D.arff";
//	public static final String DATASETPATH = "data/heart.csv";	
	public static String DATASETPATH = "data/heart.arff";
	public static String MODElPATH = "model/model.bin";
	public static String RESULTPATH = "result/result.csv";
	public static String EVALPATH = "result/eval.txt";
	
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
		Instances preprocessedData = prep.Numeric2Nominal(originalData,"last");
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");

		Filter filter = new Normalize();

		// divide dataset to train dataset 80% and test dataset 20%
		int trainSize = (int) Math.round(preprocessedData.numInstances() * 0.8);
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
		REPTree trainedModel = (REPTree) mg.buildClassifier(traindataset);
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
		REPTree loadedModel = (REPTree) mg.loadModel(MODElPATH);
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
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");

		// Save evaluation
		System.out.println("Saving evaluation ...");
		startTime = System.currentTimeMillis();
		//-------------------------------------------//
		mg.saveEvaluation(eval, EVALPATH);
		//-------------------------------------------//
		endTime = System.currentTimeMillis();
		totalTime = endTime-startTime;
		System.out.println("done "+totalTime/1000+" s");
	}

	public static void main(String[] args) throws Exception {
		DATASETPATH = args[0];
		MODElPATH = args[1];
		RESULTPATH = args[2];
		EVALPATH = args[3];
		runWeka();
		System.out.println("Done!");
	}

}
