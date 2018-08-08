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
	public static final String DATASETPATH = "data/heart.arff";
	public static final String MODElPATH = "model/model.bin";
	public static final String RESULTPATH = "result/result.csv";
	public static final String EVALPATH = "result/eval.txt";

	public static void main(String[] args) throws Exception {
		ModelGenerator mg = new ModelGenerator();		
		Instances originalData = mg.loadDataset(DATASETPATH);
		
		// Preprocess data
		Preporcess prep = new Preporcess();
		Instances preprocessedData = prep.Numeric2Nominal(originalData,"last");

		Filter filter = new Normalize();

		// divide dataset to train dataset 80% and test dataset 20%
		int trainSize = (int) Math.round(preprocessedData.numInstances() * 0.8);
		int testSize = preprocessedData.numInstances() - trainSize;
		
		
		preprocessedData.randomize(new Debug.Random(1));// if you comment this line the
												// accuracy of the model will be
												// droped down
		
		// Normalize dataset
		filter.setInputFormat(preprocessedData);		
		Instances datasetnor = Filter.useFilter(preprocessedData, filter);		

		Instances traindataset = new Instances(datasetnor, 0, trainSize);
		Instances testdataset = new Instances(datasetnor, trainSize, testSize);

		// build classifier with train dataset
		REPTree trainedModel = (REPTree) mg.buildClassifier(traindataset);
		
		// Save model
		mg.saveModel(trainedModel, MODElPATH);
		
		// Load model
		REPTree loadedModel = (REPTree) mg.loadModel(MODElPATH);

		// Evaluate classifier with test dataset
		Evaluation eval = mg.evaluateModel(loadedModel, traindataset, testdataset);
		System.out.println("Evaluation: " + eval.toSummaryString("", true));

		// Save predicted results
		mg.savePredicted(eval, RESULTPATH);

		// Save evaluation
		mg.saveEvaluation(eval, EVALPATH);

	}

}
