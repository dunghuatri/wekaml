package admicro.wekaml;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Debug;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class BaseClass<T> {
    private T object;
    public BaseClass(){}
    public BaseClass(T object){
        this.object = object;
    }

    //if featureIndex = -1 --> normal train
    //if featureIndex != -1 --> train LOFO
    public void train(String trainDataPath, String modelPath, String featureIndex) throws Exception {
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
        Instances preprocessedData = null;
        if(featureIndex == null){
            preprocessedData = prep.removeFeatures(trainData, "1-2");
        }else{
            preprocessedData = prep.removeFeatures(trainData, "1,2," + featureIndex);
        }
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
        object = (T) mg.buildClassifierSVM(datasetnor);
        // -------------------------------------------//
        endTime = System.currentTimeMillis();
        totalTime = endTime - startTime;
        System.out.println("done " + totalTime / 1000 + " s");

        // Save model
        System.out.println("Saving model ...");
        startTime = System.currentTimeMillis();
        // -------------------------------------------//
        mg.saveModel((Classifier) object, modelPath);
        // -------------------------------------------//
        endTime = System.currentTimeMillis();
        totalTime = endTime - startTime;
        System.out.println("done " + totalTime / 1000 + " s");
    }
    public void test(String modelPath, String trainDataPath, String testDataPath, String EVALPATH,
                     String RESULTPATH, Double cut_off, List<Integer> topKList, String featureIndex) throws Exception {
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
        Instances preprocessedTrainData = null;
        Instances preprocessedTestData = null;

        if(featureIndex == null){
            preprocessedTrainData = prep.removeFeatures(trainData, "1-2");
            preprocessedTestData = prep.removeFeatures(testData, "1-2");
        }else{
            preprocessedTrainData = prep.removeFeatures(trainData, "1-2" + featureIndex);
            preprocessedTestData = prep.removeFeatures(testData, "1-2" + featureIndex);
        }
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
        object = (T) mg.loadModel(modelPath);
        // -------------------------------------------//
        endTime = System.currentTimeMillis();
        totalTime = endTime - startTime;
        System.out.println("done " + totalTime / 1000 + " s");

        // Evaluate classifier with test dataset
        System.out.println("Evaluating ...");
        startTime = System.currentTimeMillis();
        // -------------------------------------------//
        Evaluation eval = mg.evaluateModel((Classifier) object, datasetnorTrain, datasetnorTest);
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
        for (int topK : topKList) {
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
    public Classifier loadModel(String modelpath){
        try {
            object = (T) SerializationHelper.read(modelpath);
//			model = (SMO) SerializationHelper.read(modelpath);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return (Classifier) object;
    }


}
