package admicro.wekaml;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import weka.core.Instance;

public class Test_ModelGenerator {

	/**
	 * Evaluate L2R
	 * 
	 * @throws IOException
	 */
	public static void evalL2R() throws IOException {
		// Danh gia Learning to rank
		System.out.println("Learning to rank");
		ModelGenerator mg = new ModelGenerator();
		List<Integer> topK = new ArrayList<>();
		topK.add(1);
		topK.add(3);
		topK.add(4);
		topK.add(5);
		topK.add(10);
		double cut_off = 0;

		List<String> listDatasetName = new ArrayList<>();
		List<String> listCriteria = new ArrayList<>();
		List<String> listFeature = new ArrayList<>();
		List<String> listTailName = new ArrayList<>();

		// Add dataset name
		// listDatasetName.add("dataset1");
		// listDatasetName.add("dataset2");
		listDatasetName.add("dataset3");

		// Add criteria name
		// listCriteria.add("event");
		// listCriteria.add("topic");
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
		// for (String dataset : listDatasetName)
		// for (String criteria : listCriteria)
		// listTailName.add("_" + dataset + "_" + criteria);

		String RESULTPATH = "C:/Users/ADMIN/Desktop/Demo/result_L2R/24_10_2018/leave_one_feature_out/Positive/"
				+ listDatasetName.get(0) + "/";

		for (int index = 0; index < listTailName.size(); index++) {
			String modelPath = RESULTPATH + listTailName.get(index);
			System.out.println("----------------------------------------------------");
			System.out.println(listTailName.get(index));

			mg.convertScoreToLabelWithId(modelPath + "/result_Id_score.csv", modelPath + "/result_Id_label.csv",
					cut_off);

			// Sort result
			HashMap<String, List<Instance>> sortedResultByPredictedScore = mg
					.sortResultByAttribute(modelPath + "/result_Id_score.csv", 2);
			mg.saveSortedResult(modelPath + "/result_Id_score_sorted.csv", sortedResultByPredictedScore);
			HashMap<String, List<Instance>> sortedResultByActualScore = mg
					.sortResultByAttribute(modelPath + "/result_Id_score.csv", 3);

			HashMap<Integer, List<String>> listResultTopK = new HashMap<>();
			for (int k : topK) {
				List<String> listResult = new ArrayList<>();
				listResultTopK.put(k, listResult);
				mg.saveEvaluationTopK(modelPath + "/evalTopK_" + Integer.toString(k) + ".csv",
						sortedResultByPredictedScore, k, cut_off, listResultTopK);
				mg.saveNDCGTopK(modelPath + "/NDCGTopK" + Integer.toString(k) + ".csv", sortedResultByPredictedScore,
						sortedResultByActualScore, k, cut_off, listResultTopK);
				mg.saveEvaluationNoTime(sortedResultByPredictedScore,
						modelPath + "/eval_" + Integer.toString(k) + ".txt", modelPath + "/result_Id_label.csv", k,
						listResultTopK);
			}
			mg.saveEvaluationSumary(modelPath + "/evalSumary.csv", listResultTopK);
		}
	}

	/**
	 * Evaluate TFIDF
	 * 
	 * @throws IOException
	 */
	public static void evalTFIDF() throws IOException {
		// Danh gia TF-IDF
		System.out.println("TF-IDF");
		ModelGenerator mg = new ModelGenerator();
		List<Integer> topK = new ArrayList<>();
		topK.add(1);
		topK.add(3);
		topK.add(4);
		topK.add(5);
		topK.add(10);
		double cut_off = 0;

		List<String> listCriteria = new ArrayList<>();

		// Add criteria name
//		listCriteria.add("event");
//		listCriteria.add("topic");
		listCriteria.add("ne");

		String RESULTPATH = "C:/Users/ADMIN/Desktop/Demo/data/feature_newsId_26_11_2018/Test/Sua_nhan/";

		for (int index = 0; index < listCriteria.size(); index++) {
			String modelPath = RESULTPATH + listCriteria.get(index);
			System.out.println("----------------------------------------------------");
			System.out.println(listCriteria.get(index));

			mg.convertScoreToLabelWithId(modelPath + "/result_Id_score.csv", modelPath + "/result_Id_label.csv",
					cut_off);

			// Sort result
			HashMap<String, List<Instance>> sortedResultByPredictedScore = mg
					.sortResultByAttribute(modelPath + "/result_Id_score.csv", 2);
			mg.saveSortedResult(modelPath + "/result_Id_score_sorted.csv", sortedResultByPredictedScore);
			HashMap<String, List<Instance>> sortedResultByActualScore = mg
					.sortResultByAttribute(modelPath + "/result_Id_score.csv", 3);

			HashMap<Integer, List<String>> listResultTopK = new HashMap<>();
			for (int k : topK) {
				List<String> listResult = new ArrayList<>();
				listResultTopK.put(k, listResult);
				mg.saveEvaluationTopK(modelPath + "/evalTopK_" + Integer.toString(k) + ".csv",
						sortedResultByPredictedScore, k, cut_off, listResultTopK);
				mg.saveNDCGTopK(modelPath + "/NDCGTopK" + Integer.toString(k) + ".csv", sortedResultByPredictedScore,
						sortedResultByActualScore, k, cut_off, listResultTopK);
				mg.saveEvaluationNoTime(sortedResultByPredictedScore,
						modelPath + "/eval_" + Integer.toString(k) + ".txt", modelPath + "/result_Id_label.csv", k,
						listResultTopK);
			}
			mg.saveEvaluationSumary(modelPath + "/evalSumary.csv", listResultTopK);
		}
	}

	/**
	 * JUST FOR TESTING CODE
	 * 
	 * @throws IOException
	 */
	public static void testNDCG() throws IOException {
		ModelGenerator mg = new ModelGenerator();
		HashMap<String, List<Instance>> sortedResultByPredictedScore = mg
				.sortResultByAttribute("test_case/inputNDCG.csv", 2);
		HashMap<String, List<Instance>> sortedResultByActualScore = mg.sortResultByAttribute("test_case/inputNDCG.csv",
				3);
		List<String[]> listResult = mg.calculateNDCGTopK(sortedResultByPredictedScore, sortedResultByActualScore, 10,
				0);
		for (int i = 0; i < listResult.size(); i++) {
			System.out.println(listResult.get(i)[0] + "-->" + listResult.get(i)[1]);
		}
		// Output: 1-->0.9608081943360616
	}

	/**
	 * JUST FOR TESTING CODE
	 */
	public static void testMRR() {
		// Test MRR
		ModelGenerator mg = new ModelGenerator();
		HashMap<String, List<Instance>> sortedResult = mg.sortResultByAttribute("test_case/inputMRR.csv", 2);
		System.out.println(mg.calculateMRR(sortedResult, 100));
		// Output: 0.611111111111111
	}

	/**
	 * JUST FOR TESTING CODE
	 */
	public static void testRMSE() {
		// Test RMSE
		ModelGenerator mg = new ModelGenerator();
		HashMap<String, List<Instance>> sortedResult = mg.sortResultByAttribute("test_case/inputMRR.csv", 2);
		System.out.println(mg.calculateRMSE(sortedResult, 100));
		// Output: 1.9148542155126762
	}

	public static void main(String[] args) throws IOException {
		System.out.println("(/・・)ノ");
		evalTFIDF();
		System.out.println("(*•̀ᴗ•́*)و ̑̑");
	}

}
