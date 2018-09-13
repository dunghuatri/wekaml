package admicro.wekaml;

import java.io.IOException;
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
		String RESULTPATH = "C:/Users/ADMIN/Desktop/Demo/result_L2R/04_09_2018/dataset1/Test_full_matrix/ne";
		int topK = 0;
		double cut_off = 0;
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
		
		mg.convertScoreToLabelWithId(RESULTPATH+"/result_Id_score.csv", RESULTPATH+"/result_Id_label.csv", cut_off);

		// Sort result
		HashMap<String, List<Instance>> sortedResultByPredictedScore = mg
				.sortResultByAttribute(RESULTPATH + "/result_Id_score.csv", 2);
		mg.saveSortedResult(RESULTPATH+"/result_Id_score_sorted.csv", sortedResultByPredictedScore);
		HashMap<String, List<Instance>> sortedResultByActualScore = mg
				.sortResultByAttribute(RESULTPATH + "/result_Id_score.csv", 3);

		mg.saveEvaluationNoTime(sortedResultByPredictedScore, RESULTPATH+ "/eval.txt", RESULTPATH+"/result_Id_label.csv");
		
		for(int k=2;k<=20;k=k+2)
		{
			topK = k;
			mg.saveEvaluationTopK(RESULTPATH+"/evalTopK_"+Integer.toString(topK)+".csv", sortedResultByPredictedScore, topK, cut_off);
			mg.saveNDCGTopK(RESULTPATH+"/NDCGTopK"+Integer.toString(topK)+".csv", sortedResultByPredictedScore, sortedResultByActualScore, topK, cut_off);
		}
		// ----//
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
		String RESULTPATH = "C:/Users/ADMIN/Desktop/Demo/result_tfidf/04_09_2018/dataset1/Test_full_matrix/ne";
		int topK = 0;
		double cut_off = 0;
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

		mg.convertScoreToLabelWithId(RESULTPATH+"/result_Id_score.csv", RESULTPATH+"/result_Id_label.csv", cut_off);

		// Sort result
		HashMap<String, List<Instance>> sortedResultByPredictedScore = mg
				.sortResultByAttribute(RESULTPATH + "/result_Id_score.csv", 2);
		mg.saveSortedResult(RESULTPATH+"/result_Id_score_sorted.csv", sortedResultByPredictedScore);
		HashMap<String, List<Instance>> sortedResultByActualScore = mg
				.sortResultByAttribute(RESULTPATH + "/result_Id_score.csv", 3);

		mg.saveEvaluationNoTime(sortedResultByPredictedScore, RESULTPATH+ "/eval.txt", RESULTPATH+"/result_Id_label.csv");
		
		for(int k=2;k<=20;k=k+2)
		{
			topK = k;
			mg.saveEvaluationTopK(RESULTPATH+"/evalTopK_"+Integer.toString(topK)+".csv", sortedResultByPredictedScore, topK, cut_off);
			mg.saveNDCGTopK(RESULTPATH+"/NDCGTopK"+Integer.toString(topK)+".csv", sortedResultByPredictedScore, sortedResultByActualScore, topK, cut_off);
		}
		
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
				.sortResultByAttribute("test_case/inputNDCG.csv", 2);
		HashMap<String, List<Instance>> sortedResultByActualScore = mg
				.sortResultByAttribute("test_case/inputNDCG.csv", 3);
		List<String[]> listResult = mg.calculateNDCGTopK(sortedResultByPredictedScore, sortedResultByActualScore, 10,
				0);
		for (int i = 0; i < listResult.size(); i++) {
			System.out.println(listResult.get(i)[0] + "-->" + listResult.get(i)[1]);
		}
		//Output: 1-->0.9608081943360616
	}

	/**
	 * JUST FOR TESTING CODE
	 */
	public static void testMRR() {
		// Test MRR
		ModelGenerator mg = new ModelGenerator();
		HashMap<String, List<Instance>> sortedResult = mg.sortResultByAttribute("test_case/inputMRR.csv",2);
		System.out.println(mg.calculateMRR(sortedResult));
		//Output: 0.611111111111111
	}
	
	/**
	 * JUST FOR TESTING CODE
	 */
	public static void testRMSE() {
		// Test RMSE
		ModelGenerator mg = new ModelGenerator();
		HashMap<String, List<Instance>> sortedResult = mg.sortResultByAttribute("test_case/inputMRR.csv",2);
		System.out.println(mg.calculateRMSE(sortedResult));
		//Output: 1.9148542155126762
	}

	public static void main(String[] args) throws IOException {
		System.out.println("(/・・)ノ");
		evalL2R();
		System.out.println("(*•̀ᴗ•́*)و ̑̑");
	}

}
