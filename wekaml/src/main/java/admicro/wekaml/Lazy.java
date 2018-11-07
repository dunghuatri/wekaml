package admicro.wekaml;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

public class Lazy {

	public List<List<String>> readData(String dataPath) throws IOException {
		List<List<String>> resultTopK = new ArrayList<>();
		try (Reader reader = Files.newBufferedReader(Paths.get(dataPath));
				CSVReader csvReader = new CSVReader(reader);) {

			String[] nextRecord = csvReader.readNext();// header
			while ((nextRecord = csvReader.readNext()) != null) {
				List<String> result = new ArrayList<>();
				result.add(nextRecord[1]);
				result.add(nextRecord[2]);
				result.add(nextRecord[3]);
				result.add(nextRecord[4]);
				result.add(nextRecord[5]);
				result.add(nextRecord[6]);
				resultTopK.add(result);
			}
		}
		return resultTopK;
	}
	
	/**
	 * So sanh 1 thuat toan
	 * 
	 * @param resultPath
	 * @param listTailName
	 * @param reptreeResult
	 * @param m5pResult
	 * @param gbrtResult
	 * @param tfidfResult
	 * @throws IOException
	 */
	public void writeData(String resultPath, List<String> listTailName,
			HashMap<String, List<List<String>>> svmResult)
			throws IOException {
		File file = new File(resultPath);
		file.getParentFile().mkdirs();
		try (Writer writer = Files.newBufferedWriter(Paths.get(resultPath));

				CSVWriter csvWriter = new CSVWriter(writer, CSVWriter.DEFAULT_SEPARATOR, CSVWriter.NO_QUOTE_CHARACTER,
						CSVWriter.DEFAULT_ESCAPE_CHARACTER, CSVWriter.DEFAULT_LINE_END);) {
			List<String> topK = new ArrayList<>();
			topK.add("1");
			topK.add("3");
			topK.add("4");
			topK.add("5");
			topK.add("10");

			for (String key : listTailName) {
				List<List<String>> svmTopK = svmResult.get(key);

				List<String> recordHeader1 = new ArrayList<>();
				List<String> recordHeader2 = new ArrayList<>();
				List<String> recordSvm = new ArrayList<>();

				for (int i = 0; i < topK.size(); i++) {
					recordHeader1.add(key);
					recordHeader1.add("Độ đo");
					for (int n = 0; n < 6; n++)
						recordHeader1.add("");

					recordHeader2.add("Thuật toán");
					recordHeader2.add("Precision@" + topK.get(i));
					recordHeader2.add("Recall@" + topK.get(i));
					recordHeader2.add("F1@" + topK.get(i));
					recordHeader2.add("NDCG@" + topK.get(i));
					recordHeader2.add("MRR");
					recordHeader2.add("RMSE");
					recordHeader2.add("");

					List<String> svmData = svmTopK.get(i);

					recordSvm.add("SVM");					
					
					for (int j = 0; j < svmData.size(); j++) {
						recordSvm.add(svmData.get(j));
					}

					recordSvm.add("");
				}
				csvWriter.writeNext(recordHeader1.toArray(new String[0]));
				csvWriter.writeNext(recordHeader2.toArray(new String[0]));
				csvWriter.writeNext(recordSvm.toArray(new String[0]));
				csvWriter.writeNext(new String[] { "" });
			}

		}
	}
	
	/**
	 * So sanh 2 thuat toan
	 * 
	 * @param resultPath
	 * @param listTailName
	 * @param reptreeResult
	 * @param m5pResult
	 * @param gbrtResult
	 * @param tfidfResult
	 * @throws IOException
	 */
	public void writeData(String resultPath, List<String> listTailName,
			HashMap<String, List<List<String>>> svmResult, HashMap<String, List<List<String>>> tfidfResult)
			throws IOException {
		File file = new File(resultPath);
		file.getParentFile().mkdirs();
		try (Writer writer = Files.newBufferedWriter(Paths.get(resultPath));

				CSVWriter csvWriter = new CSVWriter(writer, CSVWriter.DEFAULT_SEPARATOR, CSVWriter.NO_QUOTE_CHARACTER,
						CSVWriter.DEFAULT_ESCAPE_CHARACTER, CSVWriter.DEFAULT_LINE_END);) {
			List<String> topK = new ArrayList<>();
			topK.add("1");
			topK.add("3");
			topK.add("4");
			topK.add("5");
			topK.add("10");

			for (String key : listTailName) {
				List<List<String>> svmTopK = svmResult.get(key);				
				List<List<String>> tfidfTopK = tfidfResult.get(key);

				List<String> recordHeader1 = new ArrayList<>();
				List<String> recordHeader2 = new ArrayList<>();
				List<String> recordSvm = new ArrayList<>();
				List<String> recordTfidf = new ArrayList<>();

				for (int i = 0; i < topK.size(); i++) {
					recordHeader1.add(key);
					recordHeader1.add("Độ đo");
					for (int n = 0; n < 6; n++)
						recordHeader1.add("");

					recordHeader2.add("Thuật toán");
					recordHeader2.add("Precision@" + topK.get(i));
					recordHeader2.add("Recall@" + topK.get(i));
					recordHeader2.add("F1@" + topK.get(i));
					recordHeader2.add("NDCG@" + topK.get(i));
					recordHeader2.add("MRR");
					recordHeader2.add("RMSE");
					recordHeader2.add("");

					List<String> svmData = svmTopK.get(i);					
					List<String> tfidfData = tfidfTopK.get(i);

					recordSvm.add("SVM");					
					recordTfidf.add("TFIDF");

					for (int j = 0; j < svmData.size(); j++) {
						recordSvm.add(svmData.get(j));						
						recordTfidf.add(tfidfData.get(j));
					}

					recordSvm.add("");					
					recordTfidf.add("");
				}
				csvWriter.writeNext(recordHeader1.toArray(new String[0]));
				csvWriter.writeNext(recordHeader2.toArray(new String[0]));
				csvWriter.writeNext(recordSvm.toArray(new String[0]));				
				csvWriter.writeNext(recordTfidf.toArray(new String[0]));
				csvWriter.writeNext(new String[] { "" });
			}

		}
	}

	/**
	 * So sanh theo thuat toan
	 * 
	 * @param resultPath
	 * @param listTailName
	 * @param reptreeResult
	 * @param m5pResult
	 * @param gbrtResult
	 * @param tfidfResult
	 * @throws IOException
	 */
	public void writeData(String resultPath, List<String> listTailName,
			HashMap<String, List<List<String>>> reptreeResult, HashMap<String, List<List<String>>> m5pResult,
			HashMap<String, List<List<String>>> gbrtResult, HashMap<String, List<List<String>>> tfidfResult)
			throws IOException {
		File file = new File(resultPath);
		file.getParentFile().mkdirs();
		try (Writer writer = Files.newBufferedWriter(Paths.get(resultPath));

				CSVWriter csvWriter = new CSVWriter(writer, CSVWriter.DEFAULT_SEPARATOR, CSVWriter.NO_QUOTE_CHARACTER,
						CSVWriter.DEFAULT_ESCAPE_CHARACTER, CSVWriter.DEFAULT_LINE_END);) {
			List<String> topK = new ArrayList<>();
			topK.add("1");
			topK.add("3");
			topK.add("4");
			topK.add("5");
			topK.add("10");

			for (String key : listTailName) {
				List<List<String>> reptreeTopK = reptreeResult.get(key);
				List<List<String>> m5pTopK = m5pResult.get(key);
				List<List<String>> gbrtTopK = gbrtResult.get(key);
				List<List<String>> tfidfTopK = tfidfResult.get(key);

				List<String> recordHeader1 = new ArrayList<>();
				List<String> recordHeader2 = new ArrayList<>();
				List<String> recordReptree = new ArrayList<>();
				List<String> recordM5p = new ArrayList<>();
				List<String> recordGbrt = new ArrayList<>();
				List<String> recordTfidf = new ArrayList<>();

				for (int i = 0; i < topK.size(); i++) {
					recordHeader1.add(key);
					recordHeader1.add("Độ đo");
					for (int n = 0; n < 6; n++)
						recordHeader1.add("");

					recordHeader2.add("Thuật toán");
					recordHeader2.add("Precision@" + topK.get(i));
					recordHeader2.add("Recall@" + topK.get(i));
					recordHeader2.add("F1@" + topK.get(i));
					recordHeader2.add("NDCG@" + topK.get(i));
					recordHeader2.add("MRR");
					recordHeader2.add("RMSE");
					recordHeader2.add("");

					List<String> reptreeData = reptreeTopK.get(i);
					List<String> m5pData = m5pTopK.get(i);
					List<String> gbrtData = gbrtTopK.get(i);
					List<String> tfidfData = tfidfTopK.get(i);

					recordReptree.add("REPTree");
					recordM5p.add("M5P");
					recordGbrt.add("GBRT");
					recordTfidf.add("TFIDF");

					for (int j = 0; j < reptreeData.size(); j++) {
						recordReptree.add(reptreeData.get(j));
						recordM5p.add(m5pData.get(j));
						recordGbrt.add(gbrtData.get(j));
						recordTfidf.add(tfidfData.get(j));
					}

					recordReptree.add("");
					recordM5p.add("");
					recordGbrt.add("");
					recordTfidf.add("");
				}
				csvWriter.writeNext(recordHeader1.toArray(new String[0]));
				csvWriter.writeNext(recordHeader2.toArray(new String[0]));
				csvWriter.writeNext(recordReptree.toArray(new String[0]));
				csvWriter.writeNext(recordM5p.toArray(new String[0]));
				csvWriter.writeNext(recordGbrt.toArray(new String[0]));
				csvWriter.writeNext(recordTfidf.toArray(new String[0]));
				csvWriter.writeNext(new String[] { "" });
			}

		}
	}

	/**
	 * So sanh theo feature
	 * 
	 * @param resultPath
	 * @param listTailName
	 * @param listAlgorithm
	 * @throws IOException
	 */
	public void writeData(String resultPath, List<String> listTailName,
			List<HashMap<String, List<List<String>>>> listAlgorithm) throws IOException {
		File file = new File(resultPath);
		file.getParentFile().mkdirs();
		try (Writer writer = Files.newBufferedWriter(Paths.get(resultPath));

				CSVWriter csvWriter = new CSVWriter(writer, CSVWriter.DEFAULT_SEPARATOR, CSVWriter.NO_QUOTE_CHARACTER,
						CSVWriter.DEFAULT_ESCAPE_CHARACTER, CSVWriter.DEFAULT_LINE_END);) {
			List<String> topK = new ArrayList<>();
			topK.add("1");
			topK.add("3");
			topK.add("4");
			topK.add("5");
			topK.add("10");

			for (int algoIndex = 0; algoIndex < listAlgorithm.size(); algoIndex++) {
				List<String> recordHeader1 = new ArrayList<>();
				List<String> recordHeader2 = new ArrayList<>();
				boolean flagHeader = false;

				HashMap<String, List<List<String>>> algoResult = listAlgorithm.get(algoIndex);
				for (String key : listTailName) {
					List<List<String>> algoTopK = algoResult.get(key);
					List<String> recordAlgo = new ArrayList<>();

					for (int i = 0; i < topK.size(); i++) {
						switch (algoIndex) {
						case 0:
							recordHeader1.add("REPTree");
							break;
						case 1:
							recordHeader1.add("M5P");
							break;
						case 2:
							recordHeader1.add("GBRT");
							break;
						case 3:
							recordHeader1.add("TFIDF");
							break;
						}
						recordHeader1.add("Độ đo");
						for (int n = 0; n < 6; n++)
							recordHeader1.add("");

						recordHeader2.add("No Feature");
						recordHeader2.add("Precision@" + topK.get(i));
						recordHeader2.add("Recall@" + topK.get(i));
						recordHeader2.add("F1@" + topK.get(i));
						recordHeader2.add("NDCG@" + topK.get(i));
						recordHeader2.add("MRR");
						recordHeader2.add("RMSE");
						recordHeader2.add("");

						List<String> algoData = algoTopK.get(i);
						recordAlgo.add(key);

						for (int j = 0; j < algoData.size(); j++) {
							recordAlgo.add(algoData.get(j));
						}
						recordAlgo.add("");
					}
					if (flagHeader == false) {
						csvWriter.writeNext(recordHeader1.toArray(new String[0]));
						csvWriter.writeNext(recordHeader2.toArray(new String[0]));
						flagHeader = true;
					}
					csvWriter.writeNext(recordAlgo.toArray(new String[0]));
				}
				csvWriter.writeNext(new String[] { "" });
			}

		}
	}

	public void mapData() throws IOException {
		List<String> listDatasetName = new ArrayList<>();
		List<String> listCriteria = new ArrayList<>();
		List<String> listFeature = new ArrayList<>();
		List<String> listTailName = new ArrayList<>();

		// Add dataset name
		listDatasetName.add("dataset1");
//		listDatasetName.add("dataset2");
//		listDatasetName.add("dataset3");

		// Add criteria name
		listCriteria.add("event");
		listCriteria.add("topic");
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

		HashMap<String, List<List<String>>> svmResult = new HashMap<>();
//		HashMap<String, List<List<String>>> reptreeResult = new HashMap<>();
//		HashMap<String, List<List<String>>> m5pResult = new HashMap<>();
//		HashMap<String, List<List<String>>> gbrtResult = new HashMap<>();
//		HashMap<String, List<List<String>>> tfidfResult = new HashMap<>();

		for (String dataset : listDatasetName)
			for (String criteria : listCriteria) {
				String svmPath = "C:/Users/ADMIN/Desktop/Demo/auto_fill_data/24_10_2018/leave_one_feature_out/Positive/"
						+ dataset + "/" + criteria + "/SVM/";
//				String reptreePath = "C:/Users/ADMIN/Desktop/Demo/auto_fill_data/24_10_2018/leave_one_feature_out/Positive/"
//						+ dataset + "/" + criteria + "/REPTree/";
//				String m5pPath = "C:/Users/ADMIN/Desktop/Demo/auto_fill_data/24_10_2018/leave_one_feature_out/Positive/"
//						+ dataset + "/" + criteria + "/M5P/";
//				String gbrtPath = "C:/Users/ADMIN/Desktop/Demo/auto_fill_data/24_10_2018/leave_one_feature_out/Positive/"
//						+ dataset + "/" + criteria + "/GBRT/";
//				String tfidfPath = "C:/Users/ADMIN/Desktop/Demo/auto_fill_data/24_10_2018/leave_one_feature_out/Positive/"
//						+ dataset + "/" + criteria + "/TFIDF/";
				for (String feature : listFeature) {
					String name = "_" + dataset + "_" + criteria + "_" + feature;
					listTailName.add(name);
					
					List<List<String>> listSvm = readData(svmPath + name + "/evalSumary.csv");
//					List<List<String>> listReptree = readData(reptreePath + name + "/evalSumary.csv");
//					List<List<String>> listM5p = readData(m5pPath + name + "/evalSumary.csv");
//					List<List<String>> listGbrt = readData(gbrtPath + name + "/evalSumary.csv");
//					List<List<String>> listTfidf = readData(tfidfPath + name + "/evalSumary.csv");
					
					svmResult.put(name, listSvm);
//					reptreeResult.put(name, listReptree);
//					m5pResult.put(name, listM5p);
//					gbrtResult.put(name, listGbrt);
//					tfidfResult.put(name, listTfidf);
				}
				Collections.sort(listTailName);
				List<HashMap<String, List<List<String>>>> listAlgorithm = new ArrayList<>();
				listAlgorithm.add(svmResult);
//				listAlgorithm.add(reptreeResult);
//				listAlgorithm.add(m5pResult);
//				listAlgorithm.add(gbrtResult);
//				listAlgorithm.add(tfidfResult);
				// So sanh theo thuat toan
//				writeData(
//						"C:/Users/ADMIN/Desktop/Demo/auto_fill_data/24_10_2018/leave_one_feature_out/Positive/"
//								+ dataset + "/" + dataset + "_" + criteria + ".csv",
//						listTailName, reptreeResult, m5pResult, gbrtResult, tfidfResult);
				// So sanh theo feature
				writeData(
						"C:/Users/ADMIN/Desktop/Demo/auto_fill_data/24_10_2018/leave_one_feature_out/Positive/"
								+ dataset + "/" + dataset + "_feature_" + criteria + ".csv",
						listTailName, listAlgorithm);
				
				listTailName.clear();
			}
	}

	public void mapDataAllFeature() throws IOException {
		List<String> listDatasetName = new ArrayList<>();
		List<String> listCriteria = new ArrayList<>();
		List<String> listTailName = new ArrayList<>();

		// Add dataset name
		listDatasetName.add("dataset1_2");
//		listDatasetName.add("dataset2");
//		listDatasetName.add("dataset3");

		// Add criteria name
		listCriteria.add("event");
		listCriteria.add("topic");
		listCriteria.add("ne");

		HashMap<String, List<List<String>>> svmResult = new HashMap<>();
//		HashMap<String, List<List<String>>> reptreeResult = new HashMap<>();
//		HashMap<String, List<List<String>>> m5pResult = new HashMap<>();
//		HashMap<String, List<List<String>>> gbrtResult = new HashMap<>();
		HashMap<String, List<List<String>>> tfidfResult = new HashMap<>();		
		
		for(String dataset:listDatasetName)
			for(String criteria:listCriteria)
			{
				String svmPath = "C:/Users/ADMIN/Desktop/Demo/auto_fill_data/02_11_2018/all_features/"
						+ dataset + "/" + criteria + "/SVM/";
//				String reptreePath = "C:/Users/ADMIN/Desktop/Demo/auto_fill_data/24_10_2018/all_features/Positive/"
//						+ dataset + "/" + criteria + "/REPTree/";
//				String m5pPath = "C:/Users/ADMIN/Desktop/Demo/auto_fill_data/24_10_2018/all_features/Positive/"
//						+ dataset + "/" + criteria + "/M5P/";
//				String gbrtPath = "C:/Users/ADMIN/Desktop/Demo/auto_fill_data/24_10_2018/all_features/Positive/"
//						+ dataset + "/" + criteria + "/GBRT/";
				String tfidfPath = "C:/Users/ADMIN/Desktop/Demo/auto_fill_data/02_11_2018/all_features/"
						+ dataset + "/" + criteria + "/TFIDF/";

				String name = criteria;
				listTailName.add(name);				
				
				List<List<String>> listSvm = readData(svmPath + "evalSumary.csv");
//				List<List<String>> listReptree = readData(reptreePath + "evalSumary.csv");
//				List<List<String>> listM5p = readData(m5pPath + "evalSumary.csv");
//				List<List<String>> listGbrt = readData(gbrtPath + "evalSumary.csv");
				List<List<String>> listTfidf = readData(tfidfPath + "evalSumary.csv");
				
				svmResult.put(name, listSvm);
//				reptreeResult.put(name, listReptree);
//				m5pResult.put(name, listM5p);
//				gbrtResult.put(name, listGbrt);
				tfidfResult.put(name, listTfidf);

				Collections.sort(listTailName);
				List<HashMap<String, List<List<String>>>> listAlgorithm = new ArrayList<>();
				listAlgorithm.add(svmResult);
//				listAlgorithm.add(reptreeResult);
//				listAlgorithm.add(m5pResult);
//				listAlgorithm.add(gbrtResult);
				listAlgorithm.add(tfidfResult);
				// So sanh theo thuat toan				
//				writeData(
//						"C:/Users/ADMIN/Desktop/Demo/auto_fill_data/24_10_2018/all_features/Positive/" + dataset
//								+ "/" + dataset + "_" + criteria + ".csv",
//						listTailName, reptreeResult, m5pResult, gbrtResult, tfidfResult);
				writeData(
						"C:/Users/ADMIN/Desktop/Demo/auto_fill_data/02_11_2018/all_features/" + dataset
								+ "/" + dataset + "_" + criteria + ".csv",
						listTailName, svmResult, tfidfResult);
				
				listTailName.clear();
			}

	}

	public static void main(String[] args) throws IOException {
		Lazy lazy = new Lazy();
		System.out.println("(/・・)ノ");
		lazy.mapDataAllFeature();
		System.out.println("＼(ﾟｰﾟ＼)");
	}

}
