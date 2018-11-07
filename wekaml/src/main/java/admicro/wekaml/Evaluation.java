package admicro.wekaml;

import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class Evaluation {

	public static String fileName = "evalSumary.csv";
	
	public static void main(String[] args) {
		ArrayList<String> inPath = new ArrayList<>();
//		inPath.add("C:/Users/ADMIN/Desktop/Demo/auto_fill_data/24_10_2018/all_features");
//		inPath.add("C:/Users/ADMIN/Desktop/Demo/auto_fill_data/24_10_2018/all_features/Positive");
//		inPath.add("C:/Users/ADMIN/Desktop/Demo/auto_fill_data/24_10_2018/leave_one_feature_out");
		inPath.add("C:/Users/ADMIN/Desktop/Demo/auto_fill_data/24_10_2018/leave_one_feature_out/Positive");

		evaluation(inPath, fileName, false);
	}

	private static void evaluation(List<String> inputPath, String outputPath, boolean isAllFile) {
		String[] types = { "_event.csv", "_topic.csv", "_ne.csv" };
		String[] datasets = { "dataset1", "dataset2", "dataset3" };
		String path = "";
		for (int i = 0; i < inputPath.size(); i++) {
			for (String set : datasets) {
				for (String type : types) {
					path = inputPath.get(i) + "/" + set + "/" + set + type;
					if (isAllFile) {
						readFileAll(path);
					} else {
						readFileLeaveOneFeaturesOut(path);
					}

				}
			}
		}

		try (Writer writer = Files.newBufferedWriter(Paths.get(outputPath));
				CSVWriter csvWriter = new CSVWriter(writer, CSVWriter.DEFAULT_SEPARATOR,
						CSVWriter.DEFAULT_QUOTE_CHARACTER, CSVWriter.DEFAULT_ESCAPE_CHARACTER,
						CSVWriter.DEFAULT_LINE_END);) {
			// header: keyword; cosineTF; jaccardBody; jaccardTitle;
			// bm25; lm; ib; avgSim;
			// sumOfMax;maxSim;minSim;jaccardSim;timeSpan;LDASim;
			String[] header = { "newsId1", "newsId2", "predict", "actual" };
			csvWriter.writeNext(header);

			csvWriter.flush();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static HashMap<String, List<Double>> readFileAll(String inputPath) {
		String[] tops = { "1", "3", "4", "5", "10" };
		HashMap<String, List<Double>> algList = new HashMap<>();
		try (Reader reader = new BufferedReader(new InputStreamReader(new FileInputStream(inputPath), "utf-8"));
				CSVReader csvReader = new CSVReader(reader, ',', '"', 2)) {
			String[] nextRecord;
			while ((nextRecord = csvReader.readNext()) != null) {
				int topIndex = 0;
				for (int i = 0; i <= nextRecord.length - 7; i += 8) {
					try {
						List<Double> evalScore = new ArrayList<>();
						evalScore.addAll(Arrays.asList(Double.parseDouble(nextRecord[i + 1]),
								Double.parseDouble(nextRecord[i + 2]), Double.parseDouble(nextRecord[i + 3]),
								Double.parseDouble(nextRecord[i + 4]), Double.parseDouble(nextRecord[i + 5]),
								Double.parseDouble(nextRecord[i + 6])));
						algList.put(String.valueOf(tops[topIndex] + nextRecord[i]), evalScore);
						topIndex++;
					} catch (NumberFormatException e) {
						e.printStackTrace();
						System.out.println("File " + inputPath + " " + nextRecord[i]);
					}
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("File: " + inputPath);
		compareAlg(algList);
		return algList;
	}

	private static HashMap<String, List<Double>> readFileLeaveOneFeaturesOut(String inputPath) {
		String[] tops = { "1", "3", "4", "5", "10" };
		HashMap<String, List<Double>> algList = new HashMap<>();
		System.out.println("File: " + inputPath);
		try (Reader reader = new BufferedReader(new InputStreamReader(new FileInputStream(inputPath), "utf-8"));
				CSVReader csvReader = new CSVReader(reader, ',', '"', 0)) {
			String[] nextRecord;
			while ((nextRecord = csvReader.readNext()) != null) {
				if (nextRecord[0].contains("_dataset")) {
					System.out.println(nextRecord[0]);
				} else if (nextRecord[0].equals("")) {
					if (algList.size() != 0) {
						compareAlg(algList);
						algList = new HashMap<>();
					}
				} else if (!nextRecord[0].equals("Thuật toán")) {
					int topIndex = 0;
					for (int i = 0; i <= nextRecord.length - 7; i += 8) {
						try {
							List<Double> evalScore = new ArrayList<>();
							evalScore.addAll(Arrays.asList(Double.parseDouble(nextRecord[i + 1]),
									Double.parseDouble(nextRecord[i + 2]), Double.parseDouble(nextRecord[i + 3]),
									Double.parseDouble(nextRecord[i + 4]), Double.parseDouble(nextRecord[i + 5]),
									Double.parseDouble(nextRecord[i + 6])));
							algList.put(String.valueOf(tops[topIndex] + nextRecord[i]), evalScore);
							topIndex++;
						} catch (NumberFormatException e) {
							e.printStackTrace();
							System.out.println("File " + inputPath + " " + nextRecord[i]);
						}
					}

				}

			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return algList;
	}

	private static void compareAlg(HashMap<String, List<Double>> map) {
		String[] algs = new String[] { "REPTree", "M5P", "GBRT", "TFIDF" };
		String[] tops = { "1", "3", "4", "5", "10" };
		String[] measures = { "Precision", "Recall", "F1", "NDCG", "MRR", "RMSE" };
		HashMap<String, HashMap<String, Integer>> resultMap = new HashMap<>();
		HashMap<String, List<String>> finalResult = new HashMap<>();
		for (String top : tops) {
			for (int i = 0; i < 6; i++) { // độ đo
				List<String> algName = new ArrayList<>();
				if (i != 5) {
					Double maxNum = 0.0;
					for (String alg : algs) {
						if (maxNum <= map.get(top + alg).get(i)) {
							maxNum = map.get(top + alg).get(i);
						}
					}
					for (String alg : algs) {
						if (maxNum.compareTo(map.get(top + alg).get(i)) == 0) {
							algName.add(alg);
						}
					}
				} else {
					Double minNum = Double.MAX_VALUE;
					for (String alg : algs) {
						if (minNum >= map.get(top + alg).get(i)) {
							minNum = map.get(top + alg).get(i);
						}
					}
					for (String alg : algs) {
						if (minNum.compareTo(map.get(top + alg).get(i)) == 0) {
							algName.add(alg);
						}
					}
				}
				HashMap<String, Integer> result = new HashMap<>();
				for (String name : algName) {
					if (resultMap.containsKey(name)) {
						if (resultMap.get(name).containsKey(measures[i])) {
							resultMap.get(name).put(measures[i], resultMap.get(name).get(measures[i]) + 1);
						} else {
							resultMap.get(name).put(measures[i], 1);
						}
					} else {
						result.put(measures[i], 1);
						resultMap.put(name, result);
					}
				}

			}
		}
		for (String measure : measures) {
			int maximum = 0;
			List<String> algName = new ArrayList<>();
			for (String alg : resultMap.keySet()) {
				if (resultMap.containsKey(alg)) {
					if (resultMap.get(alg).containsKey(measure)) {
						if (maximum <= resultMap.get(alg).get(measure)) {
							maximum = resultMap.get(alg).get(measure);
							algName.add(alg);
						}
					}
				}

			}
			for (String name : algName) {
				if (finalResult.containsKey(name) && resultMap.get(name).get(measure) == maximum) {
					finalResult.get(name).add(measure);
				} else if (resultMap.get(name).get(measure) == maximum) {
					List<String> tmp = new ArrayList<>();
					tmp.add(measure);
					finalResult.put(name, tmp);
				}
			}
		}
		for (String key : finalResult.keySet()) {
			System.out.print("Thuật toán " + key);
			boolean flag = false;
			for (String value : finalResult.get(key)) {
				if (!value.equals("RMSE")) {
					if (!flag) {
						System.out.print(" cao nhất trên độ đo ");
						flag = true;
					}
					if (finalResult.get(key).size() == 1) {
						System.out.println(value + " ");
					} else {
						System.out.print(value + " , ");
					}
				} else {
					System.out.println(" nhỏ nhất trên độ đo " + value);
				}
			}
		}
		System.out.println("");
	}
}