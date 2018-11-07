package admicro.wekaml;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class RemoveFile {
	public static String inputPath = "C:/Users/ADMIN/Desktop/Demo/auto_fill_data/24_10_2018";
	public static String fileName = "evalSumary.csv";

	public static void main(String[] args) {
		cleanDirectories(inputPath, fileName);
	}

	private static void cleanDirectories(String inputPath, String fileName) {
		HashMap<String, Integer> allPathList = getPathList(inputPath);
		HashMap<String, Integer> pathList = new HashMap<>();
		File file = null;
		int maxNum1 = 0, maxNum2 = 0;
		for (Map.Entry<String, Integer> entry : allPathList.entrySet()) {
			if (entry.getKey().contains("all_features") && entry.getKey().contains("dataset")) {
				pathList.put(entry.getKey(), entry.getValue());
				if (maxNum1 < entry.getValue()) {
					maxNum1 = entry.getValue();
				}
			} else if (entry.getKey().contains("leave_one_feature_out") && entry.getKey().contains("_dataset")) {
				pathList.put(entry.getKey(), entry.getValue());
				if (maxNum2 < entry.getValue()) {
					maxNum2 = entry.getValue();
				}
			}
		}
		for (Map.Entry<String, Integer> entry : pathList.entrySet()) {
			if (!entry.getKey().contains(fileName)) {
				if (!entry.getKey().contains("Positive")) {
					if (entry.getValue() == maxNum1 - 1 || entry.getValue() == maxNum2 - 1) {
						System.out.println(entry.getKey());
						file = new File(entry.getKey());
						file.delete();
					}
				} else {
					if (entry.getValue() == maxNum1 || entry.getValue() == maxNum2) {
						System.out.println(entry.getKey());
						file = new File(entry.getKey());
						file.delete();
					}
				}
			}
		}

		System.out.println("Done");
	}

	private static HashMap<String, Integer> getPathList(String folderPath) {
		HashMap<String, Integer> paths = new HashMap<>();
		List<String> tmp = new ArrayList<>();
		getPath(folderPath, tmp, paths);
		for (int i = 0; i < tmp.size(); i++) {
			getPath(tmp.get(i), tmp, paths);
		}
		return paths;
	}

	private static void getPath(String folderPath, List<String> tmp, HashMap<String, Integer> paths) {
		File[] files = new File(folderPath).listFiles();
		int level = 0;
		for (File file : files) {
			if (!file.isFile()) {
				tmp.add(file.getAbsolutePath());
			} else {
				// Ex: rootname = "16_10_2018"
				String rootName = inputPath.substring(inputPath.lastIndexOf("/") + 1);
				String path = file.getAbsolutePath().substring(file.getAbsolutePath().indexOf(rootName));
				level = path.length() - path.replace("\\", "").length();
				paths.put(file.getAbsolutePath(), level - 1);
			}
		}
	}
}