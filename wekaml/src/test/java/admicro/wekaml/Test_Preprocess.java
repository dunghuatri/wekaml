package admicro.wekaml;

import weka.core.Instances;

public class Test_Preprocess {
	
	public static void Test_Numeric2Nominal() throws Exception
	{
		Preporcess prep = new Preporcess();
		String DATASETPATH = "data/heart.arff";
		ModelGenerator mg = new ModelGenerator();		
		Instances originalData = mg.loadDataset(DATASETPATH);
		Instances newData = prep.Numeric2Nominal(originalData,"last");
	}
	
	public static void Test_CSVToArff() throws Exception
	{
		Preporcess prep = new Preporcess();
		String csvFilePath = "data/heart.csv";
		String arffFilePath = "data/heart.arff";
		prep.CSVToArff(csvFilePath, arffFilePath);
	}

	public static void main(String[] args) throws Exception {		
//		Test_CSVToArff();	
		Test_Numeric2Nominal();
	}

}
