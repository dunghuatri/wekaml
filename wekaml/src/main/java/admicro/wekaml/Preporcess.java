package admicro.wekaml;

import java.io.File;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;

public class Preporcess {
	
	/**
	 * Convert numeric attributes to nominal attributes
	 * @param originalData
	 * @param range ("last" or "first-last" or "1-2")
	 * @return
	 * @throws Exception
	 */
	public Instances Numeric2Nominal(Instances originalData, String range) throws Exception
	{
        NumericToNominal convert= new NumericToNominal();
        String[] options= new String[2];
        options[0]="-R";
        options[1]=range;  //range of variables to make numeric

        convert.setOptions(options);
        convert.setInputFormat(originalData);

        Instances newData=Filter.useFilter(originalData, convert);     
        
        /*int n = originalData.numAttributes();
        System.out.println("-----Before-----");
        for(int i=0; i<n; i=i+1)
        {
            System.out.println("Attribute "+(i+1)+"\tNominal?\t"+originalData.attribute(i).isNominal());
        }

        System.out.println("-----After-----");
        for(int i=0; i<n; i=i+1)
        {
            System.out.println("Attribute "+(i+1)+"\tNominal?\t"+newData.attribute(i).name());
        }        */
        return newData;        
	}
	
	public Instances removeFeatures(Instances originalData, String range) throws Exception
	{
		Remove remove = new Remove();
		
		String[] options= new String[2];
        options[0]="-R";
        options[1]=range;
        
        remove.setOptions(options);
        remove.setInputFormat(originalData);
        
        Instances newData=Filter.useFilter(originalData, remove);
        return newData;
	}
	
	public void CSVToArff(String csvFilePath, String arffFilePath) throws IOException
	{
		 // load CSV
	    CSVLoader loader = new CSVLoader();
	    loader.setSource(new File(csvFilePath));
	    Instances data = loader.getDataSet();

	    // save ARFF
	    ArffSaver saver = new ArffSaver();
	    saver.setInstances(data);
	    saver.setFile(new File(arffFilePath));
	    saver.writeBatch();
	    // .arff file will be created in the output location
	}

	public static void main(String[] args) throws Exception {		
		
	}

}
