package admicro.wekaml;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

import weka.classifiers.Classifier;
import weka.classifiers.trees.REPTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;

public class ModelClassifier {

	private ArrayList<Attribute> attributes;
	private Instances dataRaw;

	public ModelClassifier() {
		Attribute newsId1 = new Attribute("newsId1");
		Attribute newsId2 = new Attribute("newsId2");
		Attribute keyword = new Attribute("keyword");
		Attribute cosineTF = new Attribute("cosineTF");
		Attribute jaccardBody = new Attribute("jaccardBody");
		Attribute jaccardTitle = new Attribute("jaccardTitle");
		Attribute bm25 = new Attribute("bm25");
		Attribute lm = new Attribute("lm");
		Attribute ib = new Attribute("ib");
		Attribute avgSim = new Attribute("avgSim");
		Attribute sumOfMax = new Attribute("sumOfMax");
		Attribute maxSim = new Attribute("maxSim");
		Attribute minSim = new Attribute("minSim");
		Attribute jaccardSim = new Attribute("jaccardSim");
		Attribute timeSpan = new Attribute("timeSpan");
		Attribute LDASim = new Attribute("LDASim");
		Attribute score = new Attribute("score");

		attributes = new ArrayList<Attribute>();
		attributes.add(newsId1);
		attributes.add(newsId2);
		attributes.add(keyword);
		attributes.add(cosineTF);
		attributes.add(jaccardBody);
		attributes.add(jaccardTitle);
		attributes.add(bm25);
		attributes.add(lm);
		attributes.add(ib);
		attributes.add(avgSim);
		attributes.add(sumOfMax);
		attributes.add(maxSim);
		attributes.add(minSim);
		attributes.add(jaccardSim);
		attributes.add(timeSpan);
		attributes.add(LDASim);
		attributes.add(score);

		dataRaw = new Instances("TestInstances", attributes, 0);
		dataRaw.setClassIndex(dataRaw.numAttributes() - 1);
	}

	public Instances createInstance(double keyword, double cosineTF, double jaccardBody, double jaccardTitle,
			double bm25, double lm, double ib, double avgSim, double sumOfMax, double maxSim, double minSim,
			double jaccardSim, double timeSpan, double LDASim, double score) {
		dataRaw.clear();
		double[] instanceValue = new double[] { keyword, cosineTF, jaccardBody, jaccardTitle, bm25, lm, ib, avgSim,
				sumOfMax, maxSim, minSim, jaccardSim, timeSpan, LDASim, 0 };
		dataRaw.add(new DenseInstance(1.0, instanceValue));
		return dataRaw;
	}

	public double classifiy(Instances insts, String path) {
		double result = -1;
		Classifier cls = null;
		try {
			cls = (REPTree) SerializationHelper.read(path);
			result = cls.classifyInstance(insts.firstInstance());
		} catch (Exception ex) {
			Logger.getLogger(ModelClassifier.class.getName()).log(Level.SEVERE, null, ex);
		}
		return result;
	}

	public static void main(String[] args) {
		ModelClassifier cls = new ModelClassifier();
        double classname =cls.classifiy(cls.createInstance(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15), "C:/Users/ADMIN/workspace/wekaml/wekaml/model/M5P_model.bin");
        System.out.println("( ◞･౪･)" +classname);
		
	}

}
