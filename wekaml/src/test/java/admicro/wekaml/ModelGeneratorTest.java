package admicro.wekaml;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;

import junit.framework.TestCase;
import weka.core.Instance;

/**
 * Unit Test: ModelGenerator class
 * @author Hua Tri Dung
 *
 */
public class ModelGeneratorTest extends TestCase {

	public void testCalculateMRR() {		
		ModelGenerator mg = new ModelGenerator();
		HashMap<String, List<Instance>> sortedResult = mg.sortResultByAttribute("test_case/inputMRR.csv", 2);		
		double actual = mg.calculateMRR(sortedResult);		
		double expected = 0.611111111111111;
		assertEquals(expected, actual);		
	}

	public void testCalculateRMSE() {
		ModelGenerator mg = new ModelGenerator();
		HashMap<String, List<Instance>> sortedResult = mg.sortResultByAttribute("test_case/inputMRR.csv", 2);
		double actual = mg.calculateRMSE(sortedResult);
		double expected = 1.9148542155126762;
		assertEquals(expected, actual);
	}

	public void testCalculateNDCGTopK() throws IOException {
		ModelGenerator mg = new ModelGenerator();
		HashMap<String, List<Instance>> sortedResultByPredictedScore = mg
				.sortResultByAttribute("test_case/inputNDCG.csv", 2);
		HashMap<String, List<Instance>> sortedResultByActualScore = mg
				.sortResultByAttribute("test_case/inputNDCG.csv", 3);
		List<String[]> listResult = mg.calculateNDCGTopK(sortedResultByPredictedScore, sortedResultByActualScore, 10,0);
		String actualId = listResult.get(0)[0];
		String actialValue = listResult.get(0)[1];
		String expectedId = "1";
		String expectedValue = "0.9608081943360616";
		assertEquals(expectedId, actualId);
		assertEquals(expectedValue, actialValue);		
	}

}
