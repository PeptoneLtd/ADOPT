import pickle
import unittest
import os
from adopt import MultiHead, ZScorePred
import numpy.testing as npt

# These are just non-regression tests w.r.t. ADOPT v0.4.2
# z-scores were computed using ADOPT 0.4.2 and serialized in .pickle files

SEQUENCE = "SLQDGVRQSRASDKQTLLPNDQLYQPLKDREDDQYSHLQGNQLRRN"
PROTID = "Protein 18890"
STRATEGY = "train_on_cleared_1325_test_on_117_residue_split"


def getScores(model_type, strategy):
    # Extract residue level representations
    multi_head = MultiHead(model_type)
    representation, tokens = multi_head.get_representation(SEQUENCE, PROTID)

    # Predict the Z score related to each residue in the sequence specified above
    z_score_pred = ZScorePred(strategy, model_type)
    predicted_z_scores = z_score_pred.get_z_score(representation)

    return predicted_z_scores

class TestAdopt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._test_cases = {}
        for filename in os.listdir("tests"):
            print(f"loading {filename}")
            with open(os.path.join("tests", filename), "rb") as f:
                cls._test_cases[filename.replace(".pickle", "")] = pickle.load(f)

    def test_esm1b(self):
        scores = getScores("esm-1b", STRATEGY)
        npt.assert_array_equal(scores, self._test_cases.get("esm-1b"))
    
    def test_esm1v(self):
        scores = getScores("esm-1v", STRATEGY)
        npt.assert_array_equal(scores, self._test_cases.get("esm-1v"))

    def test_combined(self):
        scores = getScores("combined", STRATEGY)
        npt.assert_array_equal(scores, self._test_cases.get("combined"))

    def test_esm_msa(self):
        scores = getScores("esm-msa", STRATEGY)
        npt.assert_array_equal(scores, self._test_cases.get("esm-msa"))

    

if __name__ == '__main__':
    unittest.main()