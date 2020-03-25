import unittest
import numpy as np
from feature_selection import GeneticFeatureSelection

class TestGeneticFeatureSelection(unittest.TestCase):

  def test_fit_transform(self):

    # synthetic data
    n_samples, n_dims = 1000, 100
    input_array = np.random.sample(size=(n_samples, n_dims))
    activity_labels = np.random.choice(2, size=n_samples, p=(0.95, 0.05))

    # Random Forest Method: usage example
    model_location = 'example_model'

    feature_selector = GeneticFeatureSelection()
    feature_selector.fit(input_array, activity_labels, num_gens=1)
    feature_selector.save('example_model')

    feature_selector2 = GeneticFeatureSelection()
    feature_selector2.load(model_location)
    reduced_features = feature_selector2.transform(input_array)
    print(reduced_features)

    # basic assertions on shape and number of results
    results_shape = reduced_features.shape
    self.assertTrue(len(results_shape) is 2)
    self.assertEquals(results_shape[0], 1000)
    self.assertTrue(results_shape[1] > 1)



if __name__ == "__main__":
  unittest.main()