import unittest
import tensorflow as tf
from tensorflow.keras import metrics
import numpy as np
import sys
sys.path.append('/home/mmorelan/proj/dementia/template/')
import custom_losses

class TestLosses(unittest.TestCase):

    def test_contrastive_loss(self):
        # Opposite class (0) with similar latent vectors (0) should yield loss = 1
        y_true = np.zeros((1))
        y_pred = np.zeros((1))
        self.assertEqual(custom_losses.contrastive_loss()(y_true, y_pred), 1)
        
        # Opposite class (0) with dissimilar latent vectors (1) should yield loss = 0
        y_true = np.zeros((1))
        y_pred = np.ones((1))
        self.assertEqual(custom_losses.contrastive_loss()(y_true, y_pred), 0)
        
        # Same class (1) with dissimilar latent vectors (1) should yield loss = 1
        y_true = np.ones((1))
        y_pred = np.ones((1))
        self.assertEqual(custom_losses.contrastive_loss()(y_true, y_pred), 1)
        
        # Same class (1) with similar latent vectors (0) should yield loss = 0
        y_true = np.ones((1))
        y_pred = np.zeros((1))
        self.assertEqual(custom_losses.contrastive_loss()(y_true, y_pred), 0)
        
    def test_cosine_similarity(self):
        # Cosine similarity implementation should yield same value as default TensorFlow implementation
        a = np.random.rand(1, 10)
        b = np.random.rand(1, 10)

        self.assertAlmostEqual(1 - metrics.CosineSimilarity()(a, b).numpy(), custom_losses.cosine_similarity([a, b]).numpy().squeeze(), places=3)
        
        # Random vectors should yield 0 < loss < 1
        a = np.random.rand(1, 10)
        b = np.random.rand(1, 10)
               
        self.assertGreater(custom_losses.cosine_similarity([a, b]).numpy().squeeze(), 0)
        self.assertLess(custom_losses.cosine_similarity([a, b]).numpy().squeeze(), 1)
        
        # Similar vectors should yield loss = 0
        a = np.ones((1, 10))
        b = np.ones((1, 10))
        self.assertAlmostEqual(custom_losses.cosine_similarity([a, b]).numpy().squeeze(), 0, places=3)
        
        # Dissimilar vectors should yield loss = 1
        a = np.zeros((1, 10))
        b = np.ones((1, 10))
        self.assertAlmostEqual(custom_losses.cosine_similarity([a, b]).numpy().squeeze(), 1, places=3)

    def test_euclidean_distance(self):
        # Random vectors should yield loss > 0
        a = np.random.rand(1, 10)
        b = np.random.rand(1, 10)
        self.assertGreater(custom_losses.euclidean_distance([a, b]), 0)
            
    def test_norm_euclidean_distance(self):
        # Random vectors should yield loss > 0
        a = np.random.rand(1, 10)
        b = np.random.rand(1, 10)
        self.assertGreater(custom_losses.norm_euclidean_distance([a, b]), 0)

if __name__ == '__main__':
    unittest.main()