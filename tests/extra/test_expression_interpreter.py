import unittest
import jax.numpy as jnp
from photon_weave.extra.expression_interpreter import interpreter

class TestExpressionInterpreter(unittest.TestCase):
    def test_s_pow(self):
        context = {}
        dimensions = []
        result = interpreter(('s_pow', 2, 3), context, dimensions)
        expected = 2 ** 3
        self.assertEqual(result, expected)
  
    def test_m_pow(self):
        context = {}
        dimensions = []
        matrix = jnp.array([[1, 2], [3, 4]])
        result = interpreter(('m_pow', matrix, 3), context, dimensions)
        expected = jnp.linalg.matrix_power(matrix, 3)
        self.assertTrue(jnp.allclose(result, expected))
