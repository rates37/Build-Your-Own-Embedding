import unittest
import numpy as np
from BuildYourOwnEmbedding.parameters import (
    UniformRangeParameter,
    RandomRangeParameter,
    ConstantParameter,
)

EPSILON = 1e-6


class TestParameters(unittest.TestCase):

    def test_uniform_range_parameter(self):

        # Test 1:
        minVal, maxVal = 0, 1
        numSamples = 5
        expectedRange = [minVal, maxVal]
        expectedValues = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        param = UniformRangeParameter(
            minValue=minVal, maxValue=maxVal, numSamples=numSamples
        )
        values = param.get_values()
        self.assertEqual(len(values), numSamples)
        self.assertTrue(
            np.all(values >= expectedRange[0]) and np.all(values <= expectedRange[1])
        )
        self.assertTrue(all(abs(values - expectedValues) < EPSILON))

        # Test 2:
        minVal, maxVal = -1, 1
        numSamples = 9
        expectedRange = [minVal, maxVal]
        expectedValues = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])

        param = UniformRangeParameter(
            minValue=minVal, maxValue=maxVal, numSamples=numSamples
        )
        values = param.get_values()
        self.assertEqual(len(values), numSamples)
        self.assertTrue(
            np.all(values >= expectedRange[0]) and np.all(values <= expectedRange[1])
        )
        self.assertTrue(all(abs(values - expectedValues) < EPSILON))

        # Test 3:
        minVal, maxVal = -24, -22
        numSamples = 8
        expectedRange = [minVal, maxVal]
        expectedValues = np.array(
            [
                -24.0,
                -23.71428571,
                -23.42857143,
                -23.14285714,
                -22.85714286,
                -22.57142857,
                -22.28571429,
                -22.0,
            ]
        )

        param = UniformRangeParameter(
            minValue=minVal, maxValue=maxVal, numSamples=numSamples
        )
        values = param.get_values()
        self.assertEqual(len(values), numSamples)
        self.assertTrue(
            np.all(values >= expectedRange[0]) and np.all(values <= expectedRange[1])
        )
        self.assertTrue(all(abs(values - expectedValues) < EPSILON))

    def test_random_range_parameter(self):

        # Test 1:
        minVal, maxVal = 0, 1
        numSamples = 5
        expectedRange = [minVal, maxVal]
        param = RandomRangeParameter(
            minValue=minVal, maxValue=maxVal, numSamples=numSamples
        )
        values = param.get_values()
        self.assertEqual(len(values), numSamples)
        self.assertTrue(
            np.all(values >= expectedRange[0]) and np.all(values <= expectedRange[1])
        )

        # Test 2:
        minVal, maxVal = -20, 20
        numSamples = 40
        expectedRange = [minVal, maxVal]
        param = RandomRangeParameter(
            minValue=minVal, maxValue=maxVal, numSamples=numSamples
        )
        values = param.get_values()
        self.assertEqual(len(values), numSamples)
        self.assertTrue(
            np.all(values >= expectedRange[0]) and np.all(values <= expectedRange[1])
        )

        # Test 3:
        minVal, maxVal = 37, 1200
        numSamples = 123
        expectedRange = [minVal, maxVal]
        param = RandomRangeParameter(
            minValue=minVal, maxValue=maxVal, numSamples=numSamples
        )
        values = param.get_values()
        self.assertEqual(len(values), numSamples)
        self.assertTrue(
            np.all(values >= expectedRange[0]) and np.all(values <= expectedRange[1])
        )

    def test_constant_parameter(self):
        # Test 1:
        value = 0.5
        param = ConstantParameter(value=value)
        values = param.get_values()
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0], value)

        # Test 2:
        value = -5
        param = ConstantParameter(value=value)
        values = param.get_values()
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0], value)

        # Test 3:
        value = 45
        param = ConstantParameter(value=value)
        values = param.get_values()
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0], value)


if __name__ == "__main__":
    unittest.main()
