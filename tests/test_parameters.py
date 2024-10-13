import unittest
import numpy as np
from BuildYourOwnEmbedding.parameters import (
    UniformRangeParameter,
    RandomRangeParameter,
    ConstantParameter,
    FixedParameterSet,
    LogRangeParameter,
)

EPSILON = 1e-6


class TestParameters(unittest.TestCase):

    # UniformRangeParameter tests:
    def test_uniform_range_parameter_simple(self) -> None:
        minVal, maxVal = 0, 1
        numSamples = 5
        expectedValues = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        param = UniformRangeParameter(
            minValue=minVal, maxValue=maxVal, numSamples=numSamples
        )
        values = param.get_values()
        self.assertEqual(len(values), numSamples)
        self.assertTrue(np.all((values >= minVal) & (values <= maxVal)))
        self.assertTrue(np.allclose(values, expectedValues, atol=EPSILON))

    def test_uniform_range_parameter_negative_values(self) -> None:
        minVal, maxVal = -1, 1
        numSamples = 9
        expectedValues = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
        param = UniformRangeParameter(
            minValue=minVal, maxValue=maxVal, numSamples=numSamples
        )
        values = param.get_values()
        self.assertEqual(len(values), numSamples)
        self.assertTrue(np.all((values >= minVal) & (values <= maxVal)))
        self.assertTrue(np.allclose(values, expectedValues, atol=EPSILON))

    def test_uniform_range_parameter_non_integer_values(self) -> None:
        minVal, maxVal = -24, -22
        numSamples = 8
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
        self.assertTrue(np.all((values >= minVal) & (values <= maxVal)))
        self.assertTrue(np.allclose(values, expectedValues, atol=EPSILON))

    # RandomRangeParameter tests:
    def test_random_range_parameter_basic(self) -> None:
        minVal, maxVal = 0, 1
        numSamples = 5
        param = RandomRangeParameter(
            minValue=minVal, maxValue=maxVal, numSamples=numSamples
        )
        values = param.get_values()
        self.assertEqual(len(values), numSamples)
        self.assertTrue(np.all((values >= minVal) & (values <= maxVal)))

    def test_random_range_parameter_large_range(self) -> None:
        minVal, maxVal = -20, 20
        numSamples = 40
        param = RandomRangeParameter(
            minValue=minVal, maxValue=maxVal, numSamples=numSamples
        )
        values = param.get_values()
        self.assertEqual(len(values), numSamples)
        self.assertTrue(np.all((values >= minVal) & (values <= maxVal)))

    def test_random_range_parameter_single_value(self) -> None:
        minVal, maxVal = 100, 100
        numSamples = 5
        param = RandomRangeParameter(
            minValue=minVal, maxValue=maxVal, numSamples=numSamples
        )
        values = param.get_values()
        self.assertTrue(np.all(values == 100))
        self.assertEqual(len(values), numSamples)

    def test_random_range_parameter_negative_range(self) -> None:
        minVal, maxVal = -1200, -37
        numSamples = 123
        param = RandomRangeParameter(
            minValue=minVal, maxValue=maxVal, numSamples=numSamples
        )
        values = param.get_values()
        self.assertEqual(len(values), numSamples)
        self.assertTrue(np.all((values >= minVal) & (values <= maxVal)))

    # ConstantParameter tests:
    def test_constant_parameter_single_value(self) -> None:
        value = 0.5
        param = ConstantParameter(value=value)
        values = param.get_values()
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0], value)

    def test_constant_parameter_negative_value(self) -> None:
        value = -5
        param = ConstantParameter(value=value)
        values = param.get_values()
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0], value)

    def test_constant_parameter_large_value(self) -> None:
        value = 45
        param = ConstantParameter(value=value)
        values = param.get_values()
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0], value)

    # FixedParameter tests:
    def test_fixed_parameter_set_basic(self) -> None:
        originalValues = [1, 2, 3, 4]
        param = FixedParameterSet(values=originalValues)
        values = param.get_values()
        self.assertEqual(len(values), len(originalValues))
        self.assertTrue(np.allclose(values, originalValues, atol=EPSILON))

    def test_fixed_parameter_set_float_values(self) -> None:
        originalValues = np.array([1.1, 2.2, 3.3, 4.4])
        param = FixedParameterSet(values=originalValues)
        values = param.get_values()
        self.assertEqual(len(values), len(originalValues))
        self.assertTrue(np.allclose(values, originalValues, atol=EPSILON))

    def test_fixed_parameter_set_large_set(self) -> None:
        originalValues = np.random.rand(1000)
        param = FixedParameterSet(values=originalValues)
        values = param.get_values()
        self.assertEqual(len(values), len(originalValues))
        self.assertTrue(np.allclose(values, originalValues, atol=EPSILON))

    # LogRangeParameter tests:
    def test_log_range_parameter_basic(self):
        minVal, maxVal = 1, 10
        numSamples = 5
        expectedValues = np.array([1.0, 1.77827941, 3.16227766, 5.62341325, 10.0])
        param = LogRangeParameter(
            minValue=minVal, maxValue=maxVal, numSamples=numSamples
        )
        values = param.get_values()
        self.assertEqual(len(values), numSamples)
        self.assertTrue(np.all((values >= minVal) & (values <= maxVal)))
        self.assertTrue(np.allclose(values, expectedValues, atol=EPSILON))

    def test_log_range_parameter_large_range(self):
        # Test logarithmic range with a large range of values
        minVal, maxVal = 1, 1e9
        numSamples = 10
        expectedValues = np.array(
            [
                1.0e0,
                1.0e1,
                1.0e2,
                1.0e3,
                1.0e4,
                1.0e5,
                1.0e6,
                1.0e7,
                1.0e8,
                1.0e9,
            ]
        )
        param = LogRangeParameter(
            minValue=minVal, maxValue=maxVal, numSamples=numSamples
        )
        values = param.get_values()
        self.assertEqual(len(values), numSamples)
        self.assertTrue(np.all((values >= minVal) & (values <= maxVal)))
        self.assertTrue(np.allclose(values, expectedValues, atol=EPSILON))

    def test_log_range_parameter_small_log_range(self):
        # Test logarithmic range with a narrow range of values
        minVal, maxVal = 1, 10
        numSamples = 5
        expectedValues = np.array([1.0, 1.77827941, 3.16227766, 5.62341325, 10.0])
        param = LogRangeParameter(
            minValue=minVal, maxValue=maxVal, numSamples=numSamples
        )
        values = param.get_values()
        self.assertEqual(len(values), numSamples)
        self.assertTrue(np.all((values >= minVal) & (values <= maxVal)))
        self.assertTrue(np.allclose(values, expectedValues, atol=EPSILON))


if __name__ == "__main__":
    unittest.main()
