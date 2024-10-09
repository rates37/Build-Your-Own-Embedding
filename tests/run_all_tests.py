import unittest
import sys, os

sys.path.append(f"{os.getcwd()}/..")


def main() -> None:
    testLoader = unittest.TestLoader()
    testSuite = unittest.TestSuite()

    # discover all tests:
    testSuite.addTests(testLoader.discover("", pattern="test_*.py"))

    # run all tests:
    runner = unittest.TextTestRunner()
    runner.run(testSuite)


if __name__ == "__main__":
    main()
