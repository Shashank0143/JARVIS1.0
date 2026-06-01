
import unittest
import sys
sys.path.append(r'C:\Users\Roopesh Chaudhary\PycharmProjects\JARVIS\datasets\training_corpus')
from buggy_calc import divide_vals

class TestCalc(unittest.TestCase):
    def test_divide(self):
        # This will trigger ZeroDivisionError
        val = divide_vals(10, 0)
        self.assertIsNotNone(val)

if __name__ == '__main__':
    unittest.main()
