# ===========================================================================================
# This program demostrates how to use the UnitTest framework in Python by solving a
# linear system
# -------------------------------------------------------------------------------------------
# Running:
#   $ python unit_test.py
# Command-line arguments:
#   -v = Verbose output (more information about the tests)
# ===========================================================================================

import unittest
import numpy as np

# Import the module containing the solver function to be tested 
from solver import solve_linear_system

# First we inherit the unittest class
class TestLinearSystem (unittest.TestCase):

    # Defining the setUp function to build the test
    def setUp (self):
        self.A = np.matrix([[ 3., 2., 4. ], [ 1., 1., 2. ], [ 4., 3., -2. ]]) # Matrix
        self.b = np.matrix([ [1.], [2.], [3.] ])                              # RHS
        self.TOLER = 1.0e-16                                                  # Tolerance

    def test_linear_system(self):
        x = solve_linear_system(self.A,self.b)          # Call the solver function
        residue = np.linalg.norm(self.A*x - self.b)     # Calculate teh residue of the system
        self.assertAlmostEqual(residue,self.TOLER)      # Check if the residue is below the tolerance

# Pass through all the test  ...
if __name__ == '__main__':
    unittest.main()