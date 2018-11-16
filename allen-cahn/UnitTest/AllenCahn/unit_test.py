# ===========================================================================================
# This program demostrates how to use the UnitTest framework in Python by solving
# the Allen-Cahn equation
# -------------------------------------------------------------------------------------------
# Running:
#   $ python unit_test.py
# Command-line arguments:
#   -v = Verbose output (more information about the tests)
# ===========================================================================================

import unittest
import numpy as np

# Import the module containing the solver function to be tested 
from solver import solve_allen_cahn

# First we inherit the unittest class
class TestAllenCahn (unittest.TestCase):

    # Defining the setUp function to build the test
    def setUp (self):
        self.nelem = 8                                  # Initial number of elements
        self.nsimulations = 5                           # Total number of simulations
        self.error_timestep = 0.1                       # Timestep in which the error will be calculated
        self.TOLER = 0.5                                # Tolerance
        self.min_h = float(self.nelem / (2**(self.nsimulations-1))) # Most refined space discretization

    def test_allen_cahn_cg_1(self):
        # Degree of the interpolation polynomium
        degree = 1

        # Reference value for convergeance
        a_ref = degree + 1

        # Lists to store the results
        array_h = []
        array_errorl2 = []

        # Open a file to store the output of the test 
        filename = "output/error_cg_%d.dat" % degree
        file = open(filename,"w")

        # Calculate 'dt' for a better adjustment of the error curve
        dt = self.min_h**degree    
        
        # Step 1)
        # First we run the simulations
        elem = self.nelem
        for k in range(self.nsimulations):
            #print("---------------------------------------------------------------------------------")
            #print("[!] Solving Allen-Cahn-1D using %d elements (dt = %.10lf)" % (elem,dt))
            h, error_L2 = solve_allen_cahn(elem,1,self.error_timestep,dt)
            array_h.append(h)
            array_errorl2.append(error_L2)
            file.write("%g %g\n" % (h,error_L2))
            elem = elem * 2
            #print("---------------------------------------------------------------------------------")
        #print("*****************************************************************************")
        file.close()
        
        # Step 2)
        # Now build the error curve
        log_h = np.log10(array_h)
        log_E = np.log10(array_errorl2)
        coeff = np.polyfit(log_h,log_E,1)
        a = coeff[0]
        b = coeff[1]
        #print("Error function")
        #print("y = %g . x + %g" % (a,b))
        #print("a_ref = %g" % (a_ref))
        
        # Step 3)
        # Finally we check the convergeance
        diff = np.abs(a-a_ref)
        self.assertLessEqual(diff,self.TOLER)

    def test_allen_cahn_cg_2(self):
        # Degree of the interpolation polynomium
        degree = 2

        # Reference value for convergeance
        a_ref = degree + 1

        # List to store the results
        array_h = []
        array_errorl2 = []

        # Open a file to store the output of the test 
        filename = "output/error_cg_%d.dat" % degree
        file = open(filename,"w")

        # Calculate 'dt' for a better adjustment of the error curve
        dt = self.min_h**degree    
        
        # First we run the simulations
        elem = self.nelem
        for k in range(self.nsimulations):
            h, error_L2 = solve_allen_cahn(elem,2,self.error_timestep,dt)
            array_h.append(h)
            array_errorl2.append(error_L2)
            file.write("%g %g\n" % (h,error_L2))
            elem = elem * 2
        file.close()
        
        # Now build the error curve
        log_h = np.log10(array_h)
        log_E = np.log10(array_errorl2)
        coeff = np.polyfit(log_h,log_E,1)
        a = coeff[0]
        b = coeff[1]
        #print("Error function")
        #print("y = %g . x + %g" % (a,b))
        #print("a_ref = %g" % (a_ref))
        
        # Finally we check the convergeance
        diff = np.abs(a-a_ref)
        self.assertLessEqual(diff,self.TOLER)
    
    def test_allen_cahn_cg_3(self):
        # Degree of the interpolation polynomium
        degree = 3

        # Reference value for convergeance
        a_ref = degree + 1

        # List to store the results
        array_h = []
        array_errorl2 = []

        # Open a file to store the output of the test 
        filename = "output/error_cg_%d.dat" % degree
        file = open(filename,"w")

        # Calculate 'dt' for a better adjustment of the error curve
        dt = self.min_h**degree    
        
        # First we run the simulations
        elem = self.nelem
        for k in range(self.nsimulations):
            h, error_L2 = solve_allen_cahn(elem,3,self.error_timestep,dt)
            array_h.append(h)
            array_errorl2.append(error_L2)
            file.write("%g %g\n" % (h,error_L2))
            elem = elem * 2
        file.close()
        
        # Now build the error curve
        log_h = np.log10(array_h)
        log_E = np.log10(array_errorl2)
        coeff = np.polyfit(log_h,log_E,1)
        a = coeff[0]
        b = coeff[1]
        #print("Error function")
        #print("y = %g . x + %g" % (a,b))
        #print("a_ref = %g" % (a_ref))

        # Finally we check the convergeance
        diff = np.abs(a-a_ref)
        self.assertLessEqual(diff,self.TOLER)

# Pass through all the test  ...
if __name__ == '__main__':
    unittest.main()