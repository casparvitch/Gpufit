"""
    Equivalent to https://github.com/gpufit/Gpufit/blob/master/Gpufit/tests/Gauss_Fit_1D.cpp
"""

import unittest
import numpy as np
import pygpufit.cpufit as cf

def generate_rabi(parameters, x):
    """
    Generates a 1D damped rabi curve.

    :param parameters: The parameters (c, omega, pos, amp, tau)
    :param x: The x values
    :return: A 1D damped rabi curve
    """

    c, omega, pos, amp, tau = parameters

    return c + amp * np.exp(-(x / tau)) * np.cos(omega * (x - pos))

class Test(unittest.TestCase):

    def test_cpufit(self):
        # constants
        n_fits = 1
        n_points = 25
        n_parameter = 5 # model will be DAMPED_RABI

        # true parameters
        true_parameters = np.array((0.01, 5, np.pi / 8, 0.4, np.pi / 3), dtype=np.float32)

        # generate data
        data = np.empty((n_fits, n_points), dtype=np.float32)
        x = np.arange(n_points, dtype=np.float32)
        data[0, :] = generate_rabi(true_parameters, x)

        # tolerance
        tolerance = 0.001

        # max_n_iterations
        max_n_iterations = 10

        # model id
        model_id = cf.ModelID.DAMPED_RABI

        # initial parameters
        initial_parameters = np.empty((n_fits, n_parameter), dtype=np.float32)
        initial_parameters[0, :] = (0, 1, 0.5, 0.5, 0.1)

        print("\n=== Cpufit test damped rabi: ===")

        # call to cpufit
        parameters, states, chi_squares, number_iterations, execution_time = cf.fit(data, None, model_id,
                                                                                    initial_parameters, tolerance=tolerance, \
                                                                                    max_number_iterations=max_n_iterations, parameters_to_fit=None, 
                                                                                    estimator_id=None, user_info=np.array(x, dtype=np.float32))

        # print results
        for i in range(n_parameter):
            print(' p{} true {} fit {}'.format(i, true_parameters[i], parameters[0, i]))
        print('fit state : {}'.format(states))
        print('chi square: {}'.format(chi_squares))
        print('iterations: {}'.format(number_iterations))
        print('time: {} s'.format(execution_time))

        self.assertTrue(chi_squares < 1e-6)
        self.assertTrue(states == 0)
        self.assertTrue(number_iterations <= max_n_iterations)
        for i in range(n_parameter):
            self.assertTrue(abs(true_parameters[i] - parameters[0, i]) < 1e-6)

if __name__ == '__main__':
    unittest.main()
