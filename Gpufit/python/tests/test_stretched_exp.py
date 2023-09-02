"""
    Equivalent to https://github.com/gpufit/Gpufit/blob/master/Gpufit/tests/Gauss_Fit_1D.cpp
"""

import unittest
import numpy as np
import pygpufit.gpufit as gf

def generate_stretched_exp(parameters, x):
    """
    Generates a 1D damped rabi curve.

    :param parameters: The parameters (c, exp_t, amp, power)
    :param x: The x values
    :return: A 1D damped rabi curve
    """

    c, exp_t, amp, power = parameters

    return c + amp * np.exp( (x / exp_t) ** power)

class Test(unittest.TestCase):

    def test_gpufit(self):
        # constants
        n_fits = 1
        n_points = 25
        n_parameter = 4  # model will be STRETCHED_EXP

        # true parameters
        true_parameters = np.array((0.01, 12, 1.0, 0.7), dtype=np.float32)

        # generate data
        data = np.empty((n_fits, n_points), dtype=np.float32)
        x = np.linspace(0.1, 20, n_points, dtype=np.float32)
        data[0, :] = generate_stretched_exp(true_parameters, x)

        # tolerance
        tolerance = 1e-9

        # max_n_iterations
        max_n_iterations = 50

        # model id
        model_id = gf.ModelID.STRETCHED_EXP

        # initial parameters
        initial_parameters = np.empty((n_fits, n_parameter), dtype=np.float32)
        initial_parameters[0, :] = (0.05, 10, 0.5, 0.5)

        print("\n=== Gpufit test stretched_exp: ===")
        self.assertTrue(gf.cuda_available(), msg=gf.get_last_error())

        # call to gpufit
        parameters, states, chi_squares, number_iterations, execution_time = gf.fit(data, None, model_id,
                                                                                    initial_parameters, tolerance=tolerance, \
                                                                                    max_number_iterations=max_n_iterations, parameters_to_fit=None, 
                                                                                    estimator_id=None, user_info=np.array(x, dtype=np.float32),)


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
            self.assertTrue(abs(true_parameters[i] - parameters[0, i]) < 1e-5)


if __name__ == '__main__':
    unittest.main()
