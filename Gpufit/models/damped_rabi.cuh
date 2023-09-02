/* Description of the calculate_lorentz_8p function
* ==============================================
*
* This function calculates the values of a one-dimensional stretched exponential
* model function, and its partial derivatives with respect to the model parameters. 
*
* This function makes use of the user information data to pass in the 
* independent variables (X values) corresponding to the data.  The X values
* must be of type REAL.
*
* How the X values are handled:
*
*       The user_info array contains the X values for 'one' fit, which are
*       used for all fits. The size of the user_info array (in bytes) must
*       equal sizeof(REAL) * n_points.
*
* Parameters:
*
* parameters: An input vector of model parameters.
*   p[0]: offset, c
*   p[1]: characteristic exponential time, charac_exp_t
*   p[2]: amplitude of exponential, amp
*   p[3]: power *inside* exponential, power_exp
*   i.e. fit model is: c + amp * exp(- (x / charac_exp_t) ^ power_exp)
*
* n_fits: The number of fits. (not used)
*
* n_points: The number of data points per fit.
*
* value: An output vector of model function values.
*
* derivative: An output vector of model function partial derivatives.
*
* point_index: The data point index.
*
* fit_index: The fit index. (not used)
*
* chunk_index: The chunk index. (not used)
*
* user_info: An input vector containing user information. 
*
* user_info_size: The size of user_info in bytes. 
*
* Calling the calculate_lorentz_8p function
* ======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/
__device__ void calculate_damped_rabi (
    float const * parameters,
    int const n_fits,
    int const n_points,
    float * value,
    float * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{
    // indices (track independent sweep vector values, 'x' (i.e. taus or freqs))

    REAL * user_info_float = (REAL*)user_info;
    REAL x = user_info_float[point_index];

    // parameters

    REAL const * p = parameters;
    /*
    *   p[0]: constant, c
    *   p[1]: rabi frequency (angular), omega
    *   p[2]: time offset of cosine, pos
    *   p[3]: amplitude, amp
    *   p[4]: decay time constant, tau
    *   i.e. fit model is: amp * exp(-(x/tau)) * cos(omega * (x-pos)) + const
    */
    REAL const c = p[0];
    REAL const omega = p[1];
    REAL const pos = p[2];
    REAL const amp = p[3];
    REAL const tau = p[4];

    value[point_index] = c + amp * std::exp(-(x/tau)) * std::cos(omega * (x-pos));

    // derivatives

    float * current_derivative = derivative + point_index;
    
    current_derivative[0 * n_points] = 1; // wrt c
    current_derivative[1 * n_points] = amp * (pos - x) * std::sin(omega * (x - pos)) * 
        std::exp(-x / tau); // wrt omega
    current_derivative[2 * n_points] = (amp * omega * std::sin(omega * (x - pos))) * 
        std::exp(-x / tau); // wrt pos
    current_derivative[3 * n_points] = std::exp(-x / tau) * std::cos(omega * (x - pos)); // wrt amp
    current_derivative[4 * n_points] = (amp * x * std::cos(omega * (x - pos))) / 
        (std::exp(x / tau) * tau * tau); // wrt tau
}
