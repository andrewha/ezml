/**
 * @file derivative_solver.cpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief Derivative-based solver class implementation
 * @version 0.1
 * @date 2024-03-28
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <armadillo>
#include "derivative_solver.hpp"
#include "predict_functions.hpp"
//#include "metrics.hpp"

DerivativeSolver::DerivativeSolver(const std::function<Derivative(const Weights&, const Features&, const Target&)>& diff_loss_func,
                                   const double learning_rate, 
                                   const size_t max_iter, 
                                   const double min_derivative_size, 
                                   const bool verbose /*=false*/)
: diff_loss_func_(diff_loss_func)
, learning_rate_(learning_rate)
, max_iter_(max_iter)
, min_derivative_size_(min_derivative_size)
{
    // Init solver here, if needed
    verbose_ = verbose;
    // Set solvers's name as string representation of its type
    name_ = Types::get_name(*this);
}

const Derivative DerivativeSolver::compute_derivative(const Weights& w, const Features& X, const Target& y_true)
{
    
    if (verbose_)
    {
        // Report metrics, if needed
    }
    
    const Derivative deriv = diff_loss_func_(w, X, y_true);
    return deriv;
}

const Weights DerivativeSolver::optimize(Weights& w, const Features& X, const Target& y)
{
    // Stopping criteria:
    // (a) Max number of iterations has exceeded `max_iter_`
    // (b) Derivative vector's size is less than `min_derivative_size_`
    Derivative deriv;
    for (size_t cur_iter = 0; cur_iter < max_iter_; ++cur_iter) // (a)
    {
        if (verbose_)
        {
            std::cout << "Iter: " << cur_iter << std::endl;
            std::cout << "Weights: " << w;
        }
        deriv = compute_derivative(w, X, y);
        double derivative_size = arma::norm(deriv);
        if (verbose_)
            std::cout << "Derivative 2-norm: " << derivative_size << std::endl;
        if (derivative_size <= min_derivative_size_) // (b)
        {
            if (verbose_)
                std::cout << "\n\033[33mEarly stopping:\033[0m Derivative is no longer decreasing\n";
            break;
        }
        w -= learning_rate_ * deriv;
    }
    return w;

}
