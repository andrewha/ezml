/**
 * @file derivative_solver.hpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief DerivativeSolver class declarations
 * @version 0.1
 * @date 2024-03-28
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef DERIVATIVE_SOLVER_HPP
#define DERIVATIVE_SOLVER_HPP

#include <functional>
#include "types.hpp"
#include "base_solver.hpp"

using namespace Types;

/**
 * @brief Derivative Solver class. Inherits from `BaseSolver` class.
 * 
 */
class DerivativeSolver : public BaseSolver
{
    public:
        
        /**
         * @brief Construct a new Derivative Solver object.
         * 
         * @param diff_loss_func Derivative of loss function
         * @param learning_rate Learning rate
         * @param max_iter Max number of optimization iterations
         * @param min_grad_size Min size of the vector of derivative
         * @param verbose 
         */
        DerivativeSolver(const std::function<Derivative(const Weights&, const Features&, const Target&)>& diff_loss_func,
                         const double learning_rate, 
                         const size_t max_iter, 
                         const double min_grad_size, 
                         const bool verbose);
        
        /**
         * @brief Return learned weights by using gradient descent for MSE loss function.
         * 
         * @param w Row vector of weights
         * @param X Matrix of feature variables
         * @param y Column vector of target variable
         * @return Weights 
         */
        const Weights optimize(Weights& w, const Features& X, const Target& y);

        /**
         * @brief Compute derivative of loss function.
         * 
         * @param w Row vector of weights
         * @param X Matrix of feature variables
         * @param y Column vector of target variable
         * @return Derivative
         */
        const Derivative compute_derivative(const Weights& w, const Features& X, const Target& y);

    private:

        /**
         * @brief Derivative of loss function used for computing optimization step.
         * 
         */
        const std::function<Derivative(const Weights&, const Features&, const Target&)>& diff_loss_func_;
        
        /**
         * @brief Learning rate.
         * 
         */
        const double learning_rate_;
        
        /**
         * @brief Max number of optimization iterations.
         * 
         */
        const size_t max_iter_;
        
        /**
         * @brief Min size of the vector of derivative.
         * 
         */
        const double min_derivative_size_;

};

#endif