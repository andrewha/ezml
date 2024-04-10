/**
 * @file ols_solver.hpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief OLSSolver class declarations
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef OLS_SOLVER_HPP
#define OLS_SOLVER_HPP

#include "types.hpp"
#include "base_solver.hpp"

using namespace Types;

/**
 * @brief Ordinary Least Squares solver class. Inherits from `BaseSolver` class.
 * 
 * @tparam SolverType Type (class) of solver
 */
class OLSSolver : public BaseSolver
{
    public:
        
        /**
         * @brief Construct a new Ordinary Least Squares Solver object.
         * 
         */
        OLSSolver();
        
        /**
         * @brief Return optimized weights using ordinary least squares closed-form formula.
         * 
         * @param w Row vector of weights -- not used
         * @param X Matrix of feature variables
         * @param y Column vector of target variable
         * @return Weights 
         */
        const Weights optimize(Weights& w, const Features& X, const Target& y);

    private:

        // Add private members, if needed
};

#endif