/**
 * @file qr_solver.hpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief QRSolver class declarations
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef QR_SOLVER_HPP
#define QR_SOLVER_HPP

#include "types.hpp"
#include "base_solver.hpp"

using namespace Types;

/**
 * @brief QR-decomposition solver class. Inherits from `BaseSolver` class.
 * 
 * @tparam SolverType Type (class) of solver
 */
class QRSolver : public BaseSolver
{
    public:
        
        /**
         * @brief Construct a new QR Solver object.
         * 
         * https://en.wikipedia.org/wiki/QR_decomposition
         * 
         */
        QRSolver();
        
        /**
         * @brief Return optimized weights using QR-decomposition closed-form formula.
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