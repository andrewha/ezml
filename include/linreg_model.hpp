/**
 * @file linreg_model.hpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief LinRegModel class declarations
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef LINREG_MODEL_HPP
#define LINREG_MODEL_HPP

#include "types.hpp"
#include "base_model.hpp"

using namespace Types;

/**
 * @brief Linear Regression model class template. Inherits from `BaseModel` class.
 * 
 * @tparam SolverType class of solver: `BaseSolver`, `OLSSolver`, `QRSolver`, `DerivativeSolver`
 */
template <typename SolverType>
class LinRegModel : public BaseModel
{
    public:

        /**
         * @brief Construct a new LinRegModel object.
         * 
         */
        LinRegModel(const SolverType& solver);

        
        /**
         * @brief Get model's weights.
         * 
         * @return const Weights 
         */
        const Weights get_weights() const;

        /**
         * @brief Fit model.
         * 
         * @param X Matrix of feature variables
         * @param y Column vector of target variable
         * @return LinRegModel
         */
        const LinRegModel fit(Features& X, const Target& y);

        /**
         * @brief Predict target variable with fitted model.
         * 
         * @param X Matrix of feature variables
         * @return const Target 
         */
        const Target predict(const Features& X) const;

    private:
        
        /**
         * @brief Row vector of model's weights.
         * 
         */
        Weights weights_;
        
        /**
         * @brief Solver to fit model with.
         * 
         */
        SolverType solver_;
        
};

#endif