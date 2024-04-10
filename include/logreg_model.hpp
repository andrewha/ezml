/**
 * @file logreg_model.hpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief LogRegModel class declarations
 * @version 0.1
 * @date 2024-03-27
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef LOGREG_MODEL_HPP
#define LOGREG_MODEL_HPP

#include "types.hpp"
#include "base_model.hpp"
#include "base_solver.hpp"
#include "derivative_solver.hpp"

using namespace Types;

/**
 * @brief Logistic Regression model class template. Inherits from `BaseModel` class.
 * 
 * @tparam SolverType class of solver: `BaseSolver`, `DerivativeSolver`
 */
template <typename SolverType>
class LogRegModel : public BaseModel
{
    public:

        /**
         * @brief Construct a new LogRegModel object.
         * 
         */
        LogRegModel(const SolverType& solver);

        
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
         * @return LogRegModel
         */
        const LogRegModel fit(Features& X, const Target& y);

        /**
         * @brief Predict (classify) target variable's class with fitted model at given threshold.
         * 
         * Positive class is 1 and negative class is 0
         * 
         * @param X Matrix of feature variables
         * @param threshold Threshold [0, 1] (double)
         * @return const Target 
         */
        const Target predict(const Features& X, const double& threshold=0.5) const;

        /**
         * @brief Predict probability of positive class of target variable  with fitted model.
         * 
         * @param X Matrix of feature variables
         * @return const Target 
         */
        const Target predict_proba(const Features& X) const;

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