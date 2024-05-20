/**
 * @file autoreg_model.hpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief AutoRegModel class declarations
 * @version 0.1
 * @date 2024-04-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef AUTOREG_MODEL_HPP
#define AUTOREG_MODEL_HPP

#include "types.hpp"
#include "base_model.hpp"

using namespace Types;

/**
 * @brief Autoregressive AR(p) model class template. Inherits from `BaseModel` class.
 * 
 * @tparam SolverType class of solver: `BaseSolver`, `OLSSolver`, `QRSolver`, `DerivativeSolver`
 */
template <typename SolverType>
class AutoRegModel : public BaseModel
{
    public:
        
        /**
         * @brief Construct a new AutoRegModel object
         * 
         * @param solver Solver type
         * @param p Model's order (lag)
         */
        AutoRegModel(const SolverType& solver);

        
        /**
         * @brief Get model's weights.
         * 
         * @return const Weights 
         */
        const Weights get_weights() const;

        /**
         * @brief Get model's sigma (standard deviation).
         * 
         * @return const double 
         */
        const double get_sigma() const;

        /**
         * @brief Get model's order (lag).
         * 
         * @return const size_t
         */
        const size_t get_order() const;

        /**
         * @brief Fit AR(p) model.
         * 
         * @param X Matrix of feature variables extracted with `AutoRegExtractor`
         * @param y Column vector of target variable extracted with `AutoRegExtractor`
         * @return AutoRegModel
         */
        const AutoRegModel fit(Features& X, const Target& y);

        /**
         * @brief Predict future values of time series with fitted model.
         * 
         * @param X Matrix of feature variables extracted with `AutoRegExtractor`
         * @param num_periods Number of forecast periods
         * @return const TimeSeries
         */
        const TimeSeries predict(const Features& X, const size_t num_periods) const;

    private:
        
        /**
         * @brief Row vector of model's weights.
         * 
         */
        Weights weights_;
        
        /**
         * @brief Model's sigma (standard deviation).
         * 
         */
        double sigma_;

        /**
         * @brief Model's order (lag).
         * 
         */
        size_t p_;

        /**
         * @brief Solver to fit model with.
         * 
         */
        SolverType solver_;
        
};

#endif