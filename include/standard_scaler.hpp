/**
 * @file standard_scaler.hpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief StandardScaler class declarations
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef STANDARD_SCALER_HPP
#define STANDARD_SCALER_HPP

#include "types.hpp"
#include "base_transformer.hpp"

using namespace Types;

/**
 * @brief Standard Scaler (\f$ z \f$-score transformation) class. Inherits from `BaseTransformer` class.
 * 
 * @tparam SolverType Type (class) of solver
 */
class StandardScaler : public BaseTransformer
{
    public:

        /**
         * @brief Construct a new Standard Scaler object.
         * 
         */
        StandardScaler();

        /**
         * @brief Fit scaler.
         * 
         * @param X Matrix of feature variables
         * @return StandardScaler
         */
        StandardScaler fit(Features& X);

        /**
         * @brief Transform feature variables with fitted scaler.
         * 
         * @param X Matrix of feature variables
         * @return const Features
         */
        const Features transform(Features& X);

        /**
         * @brief Fit scaler, then transform feature variables.
         * 
         * @param X Matrix of feature variables
         * @return const Features 
         */
        const Features fit_transform(Features& X);

        /**
         * @brief Return matrix of learned means with shape of features.
         * 
         * @return const Features 
         */
        const Features get_means() const;


        /**
         * @brief Return matrix of learned standard deviations with shape of features.
         * 
         * @return const Features 
         */
        const Features get_stddevs() const;

    private:

        /**
         * @brief Learned means of features.
         * 
         */
        Features means_;

        /**
         * @brief Learned standard deviations of features.
         * 
         */
        Features stddevs_;
        
};

#endif