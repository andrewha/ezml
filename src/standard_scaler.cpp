/**
 * @file standard_scaler.cpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief StandardScaler class implementation
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <armadillo>
#include "standard_scaler.hpp"
#include "exceptions.hpp"

StandardScaler::StandardScaler()
{
    // Init scaler here
    // Set scaler's name as string representation of its type
    name_ = Types::get_name(*this);
}

StandardScaler StandardScaler::fit(Features& X)
{
    // 1. Compute mean and standard deviation of each feature
    means_ = arma::repmat(arma::mean(X), X.n_rows, 1);
    stddevs_ = arma::repmat(arma::stddev(X), X.n_rows, 1);
    // 2. Scaler is fitted now
    mark_as_fitted_();
    // Return object for possible cascading in pipelines
    return *this;
}

const Features StandardScaler::transform(Features& X)
{
    // Throw if not fitted yet
    if (!is_fitted())
        throw NotFittedException(get_name());

    // Scale features
    return (X - means_) / stddevs_;
}

const Features StandardScaler::fit_transform(Features& X)
{
    return fit(X).transform(X);
}

const Features StandardScaler::get_means() const
{
    return means_;
}

const Features StandardScaler::get_stddevs() const
{
    return stddevs_;
}
