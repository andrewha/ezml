/**
 * @file base_model.cpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief BaseModel class implementation
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <armadillo>
#include "base_model.hpp"
#include "exceptions.hpp"

BaseModel::BaseModel()
: fitted_(false)
{
    // Init model here
    // Set model's name as string representation of its type
    name_ = Types::get_name(*this);
}

BaseModel::~BaseModel() {}

const bool BaseModel::is_fitted() const
{
    return fitted_;
}

void BaseModel::mark_as_fitted_()
{
    fitted_ = true;
}

const BaseModel BaseModel::fit(const Features& X, const Target& y)
{
    // BaseModel knows nothing about how to fit
    // So, basically learn the mean value of target variable
    y_mean_ = Target(y.n_rows, arma::fill::ones) * arma::mean(y);
    // Model is fitted now
    mark_as_fitted_();
    // Return object for possible cascading in pipelines
    return *this;
}

const Target BaseModel::predict(const Features& X) const
{
    // Throw if not fitted yet
    if (!is_fitted())
        throw NotFittedException(get_name());

    // BaseModel knows nothing about how to predict
    // So, basically return the learned mean value of target variable
    return y_mean_;
}

const std::string BaseModel::get_name() const
{
    return name_;
}