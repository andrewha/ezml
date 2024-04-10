/**
 * @file base_transformer.cpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief BaseTransformer class implementation
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <armadillo>
#include "base_transformer.hpp"
#include "exceptions.hpp"

BaseTransformer::BaseTransformer()
: fitted_(false)
{
    // Init transformer here
    // Set transformer's name as string representation of its type
    name_ = Types::get_name(*this);
}

BaseTransformer::~BaseTransformer() {}

const bool BaseTransformer::is_fitted() const
{
    return fitted_;
}

void BaseTransformer::mark_as_fitted_()
{
    fitted_ = true;
}

BaseTransformer BaseTransformer::fit(Features& X)
{
    // BaseTransformer knows nothing about how to fit
    // So, basically do nothing
    // Transformer is fitted now
    mark_as_fitted_();
    // Return object for possible cascading in pipelines
    return *this;
}

const Features BaseTransformer::transform(Features& X)
{
    // Throw if not fitted yet
    if (!is_fitted())
        throw NotFittedException(get_name());

    // BaseTransformer knows nothing about how to transform
    // So, basically return original features back
    return X;
}

const Features BaseTransformer::fit_transform(Features& X)
{
    return fit(X).transform(X);
}

const std::string BaseTransformer::get_name() const
{
    return name_;
}
