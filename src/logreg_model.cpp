/**
 * @file logreg_model.cpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief Logistic regression model class implementation
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <armadillo>
#include "logreg_model.hpp"
#include "exceptions.hpp"
#include "predict_functions.hpp"
#include "base_solver.hpp"
#include "derivative_solver.hpp"

template <typename SolverType>
LogRegModel<SolverType>::LogRegModel(const SolverType& solver)
: weights_()
, solver_(solver)
{
    // Init model here, if needed
    // Set model's name as string representation of its type
    name_ = Types::get_name(*this);
}

template <typename SolverType>
const Weights LogRegModel<SolverType>::get_weights() const
{
    return weights_;
}

template <typename SolverType>
const LogRegModel<SolverType> LogRegModel<SolverType>::fit(Features& X, const Target& y)
{

    // 1. By default, intercept weight (w_0) is learned, so add dummy feature for it
    Features dummy_feature = Features(X.n_rows, 1, arma::fill::ones);
    //std::cout << dummy_feature;
    X.insert_cols(0, dummy_feature);
    // 2. Init weights with Gaussian noise N(0, 1)
    weights_ = Weights(X.n_cols, arma::fill::randn);
    // 3. Learn weights with solver
    weights_ = solver_.optimize(weights_, X, y);
    // 4. Model is fitted now
    mark_as_fitted_();
    // Return object for possible cascading in pipelines
    return *this;
}

template <typename SolverType>
const Target LogRegModel<SolverType>::predict(const Features& X, const double& threshold) const
{

    // 1. Predict probability of positive class
    const Target y_pred_proba = predict_proba(X);
    // 2. Classify (binarize) probability at given threshold
    const Target y_pred = Predict::logreg_class(y_pred_proba, threshold);
    return y_pred;
}

template <typename SolverType>
const Target LogRegModel<SolverType>::predict_proba(const Features& X) const
{
    // Throw if not fitted yet
    if (!is_fitted())
        throw NotFittedException(get_name());

    const Target y_pred_proba = Predict::logreg_proba(X, weights_);
    return y_pred_proba;
}

// Explicitly instantiate templates for actual required types
// BaseSolver
template LogRegModel<BaseSolver>::LogRegModel(const BaseSolver&);
template const Weights LogRegModel<BaseSolver>::get_weights() const;
template const LogRegModel<BaseSolver> LogRegModel<BaseSolver>::fit(Features&, const Target&);
template const Target LogRegModel<BaseSolver>::predict(const Features&, const double&) const;
template const Target LogRegModel<BaseSolver>::predict_proba(const Features&) const;
// DerivativeSolver
template LogRegModel<DerivativeSolver>::LogRegModel(const DerivativeSolver&);
template const Weights LogRegModel<DerivativeSolver>::get_weights() const;
template const LogRegModel<DerivativeSolver> LogRegModel<DerivativeSolver>::fit(Features&, const Target&);
template const Target LogRegModel<DerivativeSolver>::predict(const Features&, const double&) const;
template const Target LogRegModel<DerivativeSolver>::predict_proba(const Features&) const;