/**
 * @file autoreg_model.cpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief AutoRegModel class implementation
 * @version 0.1
 * @date 2024-04-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <armadillo>
#include <boost/random.hpp>
#include "autoreg_model.hpp"
#include "exceptions.hpp"
#include "predict_functions.hpp"
#include "base_solver.hpp"
#include "ols_solver.hpp"
#include "qr_solver.hpp"
#include "derivative_solver.hpp"

template <typename SolverType>
AutoRegModel<SolverType>::AutoRegModel(const SolverType& solver)
: weights_()
, sigma_()
, p_()
, solver_(solver)
{
    // Init model here, if needed
    // Set model's name as string representation of its type
    name_ = Types::get_name(*this);
}

template <typename SolverType>
const Weights AutoRegModel<SolverType>::get_weights() const
{
    return weights_;
}

template <typename SolverType>
const double AutoRegModel<SolverType>::get_sigma() const
{
    return sigma_;
}

template <typename SolverType>
const size_t AutoRegModel<SolverType>::get_order() const
{
    return p_;
}

template <typename SolverType>
const AutoRegModel<SolverType> AutoRegModel<SolverType>::fit(Features& X, const Target& y)
{

    // 1. Set model's order p
    p_ = X.n_cols;
    // 2. By default, intercept weight (w_0) is learned, so add dummy feature for it
    Features dummy_feature = Features(X.n_rows, 1, arma::fill::ones);
    //std::cout << dummy_feature;
    X.insert_cols(0, dummy_feature);
    // 3. Init weights with Gaussian noise N(0, 1)
    weights_ = Weights(X.n_cols, arma::fill::randn);
    // 4. Learn weights with solver
    weights_ = solver_.optimize(weights_, X, y);
    // 5. Learn sigma
    sigma_ = arma::stddev(y);
    // 6. Model is fitted now
    mark_as_fitted_();
    // Return object for possible cascading in pipelines
    return *this;
}

template <typename SolverType>
const TimeSeries AutoRegModel<SolverType>::predict(const Features& X, const size_t num_periods) const
{
    // Throw if not fitted yet
    if (!is_fitted())
        throw NotFittedException(get_name());

    Features X_roll = X.tail_rows(1);
    //X_roll.print("\nX_new:");
    // Simulate AR process
    TimeSeries forecast;
    // Create random number generator, randomizing it on every method call
    boost::mt19937 rng(time(0));
    // Gaussian (white) noise with zero mean and learned standard deviation
    boost::random::normal_distribution<> wn(0, sigma_);
    // Create generator for normal distribution
    // Every call of `sample()` will generate one random value from the distribution `noise`
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<>> sample(rng, wn);
    Target mu;
    for (size_t period = 0; period < num_periods; ++period)
    {
        mu = Predict::linreg(X_roll, weights_); // Predict expected value
        forecast.insert_rows(forecast.n_rows, mu + sample()); // Add white noise and populate forecast
        X_roll.insert_cols(1, forecast.tail(1)); // Roll features
        X_roll = X_roll.head_cols(p_ + 1); // by keeping only p latest + 1 (intercept) values
        //X_roll.print("\nX_roll:");
    }

    return forecast;
}

// Explicitly instantiate templates for actual required types
// BaseSolver
template AutoRegModel<BaseSolver>::AutoRegModel(const BaseSolver&);
template const Weights AutoRegModel<BaseSolver>::get_weights() const;
template const double AutoRegModel<BaseSolver>::get_sigma() const;
template const size_t AutoRegModel<BaseSolver>::get_order() const;
template const AutoRegModel<BaseSolver> AutoRegModel<BaseSolver>::fit(Features&, const Target&);
template const TimeSeries AutoRegModel<BaseSolver>::predict(const Features&, const size_t) const;
// OLSSolver
template AutoRegModel<OLSSolver>::AutoRegModel(const OLSSolver&);
template const Weights AutoRegModel<OLSSolver>::get_weights() const;
template const double AutoRegModel<OLSSolver>::get_sigma() const;
template const size_t AutoRegModel<OLSSolver>::get_order() const;
template const AutoRegModel<OLSSolver> AutoRegModel<OLSSolver>::fit(Features&, const Target&);
template const TimeSeries AutoRegModel<OLSSolver>::predict(const Features&, const size_t) const;
// QRSolver
template AutoRegModel<QRSolver>::AutoRegModel(const QRSolver&);
template const Weights AutoRegModel<QRSolver>::get_weights() const;
template const double AutoRegModel<QRSolver>::get_sigma() const;
template const size_t AutoRegModel<QRSolver>::get_order() const;
template const AutoRegModel<QRSolver> AutoRegModel<QRSolver>::fit(Features&, const Target&);
template const TimeSeries AutoRegModel<QRSolver>::predict(const Features&, const size_t) const;
// DerivativeSolver
template AutoRegModel<DerivativeSolver>::AutoRegModel(const DerivativeSolver&);
template const Weights AutoRegModel<DerivativeSolver>::get_weights() const;
template const double AutoRegModel<DerivativeSolver>::get_sigma() const;
template const size_t AutoRegModel<DerivativeSolver>::get_order() const;
template const AutoRegModel<DerivativeSolver> AutoRegModel<DerivativeSolver>::fit(Features&, const Target&);
template const TimeSeries AutoRegModel<DerivativeSolver>::predict(const Features&, const size_t) const;
