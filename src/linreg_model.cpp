/**
 * @file linreg_model.cpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief Linear regression model class implementation
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <armadillo>
#include "linreg_model.hpp"
#include "exceptions.hpp"
#include "predict_functions.hpp"
#include "base_solver.hpp"
#include "ols_solver.hpp"
#include "qr_solver.hpp"
#include "derivative_solver.hpp"

template <typename SolverType>
LinRegModel<SolverType>::LinRegModel(const SolverType& solver)
: weights_()
, solver_(solver)
{
    // Init model here, if needed
    // Set model's name as string representation of its type
    name_ = Types::get_name(*this);
}

template <typename SolverType>
const Weights LinRegModel<SolverType>::get_weights() const
{
    return weights_;
}

template <typename SolverType>
const LinRegModel<SolverType> LinRegModel<SolverType>::fit(Features& X, const Target& y)
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
const Target LinRegModel<SolverType>::predict(const Features& X) const
{
    // Throw if not fitted yet
    if (!is_fitted())
        throw NotFittedException(get_name());

    const Target y_pred = Predict::linreg(X, weights_);
    return y_pred;
}

// Explicitly instantiate templates for actual required types
// BaseSolver
template LinRegModel<BaseSolver>::LinRegModel(const BaseSolver&);
template const Weights LinRegModel<BaseSolver>::get_weights() const;
template const LinRegModel<BaseSolver> LinRegModel<BaseSolver>::fit(Features&, const Target&);
template const Target LinRegModel<BaseSolver>::predict(const Features&) const;
// OLSSolver
template LinRegModel<OLSSolver>::LinRegModel(const OLSSolver&);
template const Weights LinRegModel<OLSSolver>::get_weights() const;
template const LinRegModel<OLSSolver> LinRegModel<OLSSolver>::fit(Features&, const Target&);
template const Target LinRegModel<OLSSolver>::predict(const Features&) const;
// QRSolver
template LinRegModel<QRSolver>::LinRegModel(const QRSolver&);
template const Weights LinRegModel<QRSolver>::get_weights() const;
template const LinRegModel<QRSolver> LinRegModel<QRSolver>::fit(Features&, const Target&);
template const Target LinRegModel<QRSolver>::predict(const Features&) const;
// DerivativeSolver
template LinRegModel<DerivativeSolver>::LinRegModel(const DerivativeSolver&);
template const Weights LinRegModel<DerivativeSolver>::get_weights() const;
template const LinRegModel<DerivativeSolver> LinRegModel<DerivativeSolver>::fit(Features&, const Target&);
template const Target LinRegModel<DerivativeSolver>::predict(const Features&) const;
