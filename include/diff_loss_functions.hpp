/**
 * @file diff_loss_functions.hpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief Derivatives of loss functions declarations and implementation
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef DIFF_LOSS_FUNCTIONS_HPP
#define DIFF_LOSS_FUNCTIONS_HPP

#include <functional>
#include <armadillo>
#include "types.hpp"
#include "predict_functions.hpp"
#include "exceptions.hpp"

using namespace Types;

namespace DiffLoss
{

    /**
     * @brief Gradient (first derivative) of Mean Squared Error loss.
     *
     * \f$ \displaystyle \nabla L_{MSE} = -\frac{2}{n} \sum_{i=1}^{n} X^T (y_{i} - \hat{y_{i}}) \f$,
     * where \f$ X \f$ is the features matrix,
     * \f$ y \f$ is the target vector, 
     * \f$ \hat{y} \f$ is the model's predictions vector, 
     * \f$ n \f$ is the number of predictions.
     * 
     * @param w Row vector of weights
     * @param X Matrix of feature variables
     * @param y_true Column vector of target variable
     * @return const Derivative 
     */
    static const Derivative mean_squared_error_loss_grad(const Weights& w, const Features& X, const Target& y_true)
    {
        const Target y_pred = Predict::linreg(X, w);
        const Derivative grad = -2.0 * arma::mean(X.t() * (y_true - y_pred), 1).t();
        return grad;
    }

    /**
     * @brief Laplacian (second derivative) of Mean Squared Error loss.
     * 
     * \f$ \displaystyle \nabla (\nabla L_{MSE}) = \frac{2}{n} \sum X^T X \f$,
     * where \f$ X \f$ is the features matrix,
     * \f$ n \f$ is the number of observations.
     * 
     * @param X Matrix of feature variables
     * @return const Derivative 
     */
    static const Derivative mean_squared_error_loss_lapl(const Features& X)
    {
        const Derivative lapl = 2.0 * arma::mean(X.t() * X, 1).t();
        return lapl;
    }

    /**
     * @brief Gradient to Laplacian ratio (Newton) of Mean Squared Error loss.
     * 
     * \f$ \text{Step} = \displaystyle \frac{\nabla L_{MSE}}{\nabla (\nabla L_{MSE})} \f$
     * 
     * @param w Row vector of weights
     * @param X Matrix of feature variables
     * @param y_true Column vector of target variable
     * @return const Derivative 
     */
    static const Derivative mean_squared_error_loss_newton(const Weights& w, const Features& X, const Target& y_true)
    {
        // Throw if X has more than one feature
        // Since, we add dummy feature for intercept weights, X will have two columns
        if (X.n_cols > 2)
            throw NewtonShapeException();
        // L' / L''
        const Derivative newton = mean_squared_error_loss_grad(w, X, y_true) / mean_squared_error_loss_lapl(X);
        // inv(L'') L'
        //Features hess_mat = arma::eye(w.n_cols, w.n_cols) % arma::repmat(mean_squared_error_loss_lapl(X), w.n_cols, 1);
        //const Derivative newton = (arma::inv(hess_mat) * mean_squared_error_loss_grad(w, X, y_true).t()).t();
        return newton;
    }

    /**
     * @brief Gradient (first derivative) of Log Likelihood loss.
     * 
     * \f$ \displaystyle \nabla L_{LOG} = -\frac{1}{n} \sum_{i=1}^{n} X^T (y_{i} - \hat{y_{i}}) \f$,
     * where \f$ X \f$ is the features matrix,
     * \f$ y \f$ is the target vector, 
     * \f$ \hat{y} \f$ is the model's predictions vector, 
     * \f$ n \f$ is the number of predictions.
     * 
     * @param w Row vector of weights
     * @param X Matrix of feature variables
     * @param y_true Column vector of target variable
     * @return const Derivative 
     */
    static const Derivative log_likelihood_loss_grad(const Weights& w, const Features& X, const Target& y_true)
    {
        const Target y_pred_proba = Predict::logreg_proba(X, w);
        const Derivative grad = -arma::mean(X.t() * (y_true - y_pred_proba), 1).t();
        return grad;
    }

    /**
     * @brief Laplacian (second derivative) of Log Likelihood loss.
     * 
     * \f$ \displaystyle \nabla (\nabla L_{LOG}) =  \frac{1}{n} \sum_{i=1}^{n} X^T \hat{y_{i}} (1 - \hat{y_{i}})^T \f$,
     * where \f$ X \f$ is the features matrix,
     * \f$ \hat{y} \f$ is the model's predictions vector, 
     * \f$ n \f$ is the number of predictions.
     * 
     * @param w Row vector of weights
     * @param X Matrix of feature variables
     * @return const Derivative 
     */
    static const Derivative log_likelihood_loss_lapl(const Weights& w, const Features& X)
    {
        const Target y_pred_proba = Predict::logreg_proba(X, w);
        const Derivative lapl = arma::mean(X.t() * y_pred_proba * (1.0 - y_pred_proba).t(), 1).t();
        return lapl;
    }

    /**
     * @brief Gradient to Laplacian ratio (Newton) of Log Likelihood loss.
     * 
     * \f$ \displaystyle \text{Step} = \frac{\nabla L_{LOG}}{\nabla (\nabla L_{LOG})} \f$
     * 
     * @param w Row vector of weights
     * @param X Matrix of feature variables
     * @param y_true Column vector of target variable
     * @return const Derivative 
     */
    static const Derivative log_likelihood_loss_newton(const Weights& w, const Features& X, const Target& y_true)
    {
        // Throw if X has more than one feature
        // Since, we add dummy feature for intercept weights, X will have two columns
        if (X.n_cols > 2)
            throw NewtonShapeException();
        // L' / L''
        const Derivative newton = log_likelihood_loss_grad(w, X, y_true) / log_likelihood_loss_lapl(w, X);
        // inv(L'') L'
        //Features hess_mat = arma::eye(w.n_cols, w.n_cols) % arma::repmat(log_likelihood_loss_lapl(w, X), w.n_cols, 1);
        //const Derivative newton = (arma::inv(hess_mat) * log_likelihood_loss_grad(w, X, y_true).t()).t();
        return newton;
    }

    /**
     * @brief Alias for `mean_squared_error_loss_grad` function.
     * 
     */
    static const std::function<Derivative(const Weights&, const Features&, const Target&)> MEAN_SQUARED_ERROR_LOSS_GRAD = mean_squared_error_loss_grad;

    /**
     * @brief Alias for `mean_squared_error_loss_newton` function.
     * 
     */
    static const std::function<Derivative(const Weights&, const Features&, const Target&)> MEAN_SQUARED_ERROR_LOSS_NEWTON = mean_squared_error_loss_newton;

    /**
     * @brief Alias for `log_likelihood_loss_grad` function.
     * 
     */
    static const std::function<Derivative(const Weights&, const Features&, const Target&)> LOG_LIKELIHOOD_LOSS_GRAD = log_likelihood_loss_grad;

    /**
     * @brief Alias for `log_likelihood_loss_newton` function.
     * 
     */
    static const std::function<Derivative(const Weights&, const Features&, const Target&)> LOG_LIKELIHOOD_LOSS_NEWTON = log_likelihood_loss_newton;

}

#endif
