/**
 * @file predict_functions.hpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief Prediction functions declarations and implementation
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef PREDICT_FUNCTIONS_HPP
#define PREDICT_FUNCTIONS_HPP

#include <armadillo>
#include "types.hpp"

using namespace Types;

namespace Predict
{

    /**
     * @brief Predict function for linear regression.
     *
     * \f$ \displaystyle \hat{y} = w_{0} x_{0} + w_{1} x_{1} + \ldots + w_{n} x_{n} = w X \f$,
     * 
     * where \f$ X \f$ is the features matrix,
     * \f$ w \f$ is the model's weights vector.
     * 
     * @param X Matrix of feature variables
     * @param w Row vector of weights
     * @return const Target 
     */
    static const Target linreg(const Features& X, const Weights& w)
    {    
        const Target y_pred = X * w.t();
        return y_pred;        
    }
    
    /**
     * @brief Logistic function.
     * 
     * \f$ \displaystyle \sigma(z) = \frac{1}{1 + e^{-z}} \f$
     * 
     * @param z Column vector of target variable
     * @return const Target 
     */
    static const Target logistic_function(const Target& z)
    {
        return 1 / (1 + arma::exp(-z));
    }

    /**
     * @brief Predict probability of positive class for logistic regression.
     * 
     * \f$ \displaystyle \hat{y}_{proba} = \sigma(-w X) \f$,
     * 
     * where \f$ \sigma(x) \f$ is the logistic function
     * \f$ X \f$ is the features matrix,
     * \f$ w \f$ is the model's weights vector.
     * 
     * @param X Matrix of feature variables
     * @param w Row vector of weights
     * @return const Target 
     */
    static const Target logreg_proba(const Features& X, const Weights& w)
    {    
        const Target y_pred_proba = logistic_function(linreg(X, w));
        return y_pred_proba;
    }

    /**
     * @brief Predict class for logistic regression.
     * Positive class = 1. Negative class = 0.
     * 
     * \f$ \displaystyle \hat{y}(\hat{y}_{proba}, t) = \begin{cases} 1, & \text{if } \hat{y}_{proba} \ge t \\ 0, & \text{if } \hat{y}_{proba} < t \end{cases} \f$,
     * 
     * where \f$ \hat{y}_{proba} \f$ is the predicted probability of positive class,
     * \f$ t \f$ is the decision threshold.
     * 
     * @param y_pred_proba Predicted probability of positive class
     * @param threshold Decision threshold
     * @return const Target 
     */
    static const Target logreg_class(const Target& y_pred_proba, const double threshold)
    {    
        Target y_pred(y_pred_proba);
        y_pred.elem(arma::find(y_pred >= threshold)).ones();
        y_pred.elem(arma::find(y_pred < threshold)).zeros();
        return y_pred;
    }

    /**
     * @brief Compute weights for Ordinary Least Squares.
     *
     * \f$ \displaystyle w = (X^T X)^{-1} X y \f$,
     * 
     * where \f$ X \f$ is the features matrix,
     * \f$ y \f$ is the target vector.
     * 
     * Note: X must be full rank, i.e. features must be linearly independent 
     * Otherwise, its inverse is undefined, and thus solution is also undefined 
     * @param X Matrix of feature variables
     * @param y_true Column vector of target variable
     * @return const Weights 
     */
    static const Weights ols(const Features& X, const Target& y_true)
    {
        const Weights weights = (arma::inv(X.t() * X) * X.t() * y_true).t();
        return weights;
    }


    /**
     * @brief Compute weights for QR-decomposition.
     *
     * \f$ \displaystyle X = Q R \f$,
     * 
     * \f$ \displaystyle w = R^{-1} (Q^T y) \f$,
     * 
     * where \f$ X \f$ is the features matrix,
     * \f$ y \f$ is the target vector.
     *
     * @param X Matrix of feature variables
     * @param y_true Column vector of target variable
     * @return const Weights 
     */
    static const Weights qr(const Features& X, const Target& y_true)
    {
        Features Q, R;
        arma::qr_econ(Q, R, X);
        const Weights weights = (arma::inv(R) * (Q.t() * y_true)).t();
        return weights;
    }

}

#endif
