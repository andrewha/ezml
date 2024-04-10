#include <iostream>
#include <armadillo>
#include "base_model.hpp"
#include "linreg_model.hpp"
#include "base_solver.hpp"
#include "ols_solver.hpp"
#include "qr_solver.hpp"
#include "derivative_solver.hpp"
#include "diff_loss_functions.hpp"
#include "metrics.hpp"
#include "exceptions.hpp"

int main()
{
    
    // Regression: multiple features
    // Data: Random N(0,1)
    Features X = arma::dmat(10, 5, arma::fill::randn);
    X.print("\nFeatures: random");
    Target y = arma::dvec(10, arma::fill::randn);
    y.print("\nTarget: random");

    // Create BaseModel
    {
        BaseModel model;
        std::cout << "\nModel is fitted: " << std::boolalpha << model.is_fitted() << std::endl;
        
        try
        {
            model.predict(X); // try to predict with an unfitted model
        }
        catch(NotFittedException& e)
        {
            std::cerr << e.what();
        }
        std::cout << "\nFitting with: " << model.get_name() << std::endl;
        model.fit(X, y);
        std::cout << "\nModel is fitted: " << std::boolalpha << model.is_fitted() << std::endl;
        Target y_pred = model.predict(X);
        y_pred.print("\nPredicted target");
        // Pipeline
        y_pred = model.fit(X, y).predict(X);
        y_pred.print("\nFit then predict");

        // Compute metrics
        std::cout << "\nMetrics:";
        std::cout << "\nCoef corr between X and y: " << arma::cor(X, y);
        std::cout << "\nMSE: " << Metrics::mse(y, y_pred);
        std::cout << "\nSSE: " << Metrics::sse(y, y_pred);
        std::cout << "\nSST: " << Metrics::sst(y);
        std::cout << "\nR2: " << Metrics::r2(y, y_pred);
        Target residuals = y - y_pred;
        std::cout << "\nResiduals mean: " << arma::mean(residuals);
        std::cout << "\nResiduals stddev: " << arma::stddev(residuals);
        std::cout << "\nCoef corr between residuals and predictions: " << arma::cor(y_pred, residuals) << std::endl;
    }

    // Create base solver and linear regression model
    {
        BaseSolver base_solver;
        LinRegModel lr_model(base_solver);
        std::cout << "\nLR model is fitted: " << std::boolalpha << lr_model.is_fitted() << std::endl;
        
        try
        {
            lr_model.predict(X); // try to predict with an unfitted model
        }
        catch(NotFittedException& e)
        {
            std::cerr << e.what();
        }
        
        // Create copy of features, since `fit()` changes features matrix
        Features X_copy(X);
        std::cout << "\nFitting with: " << lr_model.get_name() << std::endl;
        lr_model.fit(X_copy, y);
        Weights w = lr_model.get_weights();
        std::cout << "\nLearned weights (w_0, ..., w_" << arma::size(w).n_cols - 1 << "): " << w;
        std::cout << "\nLR model is fitted: " << std::boolalpha << lr_model.is_fitted() << std::endl;
        Target y_pred = lr_model.predict(X_copy);
        y_pred.print("\nPredicted target");

        // Compute metrics
        std::cout << "\nMetrics:";
        std::cout << "\nMSE: " << Metrics::mse(y, y_pred);
        std::cout << "\nSSE: " << Metrics::sse(y, y_pred);
        std::cout << "\nSST: " << Metrics::sst(y);
        std::cout << "\nR2: " << Metrics::r2(y, y_pred);
        Target residuals = y - y_pred;
        std::cout << "\nResiduals mean: " << arma::mean(residuals);
        std::cout << "\nResiduals stddev: " << arma::stddev(residuals);
        std::cout << "\nCoef corr between residuals and predictions: " << arma::cor(y_pred, residuals) << std::endl;
    }

    // Create OLS or QR solver and linear regression model
    {
        //OLSSolver solver;
        QRSolver solver;
        LinRegModel lr_model(solver);
        std::cout << "\nLR model is fitted: " << std::boolalpha << lr_model.is_fitted() << std::endl;
        
        // Create copy of features, since `fit()` changes features matrix
        Features X_copy(X);
        std::cout << "\nFitting with: " << lr_model.get_name() << std::endl;
        lr_model.fit(X_copy, y);
        Weights w = lr_model.get_weights();
        std::cout << "\nLearned weights (w_0, ..., w_" << arma::size(w).n_cols - 1 << "): " << w;
        std::cout << "\nLR model is fitted: " << std::boolalpha << lr_model.is_fitted() << std::endl;
        Target y_pred = lr_model.predict(X_copy);
        y_pred.print("\nPredicted target");

        // Compute metrics
        std::cout << "\nMetrics:";
        std::cout << "\nMSE: " << Metrics::mse(y, y_pred);
        std::cout << "\nSSE: " << Metrics::sse(y, y_pred);
        std::cout << "\nSST: " << Metrics::sst(y);
        std::cout << "\nR2: " << Metrics::r2(y, y_pred);
        Target residuals = y - y_pred;
        std::cout << "\nResiduals mean: " << arma::mean(residuals);
        std::cout << "\nResiduals stddev: " << arma::stddev(residuals);
        std::cout << "\nCoef corr between residuals and predictions: " << arma::cor(y_pred, residuals) << std::endl;
    }

    // Create derivative solver and linear regression model
    {
        double lr = 1e-2; // 1e-5
        size_t max_iter = 1000; // 1e3
        double max_deriv_size = 1e-3; // 1e-2
        bool verbose = false;
        // MSE grad
        DerivativeSolver solver(DiffLoss::MEAN_SQUARED_ERROR_LOSS_GRAD, lr, max_iter, max_deriv_size, verbose);
        // MSE newton
        //DerivativeSolver solver(DiffLoss::MEAN_SQUARED_ERROR_LOSS_NEWTON, lr, max_iter, max_deriv_size, verbose);
        LinRegModel lr_model(solver);
        // Create copy of features, since `fit()` changes features matrix
        Features X_copy(X);
        std::cout << "\nFitting with: " << lr_model.get_name() << std::endl;
        lr_model.fit(X_copy, y);
        Weights w = lr_model.get_weights();
        std::cout << "\nLearned weights (w_0, ..., w_" << arma::size(w).n_cols - 1 << "): " << w;
        std::cout << "\nLR model is fitted: " << std::boolalpha << lr_model.is_fitted() << std::endl;
        Target y_pred = lr_model.predict(X_copy);
        y_pred.print("\nPredicted target");
        
        // Compute metrics
        std::cout << "\nMetrics:";
        std::cout << "\nMSE: " << Metrics::mse(y, y_pred);
        std::cout << "\nSSE: " << Metrics::sse(y, y_pred);
        std::cout << "\nSST: " << Metrics::sst(y);
        std::cout << "\nR2: " << Metrics::r2(y, y_pred);
        Target residuals = y - y_pred;
        std::cout << "\nResiduals mean: " << arma::mean(residuals);
        std::cout << "\nResiduals stddev: " << arma::stddev(residuals);
        std::cout << "\nCoef corr between residuals and predictions: " << arma::cor(y_pred, residuals) << std::endl;
    }

    return EXIT_SUCCESS;
}