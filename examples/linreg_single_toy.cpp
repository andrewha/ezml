#include <iostream>
#include <armadillo>
#include "linreg_model.hpp"
#include "ols_solver.hpp"
#include "qr_solver.hpp"
#include "derivative_solver.hpp"
#include "diff_loss_functions.hpp"
#include "metrics.hpp"

int main()
{
    
    // Regression: single feature
    // Data: Toy example
    Features X = Features("0; 1; 2; 3; 4; 5; 6; 7; 8; 9");
    X.print("\nFeatures");
    Target y = Target("5 3 3 8 7 8 11 9 9 12");
    y.print("\nTarget");

    // Fit linear regression model with either OLS, or QR, or derivative solver
    
    //OLSSolver solver;
    QRSolver solver;
    double lr = 1e-1; // 1e-5
    size_t max_iter = 1000; // 1e3
    double max_deriv_size = 1e-4; // 1e-2
    bool verbose = false;
    // MSE grad
    //DerivativeSolver solver(DiffLoss::MEAN_SQUARED_ERROR_LOSS_GRAD, lr, max_iter, max_deriv_size, verbose);
    // MSE newton
    //DerivativeSolver solver(DiffLoss::MEAN_SQUARED_ERROR_LOSS_NEWTON, lr, max_iter, max_deriv_size, verbose);
    std::cout << "\nOptimizing with: " << solver.get_name() << std::endl;
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
    std::cout << "\nCoef corr between X and y: " << arma::cor(X, y);
    std::cout << "\nMSE: " << Metrics::mse(y, y_pred);
    std::cout << "\nSSE: " << Metrics::sse(y, y_pred);
    std::cout << "\nSST: " << Metrics::sst(y);
    std::cout << "\nR2: " << Metrics::r2(y, y_pred);
    Target residuals = y - y_pred;
    std::cout << "\nResiduals mean: " << arma::mean(residuals);
    std::cout << "\nResiduals stddev: " << arma::stddev(residuals);
    std::cout << "\nCoef corr between residuals and predictions: " << arma::cor(y_pred, residuals) << std::endl;

    return EXIT_SUCCESS;
}