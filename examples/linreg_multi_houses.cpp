#include <iostream>
#include <armadillo>
#include "standard_scaler.hpp"
#include "linreg_model.hpp"
#include "ols_solver.hpp"
#include "qr_solver.hpp"
#include "derivative_solver.hpp"
#include "diff_loss_functions.hpp"
#include "metrics.hpp"
#include "exceptions.hpp"

int main()
{
    // Regression: multiple features
    // Data: https://www.kaggle.com/datasets/ryanholbrook/dl-course-data?select=housing.csv
    Features data;
    data.load("./data/housing.csv", arma::csv_ascii);
    // Skip the header
    data.shed_row(0);
    Features X = data.cols(1, 8);
    X.brief_print("\nFeatures: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude");
    Target y = data.cols(9, 9);
    y.brief_print("\nTarget: log(MedHouseVal/1000)");

    // Fit linear regression model with either OLS, or QR, or derivative solver

    // Normalize features first to improve convergence of gradient descent
    StandardScaler std_scaler;
    try
    {
        std_scaler.transform(X);
    }
    catch(NotFittedException& e)
    {
        std::cerr << e.what();
    }
    std::cout << "\nNormalizing with: " << std_scaler.get_name() << std::endl;
    Features X_norm = std_scaler.fit_transform(X);
    std_scaler.get_means().brief_print("\nLearned means:");
    std_scaler.get_stddevs().brief_print("\nLearned stddevs:");
    X_norm.brief_print("\nNormalized features:");
    
    //OLSSolver solver;
    QRSolver solver;
    double lr = 1e-3; // 1e-5
    size_t max_iter = 1000; // 1e3
    double max_deriv_size = 1e-4; // 1e-2
    bool verbose = false;
    // MSE grad
    //DerivativeSolver solver(DiffLoss::MEAN_SQUARED_ERROR_LOSS_GRAD, lr, max_iter, max_deriv_size, verbose);
    // MSE newton
    //DerivativeSolver solver(DiffLoss::MEAN_SQUARED_ERROR_LOSS_NEWTON, lr, max_iter, max_deriv_size, verbose);
    LinRegModel lr_model(solver);
    // Try to predict with unfitted model
    try
    {
        lr_model.predict(X_norm);
    }
    catch(NotFittedException& e)
    {
        std::cerr << e.what();
    }
    
    // Fit linreg model
    // Note: `fit()` changes features matrix, so making a copy might be required
    // However, since `X_norm` is already a transformed version of original `X`, another copy of `X` is not required
    std::cout << "\nFitting with: " << lr_model.get_name() << std::endl;
    lr_model.fit(X_norm, y);
    Weights w = lr_model.get_weights();
    std::cout << "\nLearned weights (w_0, ..., w_" << arma::size(w).n_cols - 1 << "): " << w;
    std::cout << "\nLR model is fitted: " << std::boolalpha << lr_model.is_fitted() << std::endl;
    
    // Make predictions
    Target y_pred = lr_model.predict(X_norm);
    y_pred.brief_print("\nPredicted target:");
    
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