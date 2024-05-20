#include <iostream>
#include <armadillo>
#include "autoreg_extractor.hpp"
#include "autoreg_model.hpp"
#include "ols_solver.hpp"
#include "qr_solver.hpp"
#include "derivative_solver.hpp"
#include "diff_loss_functions.hpp"

int main()
{
    
    // Time Series: AR(p)
    // Data: Housing from Pandas reader for HOUSTNSA, 1959-01-01, 2019-06-01
    Features data;
    data.load("./data/housing_ar.csv", arma::csv_ascii);
    // Skip the header
    data.shed_row(0);
    TimeSeries process = data.col(1);
    process.brief_print("\nTime Series:");

    // Fit AR(p) model with either OLS, or QR, or derivative solver
    
    OLSSolver solver;
    //QRSolver solver;
    double lr = 1e-1; // 1e-5
    size_t max_iter = 1000; // 1e3
    double max_deriv_size = 1e-4; // 1e-2
    bool verbose = false;
    // MSE grad
    //DerivativeSolver solver(DiffLoss::MEAN_SQUARED_ERROR_LOSS_GRAD, lr, max_iter, max_deriv_size, verbose);
    // MSE newton
    //DerivativeSolver solver(DiffLoss::MEAN_SQUARED_ERROR_LOSS_NEWTON, lr, max_iter, max_deriv_size, verbose);
    std::cout << "\nOptimizing with: " << solver.get_name() << std::endl;
    
    // Extract features and target from process
    size_t p = 3; // Order of lag
    AutoRegExtractor extractor(p);
    std::cout << "\nExtracting with: " << extractor.get_name() << std::endl;
    Features X = extractor.extract_X(process);
    X.brief_print("\nExtracted features:");
    Target y = extractor.extract_y(process);
    y.brief_print("\nExtracted target:");

    AutoRegModel ar_model(solver);
    std::cout << "\nFitting with: " << ar_model.get_name() << std::endl;
    ar_model.fit(X, y);
    Weights w = ar_model.get_weights();
    double s = ar_model.get_sigma();
    size_t o = ar_model.get_order();
    std::cout << "\nLearned weights (w_0, ..., w_" << arma::size(w).n_cols - 1 << "): " << w;
    std::cout << "\nLearned sigma: " << s;
    std::cout << "\nModel's order: " << o;
    std::cout << "\nLR model is fitted: " << std::boolalpha << ar_model.is_fitted() << std::endl;
    
    // Make forecast for some number of periods
    size_t num_periods = 100;
    TimeSeries process_pred = ar_model.predict(X, num_periods);
    process_pred.print("\nForecasted process:");
    
    std::cout << "\nCoef corr between X and y: " << arma::cor(X, y);

    return EXIT_SUCCESS;
}