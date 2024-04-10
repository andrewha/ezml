#include <iostream>
#include <armadillo>
#include "logreg_model.hpp"
#include "diff_loss_functions.hpp"
#include "derivative_solver.hpp"
#include "metrics.hpp"

int main()
{
    
    // Classification: single feature
    // Data: Toy example
    Features X = Features("0; 1; 2; 3; 4; 5; 6; 7; 8; 9");
    X.print("\nFeatures");
    Target y = Target("0 0 0 0 0 1 1 1 1 1");
    y.print("\nTarget");

    // Fit logistic regression model with derivative solver

    double lr = 1e-1; // 1e-1
    size_t max_iter = 100; // 100
    double max_deriv_size = 1e-3; // 1e-2
    bool verbose = false;
    // Log-likelihood grad
    //DerivativeSolver solver(DiffLoss::LOG_LIKELIHOOD_LOSS_GRAD, lr, max_iter, max_deriv_size, verbose);
    // Log-likelihood newton
    DerivativeSolver solver(DiffLoss::LOG_LIKELIHOOD_LOSS_NEWTON, lr, max_iter, max_deriv_size, verbose);

    std::cout << "\nOptimizing with: " << solver.get_name() << std::endl;
    LogRegModel lr_model(solver);
    
    // Create copy of features, since `fit()` changes features matrix
    Features X_copy(X);
    std::cout << "\nFitting with: " << lr_model.get_name() << std::endl;
    lr_model.fit(X_copy, y);
    Weights w = lr_model.get_weights();
    std::cout << "\nLearned weights (w_0, ..., w_" << arma::size(w).n_cols - 1 << "): " << w;
    std::cout << "\nLR model is fitted: " << std::boolalpha << lr_model.is_fitted() << std::endl;

    // Make predictions -- classify
    double threshold = 0.5;
    Target y_pred = lr_model.predict(X_copy, threshold /* default = 0.5 */);
    y_pred.brief_print("\nPredicted target:");
    
    // Compute metrics
    std::cout << "\nMetrics:";
    std::cout << "\nMSE: " << Metrics::mse(y, y_pred);
    std::cout << "\nR2: " << Metrics::r2(y, y_pred);
    std::cout << "\nAccuracy score: " << Metrics::accuracy(y, y_pred);
    std::cout << "\nPrecision @ " << threshold << ": " << Metrics::precision(y, y_pred);
    std::cout << "\nRecall @ " << threshold << ": " << Metrics::recall(y, y_pred);
    std::cout << "\nF1 score @ " << threshold << ": " << Metrics::f1_score(y, y_pred);

    // Get mean target
    std::cout << "\ny_mean = " << arma::mean(BaseModel().fit(X_copy, y).predict(X_copy));
    
    // Compute and print Precision-Recall curve and its AUC
    size_t num = 1001;
    PRCurve pr_curve = Metrics::pr_curve(y, lr_model.predict_proba(X_copy), num /* default = 101 */);
    std::cout << "\nAUC_PR = " << Metrics::auc(pr_curve);
    // Compute Confusion Matrix
    Metrics::confusion_matrix(y, y_pred).print("\nConfusion matrix:");

    // Find threshold to balance precision and recall: argmin(|precision - recall|), i.e FP = FN
    //pr_curve.first.t().print("\nPrecisions:");
    //pr_curve.second.t().print("\nRecalls:");
    std::cout << "\nThreshold @ Precision = Recall: " << (double)arma::index_min(arma::abs(pr_curve.first - pr_curve.second)) / (num - 1);
    
    // Find threshold to maximize TP while minimizing FP: argmax(TPs - FPs)
    TPs tp_curve = Metrics::tp_curve(y, lr_model.predict_proba(X_copy), num /* default = 101 */);
    FPs fp_curve = Metrics::fp_curve(y, lr_model.predict_proba(X_copy), num /* default = 101 */);
    //tp_curve.t().print("\nTrue Positives count:");
    //fp_curve.t().print("\nFalse Positives count:");
    std::cout << "\nThreshold @ Precision -> max: " << (double)arma::index_max(tp_curve - fp_curve) / (num - 1);

    // Compute and print ROC curve and its AUC
    ROCCurve roc_curve = Metrics::roc_curve(y, lr_model.predict_proba(X_copy), num /* default = 101 */);
    std::cout << "\nAUC_ROC = " << Metrics::auc(roc_curve) << std::endl;
    //roc_curve.first.t().print("\nRecalls:");
    //roc_curve.second.t().print("\nFallouts:");

    return EXIT_SUCCESS;
}