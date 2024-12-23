/**
 * @file metrics.hpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief Model metrics declarations and implementation
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef METRICS_HPP
#define METRICS_HPP

#include <armadillo>
#include "types.hpp"
#include "predict_functions.hpp"

using namespace Types;

namespace Metrics
{

    /**
     * @brief Compute Mean Squared Error of predictions.
     * 
     * \f$ \displaystyle MSE(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n} (y_{i} - \hat{y_{i}})^2 \f$,
     * where \f$ y \f$ is the target vector, 
     * \f$ \hat{y} \f$ is model's predictions vector, 
     * \f$ n \f$ is the number of predictions.
     * 
     * @param y_true Column vector of ground truth target
     * @param y_pred Column vector of predicted target
     * @return const double
     */
    static const double mse(const Target& y_true, const Target& y_pred)
    {
        return arma::mean(arma::pow(y_true - y_pred, 2));
    }

    /**
     * @brief Compute Sum Squared Error of predictions. In fact, \f$ n MSE(y, \hat{y}) \f$.
     * 
     * \f$ \displaystyle SSE(y, \hat{y}) = \sum_{i=1}^{n} (y_{i} - \hat{y_{i}})^2 \f$,
     * where \f$ y \f$ is the target vector, 
     * \f$ \hat{y} \f$ is model's predictions vector, 
     * \f$ n \f$ is the number of predictions.
     * 
     * @param y_true Column vector of ground truth target
     * @param y_pred Column vector of predicted target
     * @return const double 
     */
    static const double sse(const Target& y_true, const Target& y_pred)
    {
        return arma::sum(arma::pow(y_true - y_pred, 2));
    }

    /**
     * @brief Compute Sum Squared Total variance of target. In fact, \f$ n Var(y) \f$.
     * 
     * \f$ \displaystyle SST(y) = \sum_{i=1}^{n} (y_{i} - \bar{y})^2 \f$,
     * where \f$ y \f$ is the target vector, 
     * \f$ \bar{y} \f$ is target vector's mean, 
     * \f$ n \f$ is the number of predictions.
     * 
     * @param y_true Column vector of ground truth target
     * @return const double 
     */
    static const double sst(const Target& y_true)
    {
        return arma::sum(arma::pow(y_true - arma::mean(y_true), 2));
    }

    /**
     * @brief Coefficient of determination of predictions.
     * 
     * \f$ \displaystyle R^2 = 1 - \frac{SSE(y, \hat{y})}{SST(y)} \f$.
     * Can also be expressed as \f$ \displaystyle R^2 = 1 - \frac{n MSE(y, \hat{y})}{n Var(y)} = 1 - \frac{MSE(y, \hat{y})}{Var(y)} \f$,
     * where \f$ SSE \f$ is Metrics::sse,
     * \f$ SST \f$ is Metrics::sst,
     * \f$ MSE \f$ is Metrics::mse.
     * 
     * @param y_true Column vector of ground truth target
     * @param y_pred Column vector of predicted target
     * @return const double 
     */
    static const double r2(const Target& y_true, const Target& y_pred)
    {
        return 1.0 - sse(y_true, y_pred) / sst(y_true);
    }

    /**
     * @brief Normalized accuracy score of predictions for binary classifier.
     * 
     * \f$ \displaystyle Acc = \frac{1}{n} \sum_{i=1}^{n} [\hat{y} = y] \f$,
     * where \f$ y \f$ is the target vector, 
     * \f$ \hat{y} \f$ is the model's predictions vector, 
     * \f$ n \f$ is the number of predictions.
     * 
     * @param y_true Column vector of ground truth target
     * @param y_pred Column vector of predicted target
     * @return const double 
     */
    static const double accuracy(const Target& y_true, const Target& y_pred)
    {
        return arma::mean(arma::conv_to<Target>::from(y_pred == y_true));
    }

    /**
     * @brief True Positives count.
     * 
     * \f$ \displaystyle TP = |\{y | y = 1\} \cap \{\hat{y} | \hat{y} = 1\}| \f$,
     * where \f$ y \f$ is the target vector, 
     * \f$ \hat{y} \f$ is the model's predictions vector.
     * 
     * @param y_true Column vector of ground truth target
     * @param y_pred Column vector of predicted target
     * @return const size_t 
     */
    static const size_t tp_count(const Target& y_true, const Target& y_pred)
    {
        return arma::sum(arma::conv_to<Target>::from(y_true == 1) && arma::conv_to<Target>::from(y_pred == 1));
    }

    /**
     * @brief False Positives count.
     * 
     * \f$ \displaystyle FP = |\{y | y = 0\} \cap \{\hat{y} | \hat{y} = 1\}| \f$,
     * where \f$ y \f$ is the target vector, 
     * \f$ \hat{y} \f$ is the model's predictions vector.
     * 
     * @param y_true Column vector of ground truth target
     * @param y_pred Column vector of predicted target
     * @return const size_t
     */
    static const size_t fp_count(const Target& y_true, const Target& y_pred)
    {
        return arma::sum(arma::conv_to<Target>::from(y_true == 0) && arma::conv_to<Target>::from(y_pred == 1));
    }

    /**
     * @brief True Negatives count.
     * 
     * \f$ \displaystyle TN = |\{y | y = 0\} \cap \{\hat{y} | \hat{y} = 0\}| \f$,
     * where \f$ y \f$ is the target vector, 
     * \f$ \hat{y} \f$ is the model's predictions vector.
     * 
     * @param y_true Column vector of ground truth target
     * @param y_pred Column vector of predicted target
     * @return const size_t
     */
    static const size_t tn_count(const Target& y_true, const Target& y_pred)
    {
        return arma::sum(arma::conv_to<Target>::from(y_true == 0) && arma::conv_to<Target>::from(y_pred == 0));
    }

    /**
     * @brief False Negatives count.
     * 
     * \f$ \displaystyle FN = |\{y | y = 1\} \cap \{\hat{y} | \hat{y} = 0\}| \f$,
     * where \f$ y \f$ is the target vector, 
     * \f$ \hat{y} \f$ is the model's predictions vector.
     * 
     * @param y_true Column vector of ground truth target
     * @param y_pred Column vector of predicted target
     * @return const size_t
     */
    static const size_t fn_count(const Target& y_true, const Target& y_pred)
    {
        return arma::sum(arma::conv_to<Target>::from(y_true == 1) && arma::conv_to<Target>::from(y_pred == 0));
    }

    /**
     * @brief Return confusion matrix.   
     * 
     * <!-- -->             | \f$ y = 1 \f$       | \f$ y = 0 \f$
     * :------------------: | :-----------------: | :-----------------:
     * \f$ \hat{y} = 1 \f$  | Metrics::tp_count | Metrics::fp_count
     * \f$ \hat{y} = 0 \f$  | Metrics::fn_count | Metrics::tn_count
     * 
     * @param y_true Column vector of ground truth target
     * @param y_pred Column vector of predicted target
     * @return const Types::ConfusionMatrix
     */
    static const ConfusionMatrix confusion_matrix(const Target& y_true, const Target& y_pred)
    {
        ConfusionMatrix cm = { 
                               {tp_count(y_true, y_pred), fp_count(y_true, y_pred)},
                               {fn_count(y_true, y_pred), tn_count(y_true, y_pred)}
                             };
        return cm;
    }

    /**
     * @brief Precision (Positive class).
     * 
     * \f$ \displaystyle Precision = P(y = 1 | \hat{y} = 1) = \frac{P(y = 1 \cap \hat{y} = 1)}{P(\hat{y}) = 1} = \frac{TP}{TP + FP} \f$,
     * where \f$ y \f$ is the target vector, 
     * \f$ \hat{y} \f$ is the model's predictions vector.
     * 
     * @param y_true Column vector of ground truth target
     * @param y_pred Column vector of predicted target
     * @return const double
     */
    static const double precision(const Target& y_true, const Target& y_pred)
    {
        double tp = tp_count(y_true, y_pred);
        double fp = fp_count(y_true, y_pred);
        if (tp || fp)
            return tp / (tp + fp);
        else
            return 1.0;
    }

    /**
     * @brief Recall (Positive class). Same as True Positive Rate. Same as Sensitivity.
     * 
     * \f$ \displaystyle Recall = P(y = 1 | \hat{y} = 1) = \frac{P(y = 1 \cap \hat{y} = 1)}{P(y) = 1} = \frac{TP}{TP + FN} \f$,
     * where \f$ y \f$ is the target vector, 
     * \f$ \hat{y} \f$ is the model's predictions vector.
     * 
     * @param y_true Column vector of ground truth target
     * @param y_pred Column vector of predicted target
     * @return const double
     */
    static const double recall(const Target& y_true, const Target& y_pred)
    {
        double tp = tp_count(y_true, y_pred);
        double fn = fn_count(y_true, y_pred);
        if (tp || fn)
            return tp / (tp + fn);
        else
            return 1.0;
    }

    /**
     * @brief False Positive Rate. Same as Fall-out.
     * Same as probability of Type I error (falsely rejecting correct null hypothesis).
     * 
     * \f$ \displaystyle Fallout = P(y = 0 | \hat{y} = 1) = \frac{P(y = 0 \cap \hat{y} = 1)}{P(y) = 0} = \frac{FP}{FP + TN} \f$,
     * where \f$ y \f$ is the target vector, 
     * \f$ \hat{y} \f$ is the model's predictions vector.
     * 
     * @param y_true Column vector of ground truth target
     * @param y_pred Column vector of predicted target
     * @return const double
     */
    static const double fpr(const Target& y_true, const Target& y_pred)
    {
        double fp = fp_count(y_true, y_pred);
        double tn = tn_count(y_true, y_pred);
        if (fp || tn)
            return fp / (fp + tn);
        else
            return 1.0;
    }

    /**
     * @brief F1-score (Positive class).
     * 
     * \f$ \displaystyle F_{1} = 2 \times \frac{Precision \times Recall}{Precision + Recall} \f$
     * 
     * @param y_true Column vector of ground truth target
     * @param y_pred Column vector of predicted target
     * @return const double
     */
    static const double f1_score(const Target& y_true, const Target& y_pred)
    {
        double prec = precision(y_true, y_pred);
        double rec = recall(y_true, y_pred);
        double f1 = 2.0 * prec * rec / (prec + rec);
        return f1;
    }

    /**
     * @brief Compute Precision-Recall curve: Metrics::precision vs Metrics::recall for different classification thresholds.
     * 
     * @param y_true Column vector of ground truth target
     * @param y_pred_proba Column vector of predicted probabilities of positive class
     * @param num Number of thresholds
     * @return const Types::PrecisionsRecalls
     */
    static const PRCurve pr_curve(const Target& y_true, const Target& y_pred_proba, const size_t num=101)
    {
        Precisions precisions(num);
        Recalls recalls(num);
        auto thresholds = arma::linspace(0.0, 1.0, num);
        for (size_t i = 0; i < num; ++i)
        {
            Target y_pred = Predict::logreg_class(y_pred_proba, thresholds[i]);
            precisions[i] = Metrics::precision(y_true, y_pred);
            recalls[i] = Metrics::recall(y_true, y_pred);
        }

        return PRCurve(precisions, recalls);
    }

    /**
     * @brief Compute Receiver Operating Characteristic (ROC) curve: Metrics::recall vs Metrics::fpr for different classification thresholds.
     * 
     * @param y_true Column vector of ground truth target
     * @param y_pred_proba Column vector of predicted probabilities of positive class
     * @param num Number of thresholds
     * @return const Types::PrecisionsRecalls
     */
    static const ROCCurve roc_curve(const Target& y_true, const Target& y_pred_proba, const size_t num=101)
    {
        Recalls recalls(num);
        Fallouts fallouts(num);
        auto thresholds = arma::linspace(0.0, 1.0, num);
        for (size_t i = 0; i < num; ++i)
        {
            Target y_pred = Predict::logreg_class(y_pred_proba, thresholds[i]);
            recalls[i] = Metrics::recall(y_true, y_pred);
            fallouts[i] = Metrics::fpr(y_true, y_pred);
        }

        return ROCCurve(recalls, fallouts);
    }

    /**
     * @brief Compute True Positives curve: Metrics::tp_count vs classification threshold.
     * 
     * @param y_true Column vector of ground truth target
     * @param y_pred_proba Column vector of predicted probabilities of positive class
     * @param num Number of thresholds
     * @return const Types::TPs
     */
    static const TPs tp_curve(const Target& y_true, const Target& y_pred_proba, const size_t num=101)
    {
        TPs tps(num);
        auto thresholds = arma::linspace(0.0, 1.0, num);
        for (size_t i = 0; i < num; ++i)
        {
            Target y_pred = Predict::logreg_class(y_pred_proba, thresholds[i]);
            tps[i] = Metrics::tp_count(y_true, y_pred);
        }

        return tps;
    }

    /**
     * @brief Compute False Positives curve: Metrics::fp_count vs classification threshold.
     * 
     * @param y_true Column vector of ground truth target
     * @param y_pred_proba Column vector of predicted probabilities of positive class
     * @param num Number of thresholds
     * @return const Types::FPs
     */
    static const FPs fp_curve(const Target& y_true, const Target& y_pred_proba, const size_t num=101)
    {
        FPs fps(num);
        auto thresholds = arma::linspace(0.0, 1.0, num);
        for (size_t i = 0; i < num; ++i)
        {
            Target y_pred = Predict::logreg_class(y_pred_proba, thresholds[i]);
            fps[i] = Metrics::fp_count(y_true, y_pred);
        }

        return fps;
    }

    /**
     * @brief Compute True Negatives curve: Metrics::tn_count vs classification threshold.
     * 
     * @param y_true Column vector of ground truth target
     * @param y_pred_proba Column vector of predicted probabilities of positive class
     * @param num Number of thresholds
     * @return const Types::TNs
     */
    static const TNs tn_curve(const Target& y_true, const Target& y_pred_proba, const size_t num=101)
    {
        TNs tns(num);
        auto thresholds = arma::linspace(0.0, 1.0, num);
        for (size_t i = 0; i < num; ++i)
        {
            Target y_pred = Predict::logreg_class(y_pred_proba, thresholds[i]);
            tns[i] = Metrics::tn_count(y_true, y_pred);
        }

        return tns;
    }

    /**
     * @brief Compute False Negatives curve: Metrics::fn_count vs classification threshold.
     * 
     * @param y_true Column vector of ground truth target
     * @param y_pred_proba Column vector of predicted probabilities of positive class
     * @param num Number of thresholds
     * @return const Types::FNs
     */
    static const FNs fn_curve(const Target& y_true, const Target& y_pred_proba, const size_t num=101)
    {
        FNs fns(num);
        auto thresholds = arma::linspace(0.0, 1.0, num);
        for (size_t i = 0; i < num; ++i)
        {
            Target y_pred = Predict::logreg_class(y_pred_proba, thresholds[i]);
            fns[i] = Metrics::fn_count(y_true, y_pred);
        }

        return fns;
    }

    /**
     * @brief Area Under Curve for Metrics::pr_curve or Metrics::roc_curve.
     * 
     * \f$ \displaystyle AUC_{PR} = \int_{0}^{1} Precision(Recall) \mathrm{d}Recall \f$
     * 
     * \f$ \displaystyle AUC_{ROC} = \int_{0}^{1} Recall(Fallout) \mathrm{d}Fallout \f$
     * 
     * @param pair Metrics::PRCurve or Metrics::ROCCurve (arma::drowvec)
     * @return const double
     */
    static const double auc(const std::pair<const arma::drowvec, const arma::drowvec>& pair)
    {
        return arma::conv_to<double>::from(arma::trapz(arma::fliplr(pair.second), arma::fliplr(pair.first), 1));
    }

}

#endif