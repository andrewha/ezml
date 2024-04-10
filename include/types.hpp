/**
 * @file types.hpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief Custom types used in the library
 * @version 0.1
 * @date 2024-03-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef TYPES_HPP
#define TYPES_HPP

#include <boost/type_index.hpp>
#include <armadillo>

namespace Types
{
    /**
     * Matrix of feature variables (doubles).
    */
    using Features = arma::dmat;

    /**
     * Column vector of target variable (doubles).
    */
    using Target = arma::dvec;

    /**
     * Row vector of model's weights (doubles).
    */
    using Weights = arma::drowvec;

    /**
     * Row vector of n-th order derivative of loss function (doubles).
    */
    using Derivative = arma::drowvec;

    /**
     * Confusion Matrix (doubles).
    */
    using ConfusionMatrix = arma::umat;

    /**
     * Row vector of precisions computed for different thresholds (doubles).
    */
    using Precisions = arma::drowvec;

    /**
     * Row vector of recalls computed for different thresholds (doubles).
    */
    using Recalls = arma::drowvec;

    /**
     * Row vector of fall-outs computed for different thresholds (doubles).
    */
    using Fallouts = arma::drowvec;

    /**
     * @brief Pair of row vectors of precisions and recalls -- PR curve.
     * 
     */
    using PRCurve = std::pair<const Precisions, const Recalls>;

    /**
     * @brief Pair of row vectors of recalls and fall-outs -- ROC curve.
     * 
     */
    using ROCCurve = std::pair<const Recalls, const Fallouts>;

    /**
     * Row vector of true positives computed for different thresholds (unsigned int).
    */
    using TPs = arma::urowvec;

    /**
     * Row vector of false positives computed for different thresholds (unsigned int).
    */
    using FPs = arma::urowvec;

    /**
     * Row vector of true negatives computed for different thresholds (unsigned int).
    */
    using TNs = arma::urowvec;

    /**
     * Row vector of false negatives computed for different thresholds (unsigned int).
    */
    using FNs = arma::urowvec;

    /**
     * @brief Get the object's name as string representation of its type.
     * 
     * @tparam T 
     * @return std::string 
     */
    template <typename T>
    std::string get_name(T)
    {
        return boost::typeindex::type_id<T>().pretty_name();
    }
}

#endif