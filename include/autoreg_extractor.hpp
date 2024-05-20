/**
 * @file autoreg_extractor.hpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief AutoRegExtractor class declarations
 * @version 0.1
 * @date 2024-04-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef AUTOREG_EXTRACTOR_HPP
#define AUTOREG_EXTRACTOR_HPP

#include "types.hpp"

using namespace Types;

/**
 * @brief Features and target extractor class for Autoregressive AR(p) model.
 * Extracts features and target variables from time series with lag of order p to use in `AutoRegModel`.
 * 
 */
class AutoRegExtractor
{
    public:

        /**
         * @brief Construct a new AutoRegExtractor object of order p.
         * 
         */
        AutoRegExtractor(const size_t p);

        /**
         * @brief Extract feature variables: \f$ \{ X_{t-i} \}, i = 1 \ldots p \f$.
         * 
         * @param process Time series vector
         * @return const Features 
         */
        const Features extract_X(TimeSeries& process);

        /**
         * @brief Extract target variable: \f$ y_{t} \f$.
         * 
         * @param process Time series vector
         * @return const Target 
         */
        const Target extract_y(TimeSeries& process);

        /**
         * @brief Get extractors's name.
         * 
         * @return std::string 
         */
        const std::string get_name() const;

    private:

        /**
         * Order of lag.
        */
        size_t p_;

        /**
         * @brief Extractor's name (string).
         * 
         */
        std::string name_;
        
};

#endif