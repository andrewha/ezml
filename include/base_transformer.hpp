/**
 * @file base_transformer.hpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief BaseTransformer class declarations
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef BASE_TRANSFORMER_HPP
#define BASE_TRANSFORMER_HPP

#include "types.hpp"

using namespace Types;

/**
 * @brief Base Transformer class. All concrete transformer classes must inherit from this class.
 * 
 */
class BaseTransformer
{
    public:

        /**
         * @brief Construct a new Base Transformer object.
         * 
         */
        BaseTransformer();

        /**
         * @brief Destroy the Base Transformer object.
         * 
         */
        virtual ~BaseTransformer();

        /**
         * @brief Fit transformer.
         * 
         * @return BaseTransformer
         */
        BaseTransformer fit(Features& X);

        /**
         * @brief Transform feature variables with fitted model.
         * 
         * @param X Matrix of feature variables
         * @return const Target 
         */
        const Features transform(Features& X);

        /**
         * @brief Fit transformer, then transform feature variables.
         * 
         * @param X Matrix of feature variables
         * @return const Features 
         */
        const Features fit_transform(Features& X);
        
        /**
         * @brief Check if transformer is fitted.
         * 
         * @return true 
         * @return false 
         */
        const bool is_fitted() const;

        /**
         * @brief Get transformer's name.
         * 
         * @return std::string 
         */
        const std::string get_name() const;

    protected:

        /**
         * @brief Mark transformer as fitted.
         * 
         */
        void mark_as_fitted_();

        /**
         * @brief Transformer's name (string).
         * 
         */
        std::string name_;

    private:

        /**
         * Transformer is fitted flag.
        */
        bool fitted_;
        
};

#endif