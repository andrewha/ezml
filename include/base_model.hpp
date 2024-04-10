/**
 * @file base_model.hpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief BaseModel class declarations
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef BASE_MODEL_HPP
#define BASE_MODEL_HPP

#include "types.hpp"

using namespace Types;

/**
 * @brief Base Model class. All concrete model classes must inherit from this class.
 * 
 */
class BaseModel
{
    public:

        /**
         * @brief Construct a new BaseModel object.
         * 
         */
        BaseModel();

        /**
         * @brief Destroy the BaseModel object.
         * 
         */
        virtual ~BaseModel();

        /**
         * @brief Fit model.
         * 
         * @return BaseModel
         */
        const BaseModel fit(const Features& X, const Target& y);

        /**
         * @brief Predict target variable with fitted model.
         * 
         * @param X Matrix of feature variables
         * @return const Target 
         */
        const Target predict(const Features& X) const;
        
        /**
         * @brief Check if model is fitted.
         * 
         * @return true 
         * @return false 
         */
        const bool is_fitted() const;

        /**
         * @brief Get model's name.
         * 
         * @return std::string 
         */
        const std::string get_name() const;

    protected:

        /**
         * @brief Mark model as fitted.
         * 
         */
        void mark_as_fitted_();

        /**
         * @brief Model's name (string).
         * 
         */
        std::string name_;

    private:

        /**
         * Model is fitted flag.
        */
        bool fitted_;

        /**
         * @brief Base predictions -- simply the mean value of target variable.
         * 
         */
        Target y_mean_;
        
};

#endif