/**
 * @file base_solver.hpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief BaseSolver class declarations
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef BASE_SOLVER_HPP
#define BASE_SOLVER_HPP

#include <functional>
#include "types.hpp"

using namespace Types;

/**
 * @brief Base Solver class. All concrete solver classes must inherit from this class.
 * 
 */
class BaseSolver
{
    public:
        
        /**
         * @brief Construct a new Base Solver object.
         * 
         * @param verbose Show solver's steps flag
         */
        BaseSolver(const bool verbose=false);

        /**
         * @brief Destroy the Base Solver object.
         * 
         */
        virtual ~BaseSolver();
        
        /**
         * @brief Return optimized weights.
         * 
         * @param w Column vector of weights
         * @param X Matrix of feature variables
         * @param y Column vector of target variable
         * @return Weights 
         */
        const Weights optimize(Weights& w, const Features& X, const Target& y);

        /**
         * @brief Get solver's name.
         * 
         * @return std::string 
         */
        const std::string get_name() const;

    protected:

        /**
         * @brief Solver's name (string).
         * 
         */
        std::string name_;

        /**
         * @brief Show solver's steps flag.
         * 
         */
        bool verbose_;
    
    private:
        // Add private members, if needed

};

#endif