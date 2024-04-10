/**
 * @file ols_solver.cpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief OLSSolver class implementation
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "ols_solver.hpp"
#include "predict_functions.hpp"

OLSSolver::OLSSolver()
{
    // Init solver here
    // Set solver's name as string representation of its type
    name_ = Types::get_name(*this);
}

const Weights OLSSolver::optimize(Weights& w, const Features& X, const Target& y)
{
    const Weights weights = Predict::ols(X, y);
    return weights;
}
