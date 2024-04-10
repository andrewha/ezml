/**
 * @file qr_solver.cpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief QRSolver class implementation
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "qr_solver.hpp"
#include "predict_functions.hpp"

QRSolver::QRSolver()
{
    // Init solver here
    // Set solver's name as string representation of its type
    name_ = Types::get_name(*this);
}

const Weights QRSolver::optimize(Weights& w, const Features& X, const Target& y)
{
    const Weights weights = Predict::qr(X, y);
    return weights;
}
