/**
 * @file base_solver.cpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief BaseSolver class implementation
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "base_solver.hpp"

BaseSolver::BaseSolver(const bool verbose /*=false*/)
: verbose_(verbose)
{
    // Init solver here
    // Set solver's name as string representation of its type
    name_ = Types::get_name(*this);
}

BaseSolver::~BaseSolver() {}

const Weights BaseSolver::optimize(Weights& w, const Features& X, const Target& y)
{
    // BaseSolver knows nothing about how to optimize weights
    // So, basically return back the original weights
    return w;

}

const std::string BaseSolver::get_name() const
{
    return name_;
}
