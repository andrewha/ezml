/**
 * @file autoreg_extractor.cpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief AutoRegExtractor class implementation
 * @version 0.1
 * @date 2024-04-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <armadillo>
#include "autoreg_extractor.hpp"
#include "exceptions.hpp"

AutoRegExtractor::AutoRegExtractor(const size_t p)
: p_(p)
{
    // Init extractor here
    // Set extractor's name as string representation of its type
    name_ = Types::get_name(*this);
}

const Features AutoRegExtractor::extract_X(TimeSeries& process)
{
    // Throw if p is too small or too large
    if (p_ < 1 || p_ >= process.n_rows)
        throw WrongOrderException();

    // Extract features
    Features X = process.head(process.n_rows - p_);
    
    for (size_t order = 1; order < p_; ++order)
    {    
        X.insert_cols(0, process.rows(order, process.n_rows - p_ + order - 1));
        
    }
    
    return X;
}

const Target AutoRegExtractor::extract_y(TimeSeries& process)
{
    // Throw if p is too small or too large
    if (p_ < 1 || p_ >= process.n_rows)
       throw WrongOrderException();

    // Extract target
    return process.tail(process.n_rows - p_);
}

const std::string AutoRegExtractor::get_name() const
{
    return name_;
}
