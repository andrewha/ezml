/**
 * @file exceptions.hpp
 * @author Andrei Batyrov (arbatyrov@edu.hse.ru)
 * @brief Custom exceptions declarations and implementation
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef EXCEPTIONS_HPP
#define EXCEPTIONS_HPP

#include <string>

/**
 * @brief NotFittedException class. Inherits from std::exception class.
 * 
 */
class NotFittedException : public std::exception
{
    public:
    
        /**
         * @brief Construct a new NotFittedException object.
         * 
         * @param obj_name Object's name
         */
        NotFittedException(const std::string& obj_name)
        : obj_name_(obj_name)
        { }

        /**
         * @brief Return detailed description of exception.
         * 
         * @return const std::string 
         */
        const std::string what()
        {
            const std::string message = "\n\033[91mNotFittedException: \033[33m" + obj_name_ + "\033[0m must be fitted first\n";
            return message;
        }
    
    private:

        /**
         * @brief Object's name.
         * 
         */
        std::string obj_name_;

};

/**
 * @brief NewtonShapeException class. Inherits from std::exception class.
 * 
 */
class NewtonShapeException : public std::exception
{
    public:
    
        /**
         * @brief Construct a new NewtonShapeException object.
         * 
         */
        NewtonShapeException()
        { }

        /**
         * @brief Return detailed description of exception.
         * 
         * @return const std::string 
         */
        const std::string what()
        {
            const std::string message = "\n\033[91mNewtonShapeException: \033[33mX must have one feature only\033[0m\n";
            return message;
        }
    
    private:

        // Add private member, if needed

};

#endif
