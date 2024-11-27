#!/usr/bin/bash

echo "Building the library object files..."

mkdir -p ./bin/static
TARGET="./bin/static"

# -O3 -- max speed optimization

CFLAGS="-fdiagnostics-color=always -c -g -O3 -ffast-math -std=c++17 -Wall"
INCLUDE="./include"
SRC="./src"

if
    g++ $CFLAGS -I $INCLUDE $SRC/base_model.cpp -o $TARGET/base_model.o -larmadillo -llapack;
    g++ $CFLAGS -I $INCLUDE $SRC/linreg_model.cpp -o $TARGET/linreg_model.o -larmadillo -llapack;
    g++ $CFLAGS -I $INCLUDE $SRC/logreg_model.cpp -o $TARGET/logreg_model.o -larmadillo -llapack;
    g++ $CFLAGS -I $INCLUDE $SRC/base_solver.cpp -o $TARGET/base_solver.o -larmadillo -llapack;
    g++ $CFLAGS -I $INCLUDE $SRC/ols_solver.cpp -o $TARGET/ols_solver.o -larmadillo -llapack;
    g++ $CFLAGS -I $INCLUDE $SRC/qr_solver.cpp -o $TARGET/qr_solver.o -larmadillo -llapack;
    g++ $CFLAGS -I $INCLUDE $SRC/derivative_solver.cpp -o $TARGET/derivative_solver.o -larmadillo -llapack;
    g++ $CFLAGS -I $INCLUDE $SRC/base_transformer.cpp -o $TARGET/base_transformer.o -larmadillo -llapack;
    g++ $CFLAGS -I $INCLUDE $SRC/standard_scaler.cpp -o $TARGET/standard_scaler.o -larmadillo -llapack;

then echo -e "\033[92mSuccessfully built the object files\033[0m"; 
    else echo -e "\033[91mError!\033[0m";
fi

echo "Creating the static library"
if
    ar rcs $TARGET/libezml.a $TARGET/base_model.o \
                             $TARGET/linreg_model.o \
                             $TARGET/logreg_model.o \
                             $TARGET/base_solver.o \
                             $TARGET/ols_solver.o \
                             $TARGET/qr_solver.o \
                             $TARGET/derivative_solver.o \
                             $TARGET/base_transformer.o \
                             $TARGET/standard_scaler.o;
then echo -e "\033[92mSuccessfully built the static library $TARGET/libezml.a\033[0m"; 
    else echo -e "\033[91mError!\033[0m";
fi
