#!/usr/bin/bash

EXAMPLE=$1
TARGET="./build/$EXAMPLE"
LIB="../bin/static"
SRC="$EXAMPLE.cpp"

mkdir -p ./build

echo -e "Building the example: \033[93m$TARGET\033[0m"

# -O3 -- max speed optimization

CFLAGS="-fdiagnostics-color=always -g -O3 -std=c++17"
INCLUDE="../include"

if g++ $CFLAGS -I $INCLUDE $SRC -o $TARGET -L$LIB -lezml -larmadillo -llapack;
    then echo -e "\033[92mSuccessfully built the example: \033[93m$TARGET\033[0m"; 
    else echo -e "\033[91mError!\033[0m";
fi
