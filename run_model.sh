#!/bin/bash

echo "Processing started!" 

args=()
while [ "$1" != "" ]; do
    args+=("$1")
    shift
done

echo "Input taken!"
python top.py "${args[@]}"
echo "Shell script successfully terminated!"