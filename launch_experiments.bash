#!/bin/bash

echo 'This bash script is used to install the dependencies and launch the experiments.'

# TODO: install dependencies
# pip install -r requirements.txt

while getopts d:e:s:r: flag
do
    case "${flag}" in
        d) datasets=${OPTARG};;
        e) estimators=${OPTARG};;
        s) scorers=${OPTARG};;
        r) rerun=${OPTARG};;
    esac
done

echo "datasets: $datasets";
echo "estimators: $estimators";
echo "scorers: $scorers";
echo "rerun: $rerun";

python3.10 ./launch_experiments.py -d ${datasets} -e ${estimators} -s ${scorers} -r ${rerun}
