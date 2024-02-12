#!/bin/bash

echo 'This bash script is used to install the dependencies and launch the experiments.'

# TODO: install dependencies
# pip install -r requirements.txt

while getopts d:e:s:r:v flag
do
    case "${flag}" in
        d) datasets=${OPTARG};;
        e) estimators=${OPTARG};;
        s) scorers=${OPTARG};;
        r) rerun=${OPTARG};;
        v) venv_path=${OPTARG};;
    esac
done

echo "datasets: $datasets";
echo "estimators: $estimators";
echo "scorers: $scorers";
echo "rerun: $rerun";
echo "venv_path: $venv_path";


# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv ${venv_path}
fi

# Activate virtual environment
source ${venv_path}/bin/activate

# Install dependencies using pip
pip install -r requirements.txt

# Execute the Python command
python3.10 ./launch_experiments.py -d ${datasets} -e ${estimators} -s ${scorers} -r ${rerun}

python3.10 ./generate_table_results.py --display

# Deactivate virtual environment
deactivate