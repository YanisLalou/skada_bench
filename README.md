# skada_bench

## TO USE:
- Need the branch of skada with the NhanesLeadDataset (or just remove it from the experiments.yaml + remove ./datasets/nhanes_lead_dataset.py file)
- Pip install requirements.txt
- Pip install skada
- Set experiments in the experiments.yaml (Should work with sklearn estimators & scorers + the one defined in ./estimators & ./scorers)
- Python launch_experiments.py (To launch experiments)
- Python generate_table_results.py --display (To display results in table + save results as result_table.tex)

## ISSUES:
- We need to make sure that the dataset is compatible with the estimators and scorer. For example, if the dataset is a classification dataset, then the scorer should be a classification scorer.
- We cant automatically launch experiments with every possible combination of estimators/scorers. We need to manually specify the combinations we want to try.
- We dont handle ConvergenceWarning from sklearn during the optimisation.
- ...
