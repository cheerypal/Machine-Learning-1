# F20DL CW1

## Q7

Classifiers being used on the datasets from question 5 (all on the training sets):

1. BayesNet with K2 algorithm (Max number of parents = 1) and SimpleEstimator
2. BayesNet with K2 algorithm (Max number of parents = 2) and SimpleEstimator
3. BayesNet with TAN algorithm and SimpleEstimator
4. BayesNet with HillClimber and SimpleEstimator 


| Dataset | Test | Correctly Classified Instances |
| :-------: | :----: | :-------------------------------:|
| Top5Pixels | K2 (1 Parent) | 68.2353% |
| Top5Pixels | K2 (2 Parent) | 97.6161% |
| Top5Pixels | TAN | 99.7317% |
| Top5Pixels | HillClimber | 68.2353% |
| Top10Pixels | K2 (1 Parent) | 68.1321% |
| Top10Pixels | K2 (2 Parent) | 97.1414% |
| Top10Pixels | TAN | 99.1331% |
| Top10Pixels | HillClimber | 68.1321% |
| Top20Pixels | K2 (1 Parent) | 67.6161% |
| Top20Pixels | K2 (2 Parent) | n/a |
| Top20Pixels | TAN | n/a |
| Top20Pixels | HillClimber | n/a |



