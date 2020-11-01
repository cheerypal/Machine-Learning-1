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
| Top10Pixels | K2 (1 Parent) | |
| Top10Pixels | K2 (2 Parent) | |
| Top10Pixels | TAN | |
| Top10Pixels | HillClimber | |
| Top20Pixels | K2 (1 Parent) | 67.6161% |
| Top20Pixels | K2 (2 Parent) | |
| Top20Pixels | TAN | |
| Top20Pixels | HillClimber | |


---

# Graphs

- top5pixels - K2 with 1 max parent

![top5pixels_k2_1](./bayes_net_graphs/top5Pixels_k2_1.png)

- top5pixels - K2 with 2 max parents

![top5pixels_k2_2](./bayes_net_graphs/top5Pixels_k2_2.png)

- top5pixels - TAN

![top5pixels_TAN](./bayes_net_graphs/top5Pixels_TAN.png)

- top5pixels - HillClimber

![top5pixels_HillClimber](./bayes_net_graphs/top5Pixels_HillClimber.png)

---

- top10pixels - K2 with 1 max parent

![top10pixels_k2_1](./bayes_net_graphs/top10Pixels_k2_1.png)

- top10pixels - K2 with 2 max parents

![top10pixels_k2_2](./bayes_net_graphs/top10Pixels_k2_2.png)

- top10pixels - TAN

![top10pixels_TAN](./bayes_net_graphs/top10Pixels_TAN.png)

- top10pixels - HillClimber

![top10pixels_HillClimber](./bayes_net_graphs/top10Pixels_HillClimber.png)

---

- top20pixels - K2 with 1 max parent

![top20pixels_k2_1](./bayes_net_graphs/top20Pixels_k2_1.png)

- top20pixels - K2 with 2 max parents

![top20pixels_k2_2](./bayes_net_graphs/top20Pixels_k2_2.png)

- top20pixels - TAN

![top20pixels_TAN](./bayes_net_graphs/top20Pixels_TAN.png)

- top20pixels - HillClimber

![top20pixels_HillClimber](./bayes_net_graphs/top20Pixels_HillClimber.png)
