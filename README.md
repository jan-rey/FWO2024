The respiratory contains multiple scripts. Each script is dedicated to the meta-learning of a different parameter in a Rescorla Wagner model.
Epsilon to meta-learn epsilon in an epsilon-greedy decision policy
Learning rate to meta-learn learning rate in the delta update rule
Lambda to meta-learn an unchosen action value addition
Lambda-mult to meta-learn an unchosen action value multiplication
Temperature to meta-learn temperature in a Softmax decision policy

Eac hscript simulates the meta-learning model in three environments: a stable, a volatile and an adversarial environment.
