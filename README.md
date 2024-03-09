# Mini project: KNN classifier

**Libraries used: pandas, numpy.**

**All datasets are stored in 'datasets' folder.**

**main.py** -  takes 3 arguments:
* k: positive natural number being the k-NN hyperparameter.
* train-set: name of the file containing the training set in csv format.
* test-set: name of the file containing the test set.

Program applies KNN classifying algorithm based on the train set to each vector from the test set and produces the accuracy (proportion of correctly classified examples from the test set).
The program additionally provides a CLI to enable the user to input single vectors to be classified.

**main.ipynb** - presentation of logic and actions + k-accuracy plot.

**_classes.py** - classes that are used in the program:
* KNN - K-nearest Neighbour classifying algorithm.
* Metrics - set of functions to estimate a model (KNN).
