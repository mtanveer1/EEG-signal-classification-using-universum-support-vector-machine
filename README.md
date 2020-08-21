# EEG-signal-classification-using-universum-support-vector-machine

This is implementation of the paper: B. Richhariya, M. Tanveer, EEG signal classification using universum support vector machine, Expert Systems With Applications, Volume 106, 2018, pp. 169-182, https://doi.org/10.1016/j.eswa.2018.03.053.

Description of files:

USVM.m: selecting parameters of USVM using k fold cross-validation. One can select parameters c, mu and e to be used in grid-search method.

rutsvm.m: implementation of RUTSVM-CIL algorithm. Takes parameters c, mu, e, and training data and test data, and provides accuracy obtained and running time.

For running the USVM algorithm, we have included the wpbc dataset. One can simply run the USVM.m file to check the obtained results on this sample dataset. To run experiments on more datasets, simply add datasets in the folder and run USVM.m file.

This code is for non-commercial and academic use only. Please cite the following paper if you are using this code.

Reference: B. Richhariya, M. Tanveer, EEG signal classification using universum support vector machine, Expert Systems With Applications, Volume 106, 2018, pp. 169-182, https://doi.org/10.1016/j.eswa.2018.03.053.

For further details regarding working of algorithm, please refer to the paper.

If further information is required you may contact on: phd1701241001@iiti.ac.in.
