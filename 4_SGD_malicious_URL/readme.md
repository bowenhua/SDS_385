# Mini-Batch SGD applied to a [malicious URL dataset](http://archive.ics.uci.edu/ml/datasets/URL+Reputation)

I implemented a logistic regression model with l-1 regularization for identifying malicious URL. The model is trained by mini-batch stochastic gradient descent. 

#### Reading the data

[Read.py](https://github.com/bowenhua/SDS_385/blob/master/4_SGD_malicious_URL/read.py) loads the original dataset (stored in svmlight format) using the `load_svmlight_file` function provided by Scikit-learn, and stores the data in scipy CSR sparse matrix format. 

This dataset has more than three million features and two million samples, and repetitive reading is time-consuming. Therefore, I store the preprocessed data in NumPy and SciPy binary files. Reading the binary files only takes less than ten seconds on my laptop.

#### Model

[My code](https://github.com/bowenhua/SDS_385/blob/master/4_SGD_malicious_URL/better_sgd_minibatch.py) uses a l-1 regularization with a penalty of 1e-5 (the negative log likelihood in my objective function is weighted by the number of samples). Optimization is performed by AdaGrad. 

Since the feature matrix is stored in a sparse format, the matrix-vector multiplication between the feature matrix and the coefficient vector (beta) is fast. We could exploit sparsity even more by implementing lazy updating on the coefficient vector. However, when we add a SciPy sparse matrix to a dense NumPy matrix, SciPy first converts the sparse matrix to dense format, and then preforms the addition. This prevents us from exploiting sparsity in vector addition.

Instead, I use a mini-batch SGD in which the batch size is pretty large (65536). Since we are not implementing lazy update, a large batch size is chosen to utilize the fast linear algebra implemented in NumPy (LAPACK). 

#### Results

AdaGrad is run for 300 epochs and the total training CPU time (excluding data reading) is ~1700s. That is about 5.7s per epoch. 

Without any further hyperparameter tuning, the training accuracy I get is 98.83% and the testing accuracy is 98.67%.





