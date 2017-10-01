# Peer Review for Ashutosh

## General Comments
I went through the files you have in the [matrix_inversion](https://github.com/ASinghh/matrix_inversion) folder. I like the fact that you are using Jupyter Notebook. Compared to `.py` files your format makes it easier to show codes, results, and comments in a streamlined manner. I might try to use Jupyter Notebook in the future.

Also your codes are well commented. You have comments for each block, which makes the reviewer easier to understand the notebook. Another plus is that you use plot to visualize your results.

One improvement you could have for your codes is that the $X^TWX$ matrix we want to invert has a structure of being **positive semidefinite**, while you generate random unstructured matrix. This suggestion has been mentioned in the class.

More detailed comments to follow. 

Although I find the codes, I cannot find files for other problems in the exercise.

## [Jupyter Notebook](https://github.com/ASinghh/matrix_inversion/blob/master/Comparision%2Bof%2BMatrix%2BInversion%2Bmethods.ipynb)
I will include my detailed comments of the Jupyter Notebook in this section.

In the first code block, you generated random unstructured matrix. To generate structured matrix, you could use the following function, and then compute $X^TWX$:

```python
def initialize(n, p):
    X = np.random.rand(n,p)
    W_vector = np.random.rand(n,1) #If we want to diagnolize it we cannot define W as (p,1) matrix
    y = np.random.rand(n,1)
    return (X,y, W_vector)
```

Then, you compared the following three methods:

1. direct inversion
2. LU decomposition
3. Built-in linsolve method

With a positive semidefinite matrix, you might be able to use more efficient decomposition methods such as Choleskey. Your first plot shows that the Built-in solver has the fastest performance. It's interesting that matrix inversion is faster than LU decomposition in your results.

The last part of your codes contains exploiting sparsity. In the following code block:

```python
##Exploring sparsity
P = 5000
spar = []
inb  = []
for i in range(10):
    k = 0.05 + i*0.1
    spar.append(k)
    a = scipy.sparse.rand(P,P, density= k, format='coo', dtype=None, random_state=None)
    A = scipy.sparse.csr_matrix.todense(a)
    b = np.random.rand(P,1)
    inb.append(fn(np.linalg.solve,A,b))
```

In the line where you use the `scipy.sparse.csr_matrix.todense()` function, the sparse matrix is converted to a regular dense matrix and `np.linalg.solve()` cannot exploit sparsity. My suggestion is to use `sp.sparse.linalg.lsqr()`, which is a function that exploits sparsity. There might be other (better) choices.





