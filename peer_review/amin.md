# Peer Review for Amin

I went through your Jupyter notebook for [**Proximal Gradient descent**](https://github.com/anvaribs/SDS385/blob/master/exercises/exercises06/Proximal_Gradient_Method.ipynb). 

In the beginning, you use a paragraph to talk about subgradient method and its lack of true sparsity. Although there are equations that are not typeset correctly on my computer, this is a very good motivation to proximal gradient descent.

The next section of the notebook deals with loading data and normalization. I noticed that you write your own function for normalization. Scikit learn has its implementation of data normalization and standardization in its `preprocessing` module. You might find them handy.

Then you use scikit learn's implementation of Lasso as a benchmark of comparision. I can see that you try to plot in-sample MSE versus different choices of $\lambda$, but the plot is not showing correctly on my computer.

Next, we go to the implementation of the proximal gradient descent algorithm. I like how you use several utility functions to modulize your code. The algorithm is implemented in a concise manner. The result matches with that of the benchmark.

I noticed that the convergence criterion of the proximal gradient descent is a maximum number of iteration. In addition, we could stop, for example, when the objective function does not improve much anymore (a small number `tol`), using a while loop like the following:

```python
while obj_change > tol:
    grad = calc_grad(X,y,beta)
    
    u = beta - gamma * grad
    beta = soft_thresh(u,gamma * lam)
    
    current_obj = calc_obj(X,y,beta,lam)
    obj_change = abs(current_obj - last_obj)
    last_obj = current_obj
    costs.append(current_obj)
```
   
You also implemented Backtracking Line-Search. This is definitely a bonus point since it is not required. I can clearly see in your plot that backtracking line search gives us an advantage in terms of convergence speed. 

The final part of the notebook implements Nesterovs' Accelerated Proximal Gradient method. From your plot we can clearly see that the accelerated method is much faster than the standard one, even with backtracking line-search.

Overall, I think your notebook is comprehensive. It provides a clear comparision of the three methods.


