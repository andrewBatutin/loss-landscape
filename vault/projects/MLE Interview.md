# Linear Regression

y_hat = X @ w + b 
loss = np.mean((y - y_hat) **2)

### Closed Form

```python

X_b = np.hstack([np.ones((n, 1)), X]) # add bias column

w = np.linalg.solve(X_b.T @ X_b, X_b.T @ y)

```

  

### Gradient Descent

```python

y_pred = X @ w + b

error = y_pred - y

dw = (2/n) * X.T @ error

db = (2/n) * np.sum(error)

w -= lr * dw

b -= lr * db

```

  

### Ridge

```python

w = np.linalg.solve(X_b.T @ X_b + alpha * I, X_b.T @ y)

```

  

### Metrics

```python

MSE = np.mean((y - y_pred) ** 2)

R2 = 1 - SS_res / SS_tot

```

## Questions 

Assumptions of Linear Regression

1. **Linearity**: y is linear combination of features

2. **Independence**: observations are independent

3. **Homoscedasticity**: constant variance of errors

4. **Normality**: errors are normally distributed

5. **No multicollinearity**: features not highly correlated



https://buildml.substack.com/p/i-messed-up-my-amazon-interview-lets?r=cii5h&utm_medium=ios&triedRedirect=true


[[concepts/Linear Regression]]