# Linear Regression Review 

A linear model makes a prediction by simply computing a weighted
sum of the input features, plus a constant called the bias term (also called the intercept
term):

```
y=θ 0 +θ 1 x 1 +θ 2 x 2 +⋯+θnxn
```
- ŷ is the predicted value.
- n is the number of features.
- xi is the ith feature value.
- θj is the jth model parameter (including the bias term θ 0 and the feature weights
    θ 1 , θ 2 , ⋯, θn).

This can be written much more concisely using a vectorized form:
```
y=hθ x =θT· x
```
- θ is the model’s parameter vector, containing the bias term θ 0 and the feature
    weights θ 1 to θn.
- θT is the transpose of θ (a row vector instead of a column vector).
- **x** is the instance’s feature vector, containing x 0 to xn, with x 0 always equal to 1.
- θT · **x** is the dot product of θT and **x**.
- hθ is the hypothesis function, using the model parameters θ.

Okay, that’s the Linear Regression model, so now how do we train it? 

The most common performance measure
of a regression model is the Root Mean Square Error (RMSE). Therefore, to train a Linear Regression model, you need to find the value of θ that minimizes the RMSE. In practice, it is simpler to minimize the Mean Square Error (MSE)
than the RMSE, and it leads to the same result (because the value that minimizes a
function also minimizes its square root).^1

# Gradient Descent

Gradient Descent is a very generic optimization algorithm capable of finding optimal
solutions to a wide range of problems. The general idea of Gradient Descent is to
tweak parameters iteratively in order to minimize a cost function.

Suppose you are lost in the mountains in a dense fog; you can only feel the slope of
the ground below your feet. A good strategy to get to the bottom of the valley quickly
is to go downhill in the direction of the steepest slope. This is exactly what Gradient
Descent does: it measures the local gradient of the error function with regards to the
parameter vector θ, and it goes in the direction of descending gradient. Once the gra‐
dient is zero, you have reached a minimum!

Concretely, you start by filling θ with random values (this is called random initializa‐
tion), and then you improve it gradually, taking one baby step at a time, each step
attempting to decrease the cost function (e.g., the MSE), until the algorithm converges
to a minimum (see Figure 4-3).

An important parameter in Gradient Descent is the size of the steps, determined by
the learning rate hyperparameter. If the learning rate is too small, then the algorithm
will have to go through many iterations to converge, which will take a long time.

On the other hand, if the learning rate is too high, you might jump across the valley
and end up on the other side, possibly even higher up than you were before. This
might make the algorithm diverge, with larger and larger values, failing to find a good
solution.

```
When using Gradient Descent, you should ensure that all features
have a similar scale (e.g., using Scikit-Learn’s StandardScaler
class), or else it will take much longer to converge.
```

This diagram also illustrates the fact that training a model means searching for a
combination of model parameters that minimizes a cost function (over the training
set). It is a search in the model’s parameter space: the more parameters a model has,
the more dimensions this space has, and the harder the search is: searching for a nee‐
dle in a 300-dimensional haystack is much trickier than in three dimensions. Fortu‐
nately, since the cost function is convex in the case of Linear Regression, the needle is
simply at the bottom of the bowl.

## Batch Gradient Descent

To implement Gradient Descent, you need to compute the gradient of the cost func‐
tion with regards to each model parameter θj. 

Notice that this formula involves calculations over the full training
set X , at each Gradient Descent step! This is why the algorithm is
called Batch Gradient Descent: it uses the whole batch of training
data at every step. As a result it is terribly slow on very large train‐
ing sets (but we will see much faster Gradient Descent algorithms
shortly). However, Gradient Descent scales well with the number of
features; training a Linear Regression model when there are hun‐
dreds of thousands of features is much faster using Gradient
Descent than using the Normal Equation.

```
see python notebook
```

On the left, the learning rate is too low: the algorithm will eventually reach the solu‐
tion, but it will take a long time. In the middle, the learning rate looks pretty good: in
just a few iterations, it has already converged to the solution. On the right, the learn‐
ing rate is too high: the algorithm diverges, jumping all over the place and actually
getting further and further away from the solution at every step.

To find a good learning rate, you can use grid search. However, you
may want to limit the number of iterations so that grid search can eliminate models
that take too long to converge.

You may wonder how to set the number of iterations. If it is too low, you will still be
far away from the optimal solution when the algorithm stops, but if it is too high, you
will waste time while the model parameters do not change anymore. A simple solu‐
tion is to set a very large number of iterations but to interrupt the algorithm when the
gradient vector becomes tiny—that is, when its norm becomes smaller than a tiny
number ε (called the tolerance)—because this happens when Gradient Descent has
(almost) reached the minimum.

## Stochastic Gradient Descent

The main problem with Batch Gradient Descent is the fact that it uses the whole
training set to compute the gradients at every step, which makes it very slow when
the training set is large. At the opposite extreme, Stochastic Gradient Descent just
picks a random instance in the training set at every step and computes the gradients
based only on that single instance. Obviously this makes the algorithm much faster
since it has very little data to manipulate at every iteration. It also makes it possible to
train on huge training sets, since only one instance needs to be in memory at each
iteration.

This code implements Stochastic Gradient Descent using a simple learning schedule:

```
n_epochs = 50
t0, t1 = 5, 50 # learning schedule hyperparameters
```
```
def learning_schedule(t):
return t0 / (t + t1)
```
```
theta = np.random.randn(2,1) # random initialization
```
```
for epoch in range(n_epochs):
for i in range(m):
random_index = np.random.randint(m)
xi = X_b[random_index:random_index+1]
yi = y[random_index:random_index+1]
gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
eta = learning_schedule(epoch * m + i)
theta = theta - eta * gradients
```
By convention we iterate by rounds of m iterations; *each round is called an epoch.*

## Mini-batch Gradient Descent

The last Gradient Descent algorithm we will look at is called Mini-batch Gradient
Descent. It is quite simple to understand once you know Batch and Stochastic Gradi‐
ent Descent: at each step, instead of computing the gradients based on the full train‐
ing set (as in Batch GD) or based on just one instance (as in Stochastic GD), Mini-

batch GD computes the gradients on small random sets of instances called mini-
batches. The main advantage of Mini-batch GD over Stochastic GD is that you can
get a performance boost from hardware optimization of matrix operations, especially
when using GPUs.

The algorithm’s progress in parameter space is less erratic than with SGD, especially
with fairly large mini-batches. As a result, Mini-batch GD will end up walking
around a bit closer to the minimum than SGD. But, on the other hand, it may be
harder for it to escape from local minima (in the case of problems that suffer from
local minima, unlike Linear Regression as we saw earlier). Figure 4-11 shows the
paths taken by the three Gradient Descent algorithms in parameter space during
training. They all end up near the minimum, but Batch GD’s path actually stops at the
minimum, while both Stochastic GD and Mini-batch GD continue to walk around.
However, don’t forget that Batch GD takes a lot of time to take each step, and Stochas‐
tic GD and Mini-batch GD would also reach the minimum if you used a good learn‐
ing schedule.


