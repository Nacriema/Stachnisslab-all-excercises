# Kalman Filter and Extended Kalman Filter (EKF)

## Kalman Filter
* This may use for the same purpose of Least-Square Method (because Cyrill said the least square approach as well as the
 KF will provide the same result)
* Kalman Filter is the Bayes filter for the Gaussian linear case (assume world is Gaussian distribution)
```text
These things I quite not understand: 
* Performs recursive state estimation 
* Prediction step to exploit the controls 
* Correction step to exploit the observations 

Take the "original belief" to the "new belief"

```

## Kalman Filter Example

* Assume we are on the boat somewhere on the sea at time t, then we make a prediction of our location at time delta(t) (
the prediction gave without any observation or any prior information.
  )
* At that time, we get an observation (ex a lighthouse) and the observation tell that we are at the another position 
* The kalman filter computes a weighted sum of the prediction where you think you went and what the observation tell you 
at where. Then we get the position (this is called the belief)
* We basically trade off these uncertainties and then can trust the observation more or trust the prediction more depending
on the current estimates. 

## Bayes Filter for State Estimation

## Kalman Filter 
* KF is a Bayes Filter
* Assume every thing is Gaussian


## Linear Models
* Models can be expressed through a linear function: 
```text
f(x) = Ax +b 

x can be multidimensional variable, and b is the constance term
```
* A Gaussian that is transformed through a linear function stays Gaussian

* This is said that the Optimal Solution for Linear models and with Gaussian distribution




## MORE RESOURCES
* Bayes Filter: https://johnwlambert.github.io/bayes-filter/
* Jacobian Matrix: 
  * Definition: https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant
  * Example: https://www.mathworks.com/help/symbolic/sym.jacobian.html
* Gaussian distribution: https://en.wikipedia.org/wiki/Normal_distribution
* Marginal distribution: https://en.wikipedia.org/wiki/Marginal_distribution
* Gaussian distribution over 2D data and the props: https://fabiandablander.com/statistics/Two-Properties.html
* Kalman Filter used in Online machine learning: https://en.wikipedia.org/wiki/Online_machine_learning
* Convolution and the properties - one is Linear transform: https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1
* Object Falling in Air - Kalman Filter: https://courses.cs.washington.edu/courses/cse466/12au/calendar/15b-StateEstimation.pdf
* Nice Resource of Kalman-Filter: http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
* My questions about Kalman-Filter:
  * https://dsp.stackexchange.com/questions/27484/observation-matrix-in-kalman-filter
  * https://towardsdatascience.com/5-things-you-should-know-about-covariance-26b12a0516f1
  
## TODO: 
- [ ] Complete note about Kalman Filter.
- [ ] Add some cases when we use Kalman Filter (see from the internet).
- [ ] Read some papers that used Kalman Filter.