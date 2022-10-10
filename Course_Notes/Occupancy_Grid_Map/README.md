# OCCUPANCY GRID MAP 

## 5 Minutes Overview
* This is the 2D representation of the environment.
* Look like a map for mobile robots, used for navigation tasks: ex. driving around
* Occupancy Grid Map contains:
  * Grid cells: refer to specific place in the environment. A cell is either occupied or free space. 
* Occupancy Grid Map similar to Grayscale image:
  * Black means that cell is occupied by obstacle. 
  * White means that cell is free space.
* Occupancy Grid Map similar to Architecture Floor plan
* We are not certain if a place in the environment is occupied or free, we can only estimate that. Typically use sensor data
  (noisy sensor data) to do. We maintain for every cell a probability of occupancy or free.
* Occupancy Grid Map is a set of these small cells containing probabilities. 
* Occupancy Grid Map use 3 main assumptions: 
  * A cell is either occupied or free space.
  * Grid Cells are independent to each others.
  * The world is static (nothing move around).
* Use sensor data: 
  * Laser range finder
  * Stereo camera
  * RGB-D camera
  * Sonal sensor

* Use a static state binary Bayes filter to estimate the probability distribution for every place. 
  * If we have high quality sensor 
  (distance is measured very well), we can get very sharp maps in the environment.
     > ![img.png](img.png)
  * If the measure from sensor not well, we have more blurry walls and the map not sharp.
     > ![img_1.png](img_1.png)

* Robots use Occupancy Map for:
  * Navigation
  * Path planning 
  * Localization
  * Mapping 
  * Exploration

* Extended version:
  * Voxel 3D Grid Map 

## Lecture 

### Maps 
* Maps are required for some tasks like localization, planning, ... 
* Task of understanding how the environment looks like 
* Learning maps from sensor data is one of the most fundamental tasks in robotics. 

### Features Maps vs Volumetric Maps

* Features-based map store the location of distinct points in the environment (Not used in this lecture)
* Volumetric maps (which included in the lecture)

### Description of Mapping Task 
* Compute the most likely map given the sensor data:
  > ![img_2.png](img_2.png)
  > 
  > m: map 
  > u: odometry commands
  > z: observation data 

* Today, we see **how to compute the map given the robot's pose** (replace the commands with the robot positions x - this can be achieved by 
installed a very good GPS IMU which direct provide us the pose estimation, then we can use this pose information to build the map of the environment.)
  > ![img_3.png](img_3.png)

### Grid Maps 
* Map models occupancy of the space
* Discretize the world into cells
* Grid structure is rigid 
* Each cell is assumed to be occupied or free space 
* Non parametric model (Not like the feature-based map no assumption about the features, it just a binary random variable)
* Large maps may require substantial memory resources (this is one issue, for example sparse space in 3D will consume more memory !!!)
* Do not rely on feature detector (this is for sure)

### Three assumptions with Grid Maps
#### Assumption 1
> The area that corresponds to a cell is either completely free or occupied. (binary values)
> 
> ![img_4.png](img_4.png)
> 
> Each cell is a **binary random variable** that models the occupancy
>
> ![img_5.png](img_5.png)
> * Cell is occupied p(m_i) = 1
> * Cell is free p(m_i) = 0
> * No knowledge p(m_i) = 0.5
>
> We have some notations: 
> 
> ![img_6.png](img_6.png)
> When we estimating the environment, than we can have numbers between 0 or 1 represent the probability for being occupied  (Hum, this can be ambiguous)


> NOTE:
> 
> * In reality, there something which doesn't really hold because we can have objects which are smaller than the grid size.
> * Virtual grid structure in some cases can not represent 100% correct the obstacles like wall but with for example rotated 45 degrees. There will be cells
> which partly occupied 


#### Assumption 2 

> The world is **static** (most mapping systems make this assumption): this means while record the sensor data, the world doesn't change.

#### Assumption 3

> The cells (random variables) are **independent** to each other. This means given the probability of one cell, we can not interpret the probability of 
> the neighbor cells. We use this to simplify the problem, in reality, this assumption may or may not be valid.

> **NOTE**: 
> * The designer of the Algorithm make the decision to the assumption.
> * We should revisit these assumptions when the system breaks: if the mapping fail at some points

### Joint Distribution 

> ![img_9.png](img_9.png)
> 
> * A map represent a joint probability distributions. The probability distribution about the map is the joint belief about the values of individual cells
> * Due to the definition of the joint probability, we want to simplify the potentially large equation by exploiting the assumption  - all cells are independent to each other, then:  
>  > ![img_10.png](img_10.png)
>  > Example:
>  > 
>  > ![img_11.png](img_11.png)
> * We can use the Assumption 1 to store just one probability and then use it to compute:
>  > ![img_12.png](img_12.png)
>      
 
### Estimating a Map From Data 

> Given sensor data z(1:t) (**observations**) and the poses x(1:t) (**states**) of the sensor, estimate the map: 
> ![img_14.png](img_14.png)
> a. Apply the Static State Binary Bayes Filter for each cell
> ![img_15.png](img_15.png)
> b. Apply the Markov's Assumption: 
> ![img_16.png](img_16.png)
> c. Again apply the Bayes's Rule: 
> ![img_17.png](img_17.png)
> ![img_18.png](img_18.png)
> d. Assuming independence: 
> ![img_19.png](img_19.png)
> (Probability for the cell of being occupied)
> 
> Do the same for the probability of the cell being free:
> ![img_20.png](img_20.png)
> 
> Computing the ratio of both probabilities, and eliminate the same terms in nominator and denominator:  
> 
> ![img_21.png](img_21.png)
> 
> Using the binary probability property: 
> 
> ![img_23.png](img_23.png)
> 
> * First term use the current observation z_t (what's the robot is seeing right now)
> * Second term is the recursive term, it only related to state estimate of that same cell, but using only old data up to time 
> t - 1
> Third is the prior information: what can I say about my map without having seen anything, what is my prior assumption about occupancy

> Convert from Ratio into Probability
> * We can convert this ratio (**odds ratio**) into probability (bellow is the proof): 
>  ![img_24.png](img_24.png)
> 
> (Given the odds ratio of an events, we can then calculate the probability for that event)
> 
> * Plugging this into the equation: 
> ![img_25.png](img_25.png)
> 
> **For reasons of efficiency, one performs the calculations in the log odds notation**
> 
> Notation:
> ![img_26.png](img_26.png)
> 
> Log Odd Notation:
> 
> ![img_27.png](img_27.png)
> 
> Then the Occupancy Mapping in Log Odds Form like this: 
> 
> ![img_28.png](img_28.png)
> 
> Occupancy Mapping Algorithm 
> 
> ![img_29.png](img_29.png)
> 
> NOTE: We can make this fast by breaking the maps into small chunks, and each chunks is handled with different CPU. Because this process is parallel for each cell.
> * Moravec and Elfes proposed occupancy grid mapping in the 80's 
> * Developed for noisy sonal sensors
> * Also called "mapping with known poses"
>
> 
> 

# References 
- [Wiki Bernoulli distribution - related to binary random variable](https://en.wikipedia.org/wiki/Bernoulli_distribution)
- [Wiki Joint probability distribution](https://en.wikipedia.org/wiki/Joint_probability_distribution)
  > ![img_8.png](img_8.png)
  > ![img_7.png](img_7.png) 
- [Wiki - Odd ratio](https://en.wikipedia.org/wiki/Odds_ratio)