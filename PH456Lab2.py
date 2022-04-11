"""
PH456 Lab 2
"""
#Importing modules
import numpy as np
import matplotlib.pyplot as plt



###############
#Task1
###############

#Defining a function to do integration calculations using Monte-Carlo sampling, where the inputs are
#f = integrand, a = lower boundary, b = upper boundary, dim = dimensions of the integrand, N = number of samples
def monte_carlo (f, a, b, dim, N):

    #Using an if else statement to distinguish 1-dimensional integrals to multi-dimensional integrals, and
    #generates an array/matrix of uniformly distributed pseudo-random numbers between a and b.
    #The size of the array/matrix is N rows, and dim columns.
    if dim == 1:
        ran_nums= np.random.uniform(a, b, N) 
    else: 
        ran_nums= np.random.uniform(a, b, (N,dim)) 

    #Declaring both f_x (the evaluated integrand with a number from ran_nums as the independent value) and f_x_sq (the integrand squared) as 0
    f_x = 0
    f_x_sq = 0
    
    #Using a for loop to sum up the calculated values of f_x and f_x_sq for each value of ran_nums
    for i in ran_nums:
        f_x += f(i)
        f_x_sq += (f(i))**2
    
    #Calculating the mean value of f_x and f_x_sq by dividing them by the number of samples
    mean_f = f_x/N
    mean_f_sq = f_x_sq/N
    
    integral = np.prod(b-a)* mean_f   #Solving the integral using the Monte-Carlo approximated integration formula

    variance = abs((1/N)*((mean_f_sq) - (mean_f**2)))   #Calculating the variance of the solution

    error = np.sqrt(variance)   #Calculating the error of the solution
    
    return integral, error   #The function returns an estimated answer of the calculation and the error of the answer



###############
#Task2
###############

#Defining the given integrands from this assignment
def f_2a(x):
    return 2 
def f_2b(x):
    return -x
def f_2c(x):
    return x**2
def f_2d(v):      #The dependent value (v) of the f_2d integrand is set as a 2-dimensional vector for the multi-dimensional integrand
    x, y = v
    return (x*y) + x

#Using the defined monte_carlo function and appropriate inputs to solve the four given integral problems
Q2a = monte_carlo(f_2a, 0, 1, 1, 100000)
Q2b = monte_carlo(f_2b, 0, 1, 1, 100000)
Q2c = monte_carlo(f_2c,-2, 2, 1, 100000)
Q2d = monte_carlo(f_2d, np.array([0,0]), np.array([1,1]), 2, 100000)



###############
#Task3
###############

#Defining a step function to evaluate the content of an n-sphere where 
#r = radius of the n-sphere
def f_3(v):
    r = np.sqrt(sum(v**2))
    
    #Using an if else statement to set up the step function so that any pseudo_random generated r value bigger than 2,
    #hence designating a point position outside the n-sphere, is ignored
    if r <= 2:
        return 1
    else: 
        return 0

#Using the defined monte_carlo function and appropriate inputs to evaluate the content of the n = 3, and n = 5 n-spheres
Q3_3D = monte_carlo(f_3, np.array([-2,-2,-2]), np.array([2,2,2]), 3, 100000)
Q3_5D = monte_carlo(f_3, np.array([-2,-2,-2,-2,-2]), np.array([2,2,2,2,2]), 5, 100000)



###############
#Task4
###############

#Defining the given 9-dimensional integrand 
def f_4(v):
    ax, ay, az, bx, by, bz, cx, cy, cz = v
    
    #Grouping independent values into 3 arrays to simplify the inputation of the integrand
    a = np.array([ax, ay, az])
    b = np.array([bx, by, bz])
    c = np.array([cx, cy, cz])
    return 1/(abs(np.dot((a+b),c)))

#Using the monte_carlo function and appropriate inputs to solve the given 9-dimensional integral
Q4 = monte_carlo(f_4, np.zeros(9), np.ones(9), 9, 100000)



###############
#Task5
###############

#Defining a function to use the Metropolis method to generate a non-uniform random sampling of a weighting function, where 
#x_i = desination of initial state of integrand, delta = an arbitrary number, p = probability distribution function, N = number of samples
def metropolis (x_i, delta, p, N):
     
    ran_path = []    #Declaring an empty array to store the random walks of from the initial state
    
    #Using a for loop to simulate a Metropolis random walk path where x_trial = new trial position
    for i in range(N-1):   #Range of for loop is set to N-1 to match x_i being the first position
        x_trial = x_i + np.random.uniform(-delta, delta)   #Assigning a pseudo-randomly generated value for x_trial at each N position
        w = p(x_trial)/p(x_i)   #Calculating the ratio of the probability of a walk to the probability of no walks
    
        #Using an if else statement to state the conditions needed for a random walk to take place
        if w >= 1:
            x_i = x_trial
        else: 
            r = np.random.uniform(0, 1)
            if r <= w:
                x_i = x_trial
                
        #Using an if statement to append the value of x_i, only at every EVEN positions, to lower the correlation between the values
        if i%2 == 0:
            ran_path = np.append(ran_path, x_i)
            
    return ran_path
                  
#Defining a new Monte-Carlo sampling function with the importance sampling using the Metropolis method function
#This function is used to evaluate 1-dimensional integrals 
def metro_monte_carlo (f, a, b, x_i, delta, p, N):
   
    ran_nums= metropolis (x_i, delta, p, N)    #Generating an array of non-uniform random numbers using the Metropolis function
    
    n = len(ran_nums)   #Declaring the number of samples as the number of positions in ran_path from the Metropolis function
    
    #Declaring both g (a rescaled f that equals to f/p) and g_sq as 0
    g = 0
    g_sq = 0
    
    #Using a for loop to sum up the calculated values of g and g_sq for each value of ran_nums
    for i in ran_nums:
        g_i = f(i)/p(i)
        g += g_i
        g_sq += g_i**2
    
    integral = g/n    #Solving the integral 
    
    mean_g_sq = g_sq/n  #Calculating the mean value of g_sq by dividing it by the number of samples
    
    #Calculating the variance and error of the solution
    variance = abs((1/n)*((mean_g_sq) - (integral**2)))
    error = np.sqrt(variance)
    
    #Plotting histograms from the ran_nums array to see the shape of the distribution
    plt.hist(ran_nums, bins = 250)
    plt.xlabel("Values generated from Random Walk")
    plt.ylabel("Population")
    plt.show()
    
    return integral, error
     
#Defining the 1-dimensional integrands and the corresponding sampling function
def f_5a(x):
    return 2*(np.exp(-(x**2)))
def p_a(x):
    return (np.exp(-abs(x)))/(2*(1-np.exp(-10)))

def f_5b(x):
    return 1.5*(np.sin(x))
def p_b(x):
    return ((4*x*(np.pi-x))/np.pi**2)/((2*np.pi)/(3))

#Using the new Monte_carlo function and appropriate inputs to solve the given integrals
Q5a = metro_monte_carlo(f_5a, -10, 10, 0, 1, p_a, 500000)
Q5b = metro_monte_carlo(f_5b, 0, np.pi, 1, 1, p_b, 500000)


###############
#Task6
###############

#Using the original Monte_carlo function and appropriate inputs to solve the given integrals
Q6a = monte_carlo(f_5a, -10, 10, 1, 1000000)
Q6b = monte_carlo(f_5b, 0, np.pi, 1, 1000000)