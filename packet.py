import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from numba import jit,float64
from scipy.stats import linregress

import math

import time

sns.set_style("ticks")
sns.set_context("talk")


@jit(float64[:, :](float64[:], float64[:]),nopython=True)
def recombine_arrays(arr1, arr2):
    res = np.zeros((len(arr1), 2), dtype=np.float64)
    res[:, 0] = arr1
    res[:, 1] = arr2
    for k in range(len(arr1)):
        alpha = np.random.random()
        if np.random.random() < 0.5:
            res[k, 0] = alpha * arr1[k] + (1 - alpha) * arr2[k]
            res[k, 1] = alpha * arr2[k] + (1 - alpha) * arr1[k]
        else:
            res[k, 1] = alpha * arr1[k] + (1 - alpha) * arr2[k]
            res[k, 0] = alpha * arr2[k] + (1 - alpha) * arr1[k]
    return res

#@jit(float64[:](float64[:,:],float64[:]),nopython=True)
@jit
def calc_fitness(function,x,population,target):
    fitness = np.zeros(population.shape[0],dtype=np.float64)
    y = np.zeros(target.shape[0],dtype=np.float64)

    for p in range(population.shape[0]):
        #y = function(x,population[p,:])
        y = test(x, population[p, :])
        fitness[p] = np.sqrt(np.mean(np.power(np.subtract(target,y),2)))
        #fitness[p] = np.mean(np.abs(np.subtract(target, y)))

    return fitness

@jit(float64[:,:](float64[:,:]),nopython=True)
def recombine_population(population):
    new_pop = np.zeros(population.shape,dtype=np.float64)
    n = population.shape[0]

    unfit = np.arange(int(n*2/3),n)
    mothers = np.arange(0,int(n/10))
    fathers = np.arange(int(n/10),int(n*2/3))

    i = 0
    while True:
        mother = mothers[np.random.randint(0,len(mothers))]
        father = fathers[np.random.randint(0,len(fathers))]
        childs = recombine_arrays(population[mother, :], population[father, :])
        new_pop[unfit[i],:] = childs[:, 0]
        i += 1
        if i >= len(unfit):
            break
        new_pop[unfit[i], :] = childs[:, 1]
        i += 1
        if i >= len(unfit):
            break

    rest = np.arange(0,unfit[0])

    for i in rest:
        new_pop[i, :] = population[np.random.randint(0, n-1), :]

    return new_pop

@jit(float64[:,:](float64[:,:],float64[:,:],float64,float64),nopython=True)
def mutate_population(population,bounds,mutation_rate,sigma):
    for i in range(population.shape[0]):
        #if i > 4:
            if np.random.random() < mutation_rate:
                for j in range(population.shape[1]):
                    if np.random.random() < 0.1:#< mutation_rate:
                        index = np.random.randint(0,population.shape[1])
                        #population[i, index] = bounds[0, index] + (bounds[1, index]-bounds[0, index])*np.random.random()
                        # population[i, j] += (bounds[1, j] - bounds[0, j]) * np.random.normal(0,0.5)
                        #population[i, j] += (bounds[1, j] - bounds[0, j]) * np.random.normal(0,sigma)
                        population[i, j] += (bounds[1, j] - bounds[0, j]) * (np.random.random()*2*sigma-sigma)
                        if population[i, j] > bounds[1, j]:
                            population[i, j] = bounds[1, j]
                        if population[i, j] < bounds[0, j]:
                            population[i, j] = bounds[0, j]
        #if np.random.random() < mutation_rate:
        #    index = np.random.randint(0, population.shape[1])
        #    population[i, index] = (bounds[0, index] + (bounds[1, index]-bounds[0, index])*np.random.random())
    return population

@jit(float64[:,:](float64[:,:],float64[:,:]),nopython=True)
def check_limits(population,bounds):
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            if population[i,j] < bounds[0,j]:
                population[i,j] = bounds[0,j]
            if population[i, j] > bounds[1, j]:
                population[i, j] = bounds[1, j]
    return population


population_size = 50#100 #60
max_iter = 100000
mutation_rate = 0.5

#@jit#(float64(float64[:],float64[:],float64[:],float64[:],float64[:],float64[:]))
@jit
def minimize(x0,y0,function,bounds,inital_values):
    logpoints = np.arange(500, max_iter, 500)
    checkpoints = np.arange(50, max_iter, 50)

    population = np.zeros((population_size,bounds.shape[1]),dtype=np.float64)

    convergence = np.zeros(max_iter)
    t = np.zeros(max_iter)

    for i in range(population_size):
        #a = np.random.normal(1,0.5)
        for j in range(bounds.shape[1]):
            population[i, j] = bounds[0,j] + (bounds[1,j]*np.random.random())
            #population[i, j] = inital_values[j]+a#*np.random.normal(1,0.5)#*a

    print("population initialized")
    sigma = 1#0.5
    starttime = time.time()
    for i in range(max_iter):
        fitness = calc_fitness(function,x0,population,y0)
        sorted_ind = np.argsort(fitness)

        if fitness[sorted_ind][0]*100 < 0.0001:
            break

        if i < 1000:
            sigma = 2#0.5
        elif i < 2000:
            sigma = 1#0.25
        else:
            if i in checkpoints:
                indices = np.arange(i - 500, i, step=1)
                slope, intercept, r_value, p_value, std_err = linregress(t[indices], convergence[indices])
                if (std_err > 0.01) or (slope > 0):
                    sigma *= 0.9
                if (std_err < 0.0001) and (slope > 0):
                    sigma *= 1.1
        #         if sigma < 0.001:
        #             sigma = 0.001

        population = population[sorted_ind,:]
        population = recombine_population(population)
        population = mutate_population(population, bounds, mutation_rate, sigma)
        population = check_limits(population, bounds)
        if i in logpoints:
            print("{0:7d}: fitness: {1:1.7f}%, sigma: {2:1.5f}".format(i, fitness[sorted_ind][0]*100, sigma))
        convergence[i] = fitness[sorted_ind][0]
        t[i] = time.time() - starttime
        #print(str(i)+ ": " + str(fitness[sorted][0]))

    return population[0,:], t, convergence


@jit(float64[:](float64[:],float64[:]),nopython=True)
def test(x,args):
    y = np.zeros(x.shape[0],np.float64)
    for i in range(y.shape[0]):
        y[i] = args[0]/(args[2]*math.sqrt(2*math.pi))*math.exp(-0.5*((x[i]-args[1])/args[2])**2)+args[3]/(args[5]*math.sqrt(2*math.pi))*math.exp(-0.5*((x[i]-args[4])/args[5])**2)
    return y


par = np.array([1,30,5,5,50,10],np.float64)
initial = par*np.random.normal(1,0.5,par.shape[0])
print(par)
print(initial)
x = np.linspace(0,100,30,dtype=np.float64)
y = test(x,par)

noise = np.random.normal(0, np.max(y)*0.05, y.shape[0])
#y = y + noise

#plt.plot(x,y)
#plt.show()

#pop = np.array([[1,10,10,1,50,50],[2,20,20,2,20,20]],np.float64)
#f = calc_fitness(test,x,pop,y)
#print(f)

#def minimize(x0,y0,function,initial_parameters):
bounds = np.array([[0,np.min(x),1,0,np.min(x),1],[100,np.max(x),100,100,np.max(x),100]],np.float64)
res, t, convergence = minimize(x, y, test,bounds,initial)

print(par)
print(res)
#name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_convergence.png"
#fig = plt.figure()
print("time for iteration: "+ str(np.round(np.max(t),2))+" seconds")
plt.semilogy(t,convergence)
plt.xlabel('time/s')
plt.ylabel('error')
plt.show()
#plt.savefig(name)
#plt.close()

y2 = test(x,res)

plt.plot(x,y)
plt.plot(x,y2)
plt.show()


#
#         name = "pics/"+prefixes[l]+"_"+str(dists[k])+".png"
#         fig = plt.figure()
#         cmap = sns.cubehelix_palette(light=1, as_cmap=True,reverse=False)
#         plot = plt.imshow(exposure,cmap=cmap,extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
#         plt.colorbar()
#         plt.contour(x.reshape(orig_shape), y.reshape(orig_shape), exposure, [300])#[290,300, 310])
#         #plt.scatter(x_t,y_t,c="red")
#         #plt.show()
#         plt.xlabel('x/nm')
#         plt.ylabel('y/nm')
#         plt.savefig(name)
#         plt.close()
#
#         name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_expected.png"
#         fig = plt.figure()
#         plot = plt.imshow((exposure >= 300),extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
#         #plt.contour(x_t.reshape(target_shape), y_t.reshape(target_shape), target.reshape(target_shape), [299],color="black")
#         #plt.scatter(x_t,y_t,c="red")
#         plt.scatter(x0,y0,c="blue")
#         plt.xlabel('x/nm')
#         plt.ylabel('y/nm')
#         plt.savefig(name)
#         plt.close()
#
#         name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_convergence.png"
#         fig = plt.figure()
#         print("time for iteration: "+ str(np.round(np.max(t),2))+" seconds")
#         plt.semilogy(t,convergence)
#         plt.xlabel('time/s')
#         plt.ylabel('error')
#         plt.savefig(name)
#         plt.close()
#
#         #plt.scatter(x_t,y_t,c="red")
#         #plt.scatter(x0,y0,c="blue")
#         #plt.show()
#
#         area = np.pi * (15*repetitions/np.max(repetitions))**2
#         plt.scatter(x0, y0, s=area, alpha=0.5,edgecolors="black",linewidths=1)
#         plt.axes().set_aspect('equal', 'datalim')
#         name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_scatter.png"
#         plt.savefig(name)
#         plt.close()
#         x0 = x0/1000
#         y0 = y0/1000
#         repetitions = np.array(np.round(repetitions),dtype=np.int)
#         print(repetitions)
#         for j in range(len(x0)):
#             if repetitions[j] > 1:
#                 Outputfile.write('RDOT '+str(x0[j]) + ', ' + str(y0[j]) + ', ' + str((repetitions[j])) + '\n')
#         Outputfile.write('END' + '\n')
#         Outputfile.write('\n')
#         Outputfile.write('\n')
#
# Outputfile.close()
#

