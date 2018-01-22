import random
import numpy as np
import mixem
from sklearn.model_selection import RandomizedSearchCV
from mixem.distribution import (
    NormalDistribution, 
    ExponentialDistribution, 
    LogNormalDistribution
)
from em import TruncatedNormalDistribution, CensoredNormalDistribution
from p_value import Data_Rep
from math import log


class ModelSelection:

	def __init__(self, dist_types, max_num_dists, num_random_samples):
		self.dist_types = dist_types
		self.max_num_dists = max_num_dists
		self.num_random_samples = num_random_samples
		self.data = self.get_data()

	def get_set_of_dists(self):
		set_of_distributions = []
		for i in range(self.num_random_samples):
		    distributions = []
		    for dist in self.dist_types:
		        num_dists = random.randint(0, self.max_num_dists)
		        for i in range(num_dists):
		            distributions.append(dist)
		    set_of_distributions.append(distributions)
		return set_of_distributions

	def parameterize_initial_dists(self):
		set_of_dists_with_params = []
		set_of_distributions = self.get_set_of_dists()
		for set_of_dists in set_of_distributions:
		    dist_list = []
		    for dist in set_of_dists:
		        if dist == NormalDistribution:
		            #random parameter initialization?
		            dist = dist(mu=2, sigma=1)
		            dist_list.append(dist)
		        elif dist == ExponentialDistribution:
		            dist = dist(lmbda=1)
		            dist_list.append(dist)
		        elif dist == LogNormalDistribution:
		            dist = dist(mu=0, sigma=1)
		            dist_list.append(dist)
		        elif dist == TruncatedNormalDistribution:
		            #known lower and upper bounds here
		            dist = dist(mu=0, sigma=1, lower=-1, upper=1)
		            dist_list.append(dist)
		        elif dist == CensoredNormalDistribution:
		            dist = dist(mu=0, sigma=1, lower=-1, upper=1)
		            dist_list.append(dist)
		    set_of_dists_with_params.append(dist_list)
		return set_of_dists_with_params

	def get_data(self):
		#censored
		predata1 = np.random.normal(loc=0, scale=1, size=1000)
		data1 = np.where(predata1 <= -1, -1, predata1)
		data1 = np.where(data1 >= 1, 1, data1)
		#normal
		data2 = np.random.normal(loc=2, scale=1, size=1000)
		data = np.concatenate((data1, data2))
		return data

	def get_best_model(self):
		min_BIC = np.inf
		best_rep = None
		set_of_dists_with_params = self.parameterize_initial_dists()
		for s in set_of_dists_with_params:

			rep = Data_Rep(self.data, s)
			if rep.log_l:
				num_dists = len(rep.distributions)
				BIC = 10*log(len(self.data))*num_dists - 2*rep.log_l\
				#not regularizing enough
				print("regularization: ", log(len(self.data))*num_dists)
				print("BIC: ", BIC)
				if BIC < min_BIC:
					min_BIC = BIC
					best_rep = rep
		return best_rep, min_BIC


if __name__ == "__main__":
	dist_types = [NormalDistribution, 
	             TruncatedNormalDistribution, CensoredNormalDistribution]
	max_num_dists = 1
	num_random_samples = 20
	#calculate this intelligently later

	ms = ModelSelection(dist_types=dist_types, max_num_dists=max_num_dists, num_random_samples=num_random_samples)
	best_rep, min_BIC = ms.get_best_model()
	print("min BIC: ", min_BIC)
	print("distributions: ", best_rep.distributions)
