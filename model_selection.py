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
		self.val, self.train = self.get_data()

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
		set_of_num_params = []
		set_of_distributions = self.get_set_of_dists()
		for set_of_dists in set_of_distributions:
			num_params = 0
			dist_list = []
			for dist in set_of_dists:
				if dist == NormalDistribution:
					#random parameter initialization?
					dist = dist(mu=2, sigma=1)
					dist_list.append(dist)
					num_params += 2
				elif dist == ExponentialDistribution:
					dist = dist(lmbda=1)
					dist_list.append(dist)
					num_params += 1
				elif dist == LogNormalDistribution:
					dist = dist(mu=0, sigma=1)
					dist_list.append(dist)
					num_params += 2
				elif dist == TruncatedNormalDistribution:
					#known lower and upper bounds here
					dist = dist(mu=0, sigma=1, lower=-1, upper=1)
					dist_list.append(dist)
					num_params += 2
				elif dist == CensoredNormalDistribution:
					dist = dist(mu=0, sigma=1, lower=-1, upper=1)
					dist_list.append(dist)
					num_params += 2
			set_of_dists_with_params.append(dist_list)
			set_of_num_params.append(num_params)
		return set_of_dists_with_params, set_of_num_params

	def get_data(self):
		#censored
		predata1 = np.random.normal(loc=0, scale=1, size=10000)
		# data1 = np.array(list(filter(lambda x: x < 1 and x > -1, predata1)))
		data1 = np.where(predata1 <= -1, -1, predata1)
		data1 = np.where(data1 >= 1, 1, data1)
		#normal
		data2 = np.random.normal(loc=2, scale=1, size=10000)
		data = np.concatenate((data1, data2))
		np.random.shuffle(data)
		val = data[:1000]
		train = data[1000:]
		return val, train

	def get_best_model(self):
		num_parameters = {NormalDistribution:2, TruncatedNormalDistribution:2, CensoredNormalDistribution:2}
		min_BIC = np.inf
		best_rep = None
		set_of_dists_with_params, set_of_num_params = self.parameterize_initial_dists()
		for i in range(len(set_of_dists_with_params)):
			rep = Data_Rep(self.train, set_of_dists_with_params[i])
			if rep.log_l:
				total_num_params = set_of_num_params[i]
				BIC = log(len(self.val))*total_num_params - 2*rep.log_l

				#not regularizing enough
				# try validation set ?
				# 1. make sure censored is working properly
				#2. investigate BIC
				# 3. validation set
				# GMM selection sklearn example
				print("\n")
				print("distributions: ", rep.distributions)
				print("data log: ", log(len(self.val)))
				print("log likelihood term: ", rep.log_l)
				print("regularization: ", log(len(self.val))*total_num_params)
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
	print("weights: ", best_rep.weights)
	print("distributions: ", best_rep.distributions)

	# look at sklearn GMM BIC
	# make sure censored dist estimate params isnt pushing stdev to inf
