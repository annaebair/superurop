import mixem
import numpy as np
from scipy.stats import norm, expon, truncnorm
import matplotlib.pyplot as plt
from em import Moments, TruncatedNormalDistribution


class Data_Rep:

	def __init__(self, data, dist_list):
		self.cdf = self._mixture_cdf(data, dist_list)


	def get_p_value(self, value):

		p_value = 1 - self.cdf(value)
		return p_value


	def _mixture_cdf(self, data, dist_list):

		weights, distributions, log_l = mixem.em(data, dist_list)
		scipy_dists = self._get_scipy_dists(distributions)

		return lambda query: sum([w * dist.cdf(query) for w, dist in zip(weights, scipy_dists)])


	def _get_scipy_dists(self, distributions):

		scipy_dists = []

		for dist in distributions:
			dist_type = type(dist)

			if dist_type == mixem.distribution.NormalDistribution:
				loc = dist.mu
				scale = dist.sigma
				ss_dist = norm(loc=loc, scale=scale)
				
			elif dist_type == mixem.distribution.ExponentialDistribution:
				l = dist.lmbda
				ss_dist = expon(l)

			elif dist_type == TruncatedNormalDistribution:
				a = dist.a
				b = dist.b
				loc = dist.mu
				scale = dist.sigma
				ss_dist = truncnorm(a=a, b=b, loc=loc, scale=scale)

			scipy_dists.append(ss_dist)

		return scipy_dists



if __name__ == "__main__":

	### Normal Data ###

	# dist1 = mixem.distribution.NormalDistribution(0, 2)
	# dist2 = mixem.distribution.NormalDistribution(11, 3)

	# data1 = np.random.normal(loc=0, scale=0.75, size=1000)
	# data2 = np.random.normal(loc=10, scale=0.9, size=1000)

	### Exponential Data ###

	# dist1 = mixem.distribution.ExponentialDistribution(2)
	# dist2 = mixem.distribution.ExponentialDistribution(3)

	# data1 = np.random.exponential(scale=0.5, size=1000)
	# data2 = np.random.exponential(scale=1.0/20, size=1000)

	### Trucated Normal Data ###
	
	dist1 = TruncatedNormalDistribution(mu=0, sigma=1, lower=-0.5, upper=1)
	dist2 = TruncatedNormalDistribution(mu=5, sigma=1, lower=4, upper=6)

	predata1 = np.random.normal(loc=0, scale=1, size=10000)
	predata2 = np.random.normal(loc=6, scale=0.75, size=1000)

	data1 = np.array(list(filter(lambda x: x > -0.5 and x < 1, predata1)))
	data2 = np.array(list(filter(lambda x: x > 4 and x < 6, predata2)))


	data = np.concatenate((data2, data1))
	dist_list = [dist1, dist2]
	mixture = Data_Rep(data, dist_list)
	query = 7
	bincount = len(data1)
	plt.hist(data, bins=100)
	plt.show()

	print("p-value of %s:" %query, mixture.get_p_value(query))
