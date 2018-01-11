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
		# prob = mixem.probability(data, np.array([0.5, 0.5]), dist_list)
		# print(sum(np.log(prob)))
		weights, distributions, log_l = mixem.em(data, dist_list, progress_callback=None, initial_weights=[0.5, 0.5], max_iterations=200)
		# prob = mixem.probability(data, weights, distributions)
		# print(weights)
		# print(log_l)
		# print(sum(np.log(prob)))
		scipy_dists = self._get_scipy_dists(distributions)
		return lambda query: sum([w * dist.cdf(query) for w, dist in zip(weights, scipy_dists)])


	def _get_scipy_dists(self, distributions):

		scipy_dists = []

		for dist in distributions:
			dist_type = type(dist)
			# print(dist_type)

			if dist_type == mixem.distribution.NormalDistribution:
				loc = dist.mu
				scale = dist.sigma
				ss_dist = norm(loc=loc, scale=scale)
				# print("here")
				
			elif dist_type == mixem.distribution.ExponentialDistribution:
				l = dist.lmbda
				ss_dist = expon(l)

			elif dist_type == TruncatedNormalDistribution:
				a = dist.a
				b = dist.b
				loc = dist.mu
				scale = dist.sigma
				print(a, b, loc, scale)
				ss_dist = truncnorm(a=(a-loc)/scale, b=(b-loc)/scale, loc=loc, scale=scale)
				x = np.arange(-10, 10, 0.01)
				plt.plot(ss_dist.pdf(x))
				plt.show()

			scipy_dists.append(ss_dist)
		return scipy_dists



if __name__ == "__main__":


	dist1 = TruncatedNormalDistribution(mu=0, sigma=1, lower=-1, upper=1)
	dist2 = TruncatedNormalDistribution(mu=3, sigma=1, lower=2, upper=4)

	predata1 = np.random.normal(loc=0, scale=1, size=100000)
	data1 = np.array(list(filter(lambda x: x > -1 and x < 1, predata1)))

	predata2 = np.random.normal(loc=3, scale=1, size=100000)
	data2 = np.array(list(filter(lambda x: x > 2 and x < 4, predata2)))
	
	# dist1 = mixem.distribution.NormalDistribution(11, 3)
	# data1 = np.random.normal(loc=10, scale=0.9, size=10000)


	data = np.concatenate((data2, data1))
	dist_list = [dist1, dist2]
	mixture = Data_Rep(data, dist_list)
	scipy_dist_1, scipy_dist_2 = mixture._get_scipy_dists(dist_list)
	# x = np.arange(-10, 10, 0.01)
	# plt.plot(scipy_dist_2.pdf(x))
	# # plt.hist(data, bins=1000)
	# plt.show()
	

	# for i in [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
	# # for i in [-10, -5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 3, 5, 6, 7, 10]:
	# 	print("VAL=", i)
	# 	print("p-value of %s:" %i, mixture.get_p_value(i), "\n")

	# plt.hist(data, bins=100)
	# rv = truncnorm(a=-1, b=1, loc=3)
	# x = np.arange(0, 10, 0.01)
	# plt.plot(x, rv.pdf(x))
	# plt.show()
