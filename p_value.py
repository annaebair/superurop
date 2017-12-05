import mixem
import numpy as np
from scipy.stats import norm


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

		return scipy_dists



if __name__ == "__main__":

	dist1 = mixem.distribution.NormalDistribution(0, 2)
	dist2 = mixem.distribution.NormalDistribution(11, 3)

	data1 = np.random.normal(loc=0, scale=0.75, size=1000)
	data2 = np.random.normal(loc=10, scale=0.9, size=1000)

	data = np.concatenate((data2, data1))

	dist_list = [dist1, dist2]
	mixture = Data_Rep(data, dist_list)
	query = 9

	print("p-value of %s:" %query, mixture.get_p_value(query))
