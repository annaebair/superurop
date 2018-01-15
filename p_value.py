import mixem
import numpy as np
from math import exp
from scipy.stats import norm, expon, truncnorm, lognorm
import matplotlib.pyplot as plt
from em import Moments, TruncatedNormalDistribution, CensoredNormalDistribution

class censorednorm:
	def __init__(self, loc, scale, a, b):
		self.loc = loc
		self.scale = scale
		self.a = a
		self.b = b

	def cdf(self, query):
		if query < self.a:
			return 0
		elif query > self.b:
			return 1
		else:
			return scipy.stats.norm(loc=self.loc, scale=self.scale)


class Data_Rep:

	def __init__(self, data, dist_list):
		self.cdf = self._mixture_cdf(data, dist_list)

	def get_p_value(self, value):

		p_value = 1 - self.cdf(value)
		return p_value


	def _mixture_cdf(self, data, dist_list):
		weights, distributions, log_l = mixem.em(data, dist_list)
		self.scipy_dists = self.get_scipy_dists(distributions)
		return lambda query: sum([w * dist.cdf(query) for w, dist in zip(weights, self.scipy_dists)])


	@staticmethod
	def get_scipy_dists(distributions):
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
				loc = dist.mu
				scale = dist.sigma
				a = float(dist.a-loc)/scale
				b = float(dist.b-loc)/scale
				ss_dist = truncnorm(a=a, b=b, loc=loc, scale=scale)

			elif dist_type == mixem.distribution.LogNormalDistribution:
				scale = exp(dist.mu)
				s = dist.sigma
				ss_dist = lognorm(s=s, scale=scale)

			elif dist_type == CensoredNormalDistribution:
				loc = dist.mu
				scale = dist.sigma
				a = dist.a
				b = dist.b
				ss_dist = censorednorm(loc, scale, a, b)

			scipy_dists.append(ss_dist)
		return scipy_dists


if __name__ == "__main__":

	dist1 = CensoredNormalDistribution(mu=0, sigma=1, lower=-1, upper=1)
	dist2 = CensoredNormalDistribution(mu=3, sigma=1, lower=2, upper=4)

	predata1 = np.random.normal(loc=0, scale=1, size=1000)
	# data1 = np.array(list(filter(lambda x: x > -1 and x < 1, predata1)))
	nextdata1 = np.where(predata1 >= -1, predata1, -1)
	data1 = np.where(nextdata1 <= 1, predata1, 1)

	predata2 = np.random.normal(loc=3, scale=1, size=1000)
	nextdata2 = np.where(predata1 >= 2, predata1, 2)
	data2 = np.where(nextdata1 <= 4, predata1, 4)
	# data2 = np.array(list(filter(lambda x: x > 2 and x < 4, predata2)))
	
	# dist1 = mixem.distribution.NormalDistribution(11, 3)
	# data1 = np.random.normal(loc=10, scale=0.9, size=10000)

	# dist1 = mixem.distribution.LogNormalDistribution(mu=3, sigma=1)
	# dist2 = mixem.distribution.LogNormalDistribution(mu=10, sigma=1)

	# data1 = np.random.lognormal(mean=3, sigma=1, size=1000)
	# data2 = np.random.lognormal(mean=10, sigma=1, size=1000)

	data = np.concatenate((data1, data2))
	dist_list = [dist1, dist2]
	scipy_dist_1,scipy_dist_2 = Data_Rep.get_scipy_dists(dist_list)

	mixture = Data_Rep(data, dist_list)
	post_scipy_dist_1, post_scipy_dist_2 = mixture.scipy_dists



	# x = np.arange(
	# 	min(
	# 		-1.5, scipy_dist_1.ppf(0.001), scipy_dist_2.ppf(0.001),
	# 	    post_scipy_dist_1.ppf(0.001), post_scipy_dist_2.ppf(0.001)
	# 	),
	# 	max(
	# 		4.5, scipy_dist_1.ppf(0.999), scipy_dist_2.ppf(0.999),
	# 		post_scipy_dist_1.ppf(0.999), post_scipy_dist_2.ppf(0.999)
	# 	),
	# 	0.01
	# )

	# x = np.arange(0, 100000, 0.01)

	# plt.subplot(2, 1, 1)
	# plt.hist(data1, bins=100, normed=True)
	# plt.plot(x, post_scipy_dist_1.pdf(x))
	# plt.xlim(0, 600)
	# plt.title("Distribution 1")

	# plt.subplot(2, 1, 2)
	# plt.hist(data2, bins=300, normed=True)
	# plt.plot(x, post_scipy_dist_2.pdf(x))
	# plt.xlim(0, 50000)
	# plt.title("Distribution 2")
	# plt.tight_layout()

	# plt.show()
