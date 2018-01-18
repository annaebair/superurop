import mixem
import numpy as np
from math import exp, isclose
import math
from scipy.stats import norm, expon, truncnorm, lognorm
import matplotlib.pyplot as plt
from em import Moments, TruncatedNormalDistribution, CensoredNormalDistribution

np.set_printoptions(threshold=np.nan)


class censorednorm:

	def __init__(self, loc, scale, a, b):
		self.loc = loc
		self.scale = scale
		self.a = a
		self.b = b
		# print("loc: ", loc, "; scale: ", scale)

	def pdf(self, query):
		if query < self.a:
			return 0
		elif query <= self.a + 0.01:
			return 500*norm.cdf((self.a-self.loc)/self.scale)
		elif query < self.b:
			return norm.pdf((query-self.loc)/self.scale)
		elif query <= self.b + 0.01:
			return 500*(1-norm.cdf((self.b-self.loc)/self.scale))
		else:
			return 0


class Data_Rep:

	def __init__(self, data, dist_list):
		self.cdf = self._mixture_cdf(data, dist_list)

	def get_p_value(self, value):
		p_value = 1 - self.cdf(value)
		return p_value

	def _mixture_cdf(self, data, dist_list):
		weights, distributions, log_l = mixem.em(data, dist_list, max_iterations=4)
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

	predata1 = np.random.normal(loc=0, scale=1, size=1000)
	predata2 = np.random.normal(loc=3, scale=1, size=1000)

	dist1 = CensoredNormalDistribution(mu=0, sigma=1, lower=0, upper=1)
	dist2 = CensoredNormalDistribution(mu=3, sigma=1, lower=2, upper=4)
	#FIX HERE
	nextdata1 = np.where(predata1 <= 0, 0, predata1)
	data1 = np.where(nextdata1 >= 1, 1, nextdata1)

	nextdata2 = np.where(predata2 <= 2, 2, predata2)
	data2 = np.where(nextdata2 >= 4, 4, nextdata2)
	
	# dist1 = TruncatedNormalDistribution(mu=0, sigma=1, lower=0, upper=1)
	# dist2 = TruncatedNormalDistribution(mu=3, sigma=1, lower=2, upper=4)
	# data1 = np.array(list(filter(lambda x: x > 0 and x < 1, predata1)))
	# data2 = np.array(list(filter(lambda x: x > 2 and x < 4, predata2)))
	
	data = np.concatenate((data1, data2))
	dist_list = [dist1, dist2]
	# scipy_dist_1,scipy_dist_2 = Data_Rep.get_scipy_dists(dist_list)
	mixture = Data_Rep(data, dist_list)
	# post_scipy_dist_1, post_scipy_dist_2 = mixture.scipy_dists

	x = np.arange(-2, 5, 0.001)
	# pre_pdf = [scipy_dist_1.pdf(i) for i in x]
	# pre_pdf_2 = [scipy_dist_2.pdf(i) for i in x]

	# pdf = [post_scipy_dist_1.pdf(i) for i in x]
	# pdf_2 = [post_scipy_dist_2.pdf(i) for i in x]

	plt.hist(predata1, bins=100, normed=True)
	# # plt.plot(x, pre_pdf, label="pre1")
	# # plt.plot(x, pre_pdf_2, label="pre2")
	# plt.plot(x, pdf, label="post1")
	# plt.plot(x, pdf_2, label="post2")
	# plt.legend()
	# plt.ylim(0, 6)
	# plt.show()


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
