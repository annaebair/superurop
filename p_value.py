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

	def cdf(self, query):
		if query < self.a:
			return 0
		elif query < self.b:
			return norm.cdf((query-self.loc)/self.scale)
		else:
			return 1

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
		self.weights, self.distributions, self.log_l = mixem.em(data, dist_list, max_iterations=500, progress_callback=None)
		# print("probability: ", sum(mixem.probability(data, np.array([0.5, 0.5]), dist_list)))
		# print("probability: ", sum(mixem.probability(data, self.weights, self.distributions)))
		# print("distributions: ", self.distributions)
		# print("weights: ", self.weights)
		# print("log_l: ", self.log_l)
		self.scipy_dists = self.get_scipy_dists(self.distributions)
		return lambda query: sum([w * dist.cdf(query) for w, dist in zip(self.weights, self.scipy_dists)])

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

	# Censored
	predata1 = np.random.normal(loc=0, scale=1, size=1000)
	data1 = np.where(predata1 <= -3, -3, predata1)
	data1 = np.where(data1 >= 4, 4, data1)
	dist1 = CensoredNormalDistribution(mu=0, sigma=1, lower=-3, upper=4)

	# Censored
	predata4 = np.random.normal(loc=3, scale=1, size=1000)
	data4 = np.where(predata4 <= -3, -3, predata4)
	data4 = np.where(data4 >= 4, 4, data4)
	dist4 = CensoredNormalDistribution(mu=3, sigma=1, lower=-3, upper=4)

	# Truncated
	predata2 = np.random.normal(loc=0, scale=1, size=1447)
	data2 = np.array(list(filter(lambda x: x > -1 and x < 3, predata2)))
	dist2 = TruncatedNormalDistribution(mu=0.5, sigma=1, lower=-1, upper=3)

	# Truncated
	predata5 = np.random.normal(loc=2, scale=1, size=1547)
	data5 = np.array(list(filter(lambda x: x > 0 and x < 4, predata5)))
	dist5 = TruncatedNormalDistribution(mu=2, sigma=1, lower=0, upper=4)

	# Normal
	data3 = np.random.normal(loc=2, scale=1, size=1000)
	dist3 = mixem.distribution.NormalDistribution(mu=0.5, sigma=1)
	
	data = np.concatenate((data1, data4))
	dist_list = [dist1, dist4]
	mixture = Data_Rep(data, dist_list)
	post_scipy_dist_1, post_scipy_dist_3 = mixture.scipy_dists

	def find_best_initialization(data1, data2):
		best_log_l = -np.inf
		best_mus = None
		i=0
		best_dists = None
		for mu_1 in np.arange(-3, -1, 0.1):
			for mu_2 in np.arange(3, 5, 0.1):
				print(f'Iteration {i}: {mu_1} & {mu_2}')
				dist1 = CensoredNormalDistribution(mu=mu_1, sigma=1, lower=-3, upper=4)
				dist2 = CensoredNormalDistribution(mu=mu_2, sigma=1, lower=-3, upper=4)
				data = np.concatenate((data1, data2))
				dist_list = [dist1, dist2]
				mixture = Data_Rep(data, dist_list)
				mixture_log_l = mixture.log_l
				if mixture_log_l > best_log_l:
					best_log_l = mixture_log_l
					best_mus = (mu_1, mu_2)
					best_dists = mixture.distributions
				i += 1
		print("Best log likelihood: ", best_log_l)
		print("Best mus: ", best_mus)
		print("Best dists: ", best_dists)

	find_best_initialization(data1, data4)

	x = np.arange(-7, 7, 0.001)

	pdf = [post_scipy_dist_1.pdf(i) for i in x]
	pdf_3 = [post_scipy_dist_3.pdf(i) for i in x]

	for i in [-1, 0, 0.5, 1, 1.5, 3, 4, 4.5, 5, 6, 7]:
		query = i
		p_val = mixture.get_p_value(query)
		# print("p value of %s: " % query, p_val)

	plt.hist(data1, bins=100, normed=True)
	plt.hist(data4, bins=100, normed=True)

	plt.plot(x, pdf)
	plt.plot(x, pdf_3)
	plt.ylim(0, 6)
	plt.show()
