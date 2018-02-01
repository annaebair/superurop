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
		self.weights, self.distributions, self.log_l = mixem.em(data, dist_list, max_iterations=200, progress_callback=None)
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

	# Censored 1
	predata1 = np.random.normal(loc=0, scale=1, size=1000)
	censored_data_1 = np.where(predata1 <= -1, -1, predata1)
	censored_data_1 = np.where(censored_data_1 >= 3, 3, censored_data_1)
	censored_dist_1 = CensoredNormalDistribution(mu=1.0, sigma=1, lower=-1, upper=3)

	# Censored 2
	predata2 = np.random.normal(loc=3, scale=1, size=1000)
	censored_data_2 = np.where(predata2 <= -3, -3, predata2)
	censored_data_2 = np.where(censored_data_2 >= 4, 4, censored_data_2)
	censored_dist_2 = CensoredNormalDistribution(mu=1.5, sigma=1, lower=-3, upper=4)

	# Truncated 1
	predata3 = np.random.normal(loc=1, scale=1, size=100000)
	truncated_data_1 = np.array(list(filter(lambda x: x > 0 and x < 2, predata3)))
	truncated_dist_1 = TruncatedNormalDistribution(mu=1.106, sigma=1, lower=0, upper=2)

	# Truncated 2
	predata4 = np.random.normal(loc=5, scale=1, size=146500)
	truncated_data_2 = np.array(list(filter(lambda x: x > 4 and x < 6, predata4)))
	truncated_dist_2 = TruncatedNormalDistribution(mu=5, sigma=1.004, lower=4, upper=6)

	# Normal 1
	normal_data_1 = np.random.normal(loc=0.5, scale=1, size=100000)
	normal_dist_1 = mixem.distribution.NormalDistribution(mu=0.5, sigma=1)

	# Normal 2
	normal_data_2 = np.random.normal(loc=3, scale=2, size=100000)
	normal_dist_2 = mixem.distribution.NormalDistribution(mu=3.434, sigma=1.964)

	trunced_normal = np.array(list(filter(lambda x: x > 0 and x < 2, normal_data_2)))
	trunced_normal_dist = TruncatedNormalDistribution(mu=3, sigma=2, lower=0, upper=2)

	# print("trunced normal mean: ", sum(trunced_normal)/len(trunced_normal))

	data = np.concatenate((normal_data_2, truncated_data_1))
	np.random.shuffle(data)
	val_set = data[:1000]
	train_set = data[1000:]
	# print("val length: ", len(val_set))
	# print("train length: ", len(train_set))
	# print("trunc len: ", len(truncated_data_1))

	pre_scipy_dist_1 = norm(3, 2)
	pre_scipy_dist_2 = truncnorm(a=-1, b=1, loc=1, scale=1)

	dist_list = [normal_dist_2, truncated_dist_1]
	mixture = Data_Rep(data, dist_list)
	post_scipy_dist_1, post_scipy_dist_2 = mixture.scipy_dists
	weight_normal = len(normal_data_2)/len(data)
	weight_trunc = 1-weight_normal
	pre_weights = np.array([weight_normal, weight_trunc])

	sorted_val = np.array(sorted(val_set))

	weights, distributions, log_l = mixem.em(train_set, dist_list, max_iterations=200, progress_callback=None)
	post_prob = mixem.probability(sorted_val, weights, distributions)


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

	x = np.arange(-7, 7, 0.001)

	pdf1 = [post_scipy_dist_1.pdf(i) for i in x]
	pdf2 = [post_scipy_dist_2.pdf(i) for i in x]
	prepdf1 = [pre_scipy_dist_1.pdf(i) for i in x]
	prepdf2 = [pre_scipy_dist_2.pdf(i) for i in x]


	def get_p_values():
		for i in [-1, 0, 0.5, 1, 1.5, 3, 4, 4.5, 5, 6, 7]:
			query = i
			p_val = mixture.get_p_value(query)
			print("p value of %s: " % query, p_val)

	# plt.plot(val_set, pre_prob, 'bo')
	# plt.plot(val_set, post_prob, 'go')
	# plt.show()
	
	plt.figure
	plt.subplot(221)
	plt.hist(truncated_data_1, bins=100, normed=True)
	plt.hist(normal_data_2, bins=100, normed=True)
	plt.plot(x, pdf1, label="post1")
	plt.plot(x, pdf2, label="post2")
	plt.plot(x, prepdf1, label="pre1")
	plt.plot(x, prepdf2, label="pre2")
	plt.ylim(0, 1)
	plt.xlim(-5, 10)
	plt.legend()
	# plt.show()

	plt.subplot(222)
	plt.hist(data, bins=200, normed=True)
	plt.plot(x, pdf1, label="post1")
	plt.plot(x, pdf2, label="post2")
	plt.plot(x, prepdf1, label="pre1")
	plt.plot(x, prepdf2, label="pre2")
	plt.ylim(0, 1)
	plt.xlim(-5, 10)

	plt.subplot(223)
	d = np.concatenate((truncated_data_1, trunced_normal))
	plt.hist(d, bins=100, normed=True)
	plt.ylim(0, 1)
	plt.xlim(0, 2)


	plt.subplot(224)
	generated = np.random.normal(loc=1.112, scale=1.004, size=100000)
	generated_trunc = np.array(list(filter(lambda x: x > 0 and x < 2, generated)))
	plt.hist(generated_trunc, bins=100, normed=True)
	plt.ylim(0, 1)
	plt.xlim(0, 2)
	plt.show()
