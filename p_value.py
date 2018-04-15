import mixem
import numpy as np
from math import exp, isclose
import math
import time
from scipy.stats import norm, expon, truncnorm, lognorm
import scipy.stats as ss
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
		self.joint_pdf = self.joint_pdf()

	def get_p_value(self, value):
		p_value = 1 - self.cdf(value)
		return p_value

	def alt_p_val_calc(self, x, query):
		diff = [x[i]-query for i in range(len(x))]
		closest_ind = np.abs(diff).argmin()
		joint_pdf = self.joint_pdf
		joint_pdf_val = joint_pdf[closest_ind]
		locs = np.where(joint_pdf <= joint_pdf_val)
		last = 0
		first = 0
		total_area = 0
		for i in locs[0]:
			if i-last > 1:
				area = self.cdf(x[last]) - self.cdf(x[first])
				total_area += area
				first = i
			last = i
		return joint_pdf_val, total_area

	def get_cdf_val(self, vaue):
		return self.cdf(value)

	def _mixture_cdf(self, data, dist_list):
		self.weights, self.distributions, self.log_l = mixem.em(data, dist_list, max_iterations=200, progress_callback=None)
		self.scipy_dists = self.get_scipy_dists(self.distributions)
		return lambda query: sum([w * dist.cdf(query) for w, dist in zip(self.weights, self.scipy_dists)])

	def joint_pdf(self):
		x = np.arange(-7, 7, 0.001)
		joint_pdf = None
		for i in range(len(self.scipy_dists)):
			dist = self.scipy_dists[i]
			pdf = [dist.pdf(i) for i in x]
			if joint_pdf is None:
				joint_pdf = np.array(pdf)
			else:
				joint_pdf += np.array(pdf)
		return joint_pdf

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

