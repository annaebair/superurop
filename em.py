from scipy.stats import norm, truncnorm
import scipy.stats
from scipy.special import erf
from math import exp, sqrt, pi
import matplotlib.pyplot as plt
import numpy as np
import sklearn.mixture
import mixem
from mixem.distribution.distribution import Distribution


class Moments():

	def __init__(self, mu, sigma, a, b):
		self.mu = mu
		self.sigma = sigma
		self.a = a
		self.b = b
		self._zeroth_moment = self._zeroth_moment()
		self.first_moment = self._first_moment()
		self.second_moment = self._second_moment()

	def _zeroth_moment(self):
		upper_erf = erf((self.b-self.mu)/(self.sigma*sqrt(2)))
		lower_erf = erf((self.a-self.mu)/(self.sigma*sqrt(2)))
		return 0.5*(upper_erf - lower_erf)

	def _first_moment(self):
		lower_exp = exp(-((self.a-self.mu)/(self.sigma*sqrt(2)))**2)
		upper_exp = exp(-((self.b-self.mu)/(self.sigma*sqrt(2)))**2)
		return (self.sigma/sqrt(2*pi))*(lower_exp-upper_exp) + self.mu*self._zeroth_moment

	def _second_moment(self):
		lower_exp = exp(-((self.a-self.mu)/(self.sigma*sqrt(2)))**2)
		upper_exp = exp(-((self.b-self.mu)/(self.sigma*sqrt(2)))**2)
		return (self.sigma/sqrt(2*pi))*((self.a+self.mu)*lower_exp-(self.b+self.mu)*upper_exp) \
			+ (self.mu**2 + self.sigma**2)*self._zeroth_moment


class TruncatedNormalDistribution(Distribution):
    """Truncated normal distribution with parameters (mu, sigma)."""

    def __init__(self, mu, sigma, lower, upper):
        self.mu = mu
        self.sigma = sigma
        self.a = lower
        self.b = upper
        self.ss_dist = truncnorm(a=(self.a-self.mu)/self.sigma, b=(self.b-self.mu)/self.sigma, \
        	loc=self.mu, scale=self.sigma)

    def log_density(self, data):
        assert(len(data.shape) == 1), "Expect 1D data!"

        lower_cdf = norm.cdf(self.a, loc=self.mu, scale=self.sigma)
        upper_cdf = 1 - norm.cdf(self.b, loc=self.mu, scale=self.sigma)

        log_density = - ((data - self.mu) ** 2 / (2 * self.sigma ** 2) - np.log(self.sigma) \
        	- 0.5 * np.log(2 * np.pi) - np.log(1-(lower_cdf+upper_cdf)))
        log_density = np.where( np.logical_or(data < self.a, data > self.b) , -np.inf, log_density)
        return log_density

    def estimate_parameters(self, data, weights):
        assert(len(data.shape) == 1), "Expect 1D data!"

        wsum = np.sum(weights)

        moments = Moments(0, self.sigma, self.a-self.mu, self.b-self.mu)
        m_k = moments.first_moment
        H_k = self.sigma - moments.second_moment

        self.mu = np.sum(weights * data) / wsum - m_k
        self.sigma = np.sqrt(np.sum(weights * (data - self.mu) ** 2) / wsum + H_k)


    def __repr__(self):
        return "TruncNorm[μ={mu:.4g}, σ={sigma:.4g}]".format(mu=self.mu, sigma=self.sigma)
