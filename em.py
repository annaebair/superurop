from scipy.stats import norm
import scipy.stats
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
		self.first_moment = self._first_moment()
		self.second_moment = self._second_moment()

	def _first_moment(self):
		a_pdf = norm.pdf(self.a, loc=self.mu, scale=self.sigma)
		b_pdf = norm.pdf(self.b, loc=self.mu, scale=self.sigma)
		a_cdf = norm.cdf(self.a, loc=self.mu, scale=self.sigma)
		b_cdf = norm.cdf(self.b, loc=self.mu, scale=self.sigma)

		M1 = self.mu - self.sigma*(b_pdf-a_pdf)/(b_cdf-a_cdf)
		return M1

	def _second_moment(self):
		a_pdf = norm.pdf(self.a, loc=self.mu, scale=self.sigma)
		b_pdf = norm.pdf(self.b, loc=self.mu, scale=self.sigma)
		a_pdf_der = -a_pdf*self.a
		b_pdf_der = -b_pdf*self.b
		a_cdf = norm.cdf(self.a, loc=self.mu, scale=self.sigma)
		b_cdf = norm.cdf(self.b, loc=self.mu, scale=self.sigma)

		derivative_term = (b_pdf_der - a_pdf_der)/(b_cdf-a_cdf)

		M2 = self.sigma**2 + self.mu**2 + self.sigma**2 * derivative_term - 2*self.mu*self.sigma*self.first_moment
		return M2


class TruncatedNormalDistribution(Distribution):
    """Truncated normal distribution with parameters (mu, sigma)."""

    def __init__(self, mu, sigma, lower, upper):
        self.mu = mu
        self.sigma = sigma
        self.a = lower
        self.b = upper
        self.first_moment = Moments(self.mu, self.sigma, self.a, self.b).first_moment
        self.second_moment = Moments(self.mu, self.sigma, self.a, self.b).second_moment

    def log_density(self, data):
        assert(len(data.shape) == 1), "Expect 1D data!"

        return - (data - self.mu) ** 2 / (2 * self.sigma ** 2) - np.log(self.sigma) - 0.5 * np.log(2 * np.pi)

    def estimate_parameters(self, data, weights):
        assert(len(data.shape) == 1), "Expect 1D data!"

        wsum = np.sum(weights)

        moments = Moments(0, self.sigma, self.lower-self.mu, sef.upper-self.mu)
        m_k = moments.first_moment
        H_k = self.sigma - moments.second_moment

        self.mu = np.sum(weights * data) / wsum
        self.sigma = np.sqrt(np.sum(weights * (data - self.mu) ** 2) / wsum)

    def __repr__(self):
        return "Norm[μ={mu:.4g}, σ={sigma:.4g}]".format(mu=self.mu, sigma=self.sigma)


data1 = np.random.lognormal(mean=0, sigma=0.75, size=1000)
data2 = np.random.lognormal(mean=10, sigma=0.9, size=1000)
data3 = np.concatenate((data2, data1))
print(data3)

plt.hist(data3, normed=True, bins=1000)
# plt.xlim(0, 1000)
plt.show()

mean_1, stdev_1 = norm.fit(data1)
mean, std = norm.fit(data3)

dist1 = mixem.distribution.LogNormalDistribution(0, 2)
dist2 = mixem.distribution.LogNormalDistribution(11, 3)
print(dist2.sigma)
print(dist2.mu)

weights, distributions, log_l = mixem.em(data3, [dist1, dist2], max_iterations=200)

print("Weights: ", weights)
print("Distributions: ", distributions)
print("Log Likelihood: ", log_l)

prob = mixem.probability(data3, weights, distributions)
print("Probability: ", len(prob))