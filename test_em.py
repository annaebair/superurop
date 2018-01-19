import pytest
import numpy as np
import scipy.stats as s
from em import Moments, TruncatedNormalDistribution

class TestEM:

	################ Moments #################	

	def test_first_moment_standard_normal(self):
		first = Moments(0, 1, -np.inf, np.inf).first_moment
		assert first == 0

	def test_second_moment_standard_normal(self):
		second = Moments(0, 1, -np.inf, np.inf).second_moment
		assert second == 1.0

	def test_first_moment_positive_mean(self):
		first = Moments(5, 3, -np.inf, np.inf).first_moment
		assert first == 5

	def test_first_moment_negative_mean(self):
		first = Moments(-5, 3, -np.inf, np.inf).first_moment
		assert first == -5

	def test_second_moment_positive_mean(self):
		mu = 5
		sigma = 2
		a = -np.inf
		b = np.inf
		actual = Moments(mu=mu, sigma=sigma, a=a, b=b).second_moment
		expected = mu ** 2 + sigma ** 2
		assert actual == expected

	def test_second_moment_negative_mean(self):
		mu = -4
		sigma = 2
		a = -np.inf
		b = np.inf
		actual = Moments(mu=mu, sigma=sigma, a=a, b=b).second_moment
		expected = mu ** 2 + sigma ** 2
		assert actual == expected

	def test_second_moment_positive(self):
		mu = -4
		sigma = 2
		a = -np.inf
		b = np.inf
		actual = Moments(mu=mu, sigma=sigma, a=a, b=b).second_moment
		assert actual > 0

	def test_second_moment_positive_truncated(self):
		mu = 4
		sigma = 2
		a = -2
		b = 5
		actual = Moments(mu=mu, sigma=sigma, a=a, b=b).second_moment
		assert actual > 0

	def test_second_moment_positive_right_of_mean(self):
		mu = 4
		sigma = 2
		a = 5
		b = 7
		actual = Moments(mu=mu, sigma=sigma, a=a, b=b).second_moment
		assert actual > 0

	def test_second_moment_positive_left_of_mean(self):
		mu = 4
		sigma = 2
		a = 0
		b = 1
		actual = Moments(mu=mu, sigma=sigma, a=a, b=b).second_moment
		assert actual > 0

	################ Truncated log_density #################

	def test_log_density_trunc_norm(self):
		pass

	def test_log_density_trunc_right(self):
		pass

	def test_log_density_trunc_left(self):
		pass

	################ Censored log_density #################

	def test_log_density_cens_norm(self):
		pass

	def test_log_density_cens_right(self):
		pass

	def test_log_density_cens_left(self):
		pass

	################ Truncated estimate_parameters #################	

	def test_estimate_parameters_trunc_norm(self):
		pass

	def test_estimate_parameters_trunc_right(self):
		pass

	def test_estimate_parameters_trunc_left(self):
		pass

	################ Censored estimate_parameters #################	

	def test_estimate_parameters_cens_norm(self):
		pass

	def test_estimate_parameters_cens_right(self):
		pass

	def test_estimate_parameters_cens_left(self):
		pass


