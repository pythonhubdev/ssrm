from scipy.stats import multinomial
from ssrm_test import ssrm_test as st

# # Set the seed of our random number generator for reproducibility. Don't worry about this
# np.random.seed(0)

# # Our intended allocation probabilities
# p_0 = [0.1, 0.5, 0.4]

# # The actual allocation probabilities
# p = [0.1, 0.49, 0.41]

# # Specify number of visitors
# n = 10000

# # Generate allocations
# data = multinomial.rvs(1, p, size=n)


class SSRMSuite:
    def setup(self):
        self.intended_p_alloc = [0.1, 0.5, 0.4]
        self.actual_p_alloc = [0.1, 0.49, 0.41]
        self.n = 10000
        self.data = multinomial.rvs(1, self.actual_p_alloc, size=self.n)

    def time_sequential_posteriors(self):
        st.sequential_posteriors(self.data, self.intended_p_alloc)

    def time_sequential_posterior_probabilities(self):
        st.sequential_posterior_probabilities(self.data, self.intended_p_alloc)

    def time_sequential_p_values(self):
        st.sequential_p_values(self.data, self.intended_p_alloc)

    def time_sequential_bayes_factors(self):
        st.sequential_bayes_factors(self.data, self.intended_p_alloc)

    def time_srm_test(self):
        st.srm_test(self.data, self.intended_p_alloc)
