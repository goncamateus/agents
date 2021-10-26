import numpy as np


class OrnsteinUhlenbeckNoise:
    """Add Ornstein-Uhlenbeck noise to continuous actions.

        https://en.wikipedia.org/wiki/Ornstein-Uhlenbeck_process_

        Parameters
        ----------
        mu : float or ndarray, optional

            The mean towards which the Ornstein-Uhlenbeck process should revert; must be
            broadcastable with the input actions.

        sigma : positive float or ndarray, optional

            The spread of the noise of the Ornstein-Uhlenbeck process; must be
            broadcastable with the input actions.

        theta : positive float or ndarray, optional

            The (element-wise) dissipation rate of the Ornstein-Uhlenbeck process; must
            be broadcastable with the input actions.

        min_value : float or ndarray, optional

            The lower bound used for clipping the output action; must be broadcastable with the input
            actions.

        max_value : float or ndarray, optional

            The upper bound used for clipping the output action; must be broadcastable with the input
            actions.

        random_seed : int, optional

            Sets the random state to get reproducible results.
    """

    def __init__(
            self,
            mu=0.,
            sigma=1.,
            theta=0.15,
            min_value=None,
            max_value=None,
            random_seed=None
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.min_value = -1e15 if min_value is None else min_value
        self.max_value = 1e15 if max_value is None else max_value
        self.random_seed = random_seed
        self.rnd = np.random.RandomState(self.random_seed)
        self.reset()

    def reset(self):
        """Reset the Ornstein-Uhlenbeck process."""
        self._noise = None

    def __call__(self, a):
        """Add some Ornstein-Uhlenbeck to a continuous action.

            Parameters
            ----------
            a : action

                A single action

            Returns
            -------
            a_noisy : action

                An action with OU noise added
        """
        a = np.asarray(a)
        if self._noise is None:
            self._noise = np.ones_like(a) * self.mu

        white_noise = np.asarray(self.rnd.randn(*a.shape), dtype=a.dtype)
        self._noise += self.theta * \
            (self.mu - self._noise) + self.sigma * white_noise
        self._noise = np.clip(self._noise, self.min_value, self.max_value)
        return a + self._noise
