import numpy as np
from .strategy import Strategy
from .builder import STRATEGIES


@STRATEGIES.register_module()
class RandomSampling(Strategy):
	def __init__(self, dataset, args, logger, timestamp, work_dir):
		super(RandomSampling, self).__init__(dataset, args, logger, timestamp, work_dir)

	def query(self, n):
		"""Randomly query samples in the unlabeled pool.

			:param n: (int)The number of samples to query.

			Returns:
				np.ndarray[int]: The indices of queried samples.

		"""
		return np.random.choice(np.where(self.dataset.INDEX_LB == 0)[0], n, replace=False)
