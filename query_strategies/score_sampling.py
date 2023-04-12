import numpy as np
from .strategy import Strategy
from .builder import STRATEGIES
import torch


@STRATEGIES.register_module()
class ScoreBasedSampling(Strategy):
	def __init__(self, dataset, args, logger, timestamp, work_dir, n_drop=1):
		super(ScoreBasedSampling, self).__init__(dataset, args, logger, timestamp, work_dir)
		self.n_drop = n_drop
		# Don't forget to set dropout rate when n_drop>1!

	def query(self, n):
		idxs_u = np.arange(len(self.dataset.DATA_INFOS['train_full_main_category']))[self.dataset.INDEX_ULB]
		aesthetic_scores = self.predict(self.scoring_net, split='train_full_main_category', metric='aesthetic_score')
		for idx_u in idxs_u:
			self.dataset.DATA_INFOS['train_full_main_category'][int(idx_u)]['aesthetic_score'] = \
				int(aesthetic_scores[idx_u])
		classifier_scores = self.predict(self.cls_net, split='train_full_main_category', metric='tag_matching_score')
		total_score = (aesthetic_scores + classifier_scores)[idxs_u]
		return idxs_u[total_score.sort()[1][-n:].cpu().numpy()]
