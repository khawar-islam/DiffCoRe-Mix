import clip
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityChecker:
    @staticmethod
    def cosine_similarity(features1, features2):
        return cosine_similarity(features1, features2)[0][0]