import pyspark
import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
from google.cloud import storage
from collections import defaultdict
from contextlib import closing
import numpy as np
from google.cloud.storage import bucket
import math
from itertools import chain
import time
from inverted_index_gcp import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   BM25_from_index   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_top_n(sim_dict, N=3):
    """
    Sort and return the highest N documents according to the score.

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """

    return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]


# When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.
class BM25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index

    t: String. type of index- can be "body, "title or "anchor"
    """

    def __init__(self, index, DL, t, k1=1.5, b=0.75):
        self.type = t
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(DL)
        self.DL = DL
        self.AVGDL = sum(DL.values()) / self.N

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            n_ti = self.index.df[term]
            idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
        return idf

    def search(self, query, N=3):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """

        scores = {}
        q = []
        dict_pls = {}
        candidates = []
        for i in np.unique(query):
            if i in self.index.df.keys():
                q.append(i)
                pls_term = read_posting_list(self.index, i, self.type)
                candidates = candidates + list(map(lambda x: x[0], pls_term))
                dict_pls[i] = Counter(dict(pls_term))
        self.idf = self.calc_idf(q)
        candidates = np.unique(candidates)
        for key in candidates:
            scores[key] = self._score(q, key, dict_pls)
        return get_top_n(scores, N)



    def _score(self, query, doc_id, dict_pls):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        score = 0.0
        doc_len = self.DL[doc_id]
        for term in query:
            freq = dict_pls[term][doc_id]
            numerator = self.idf[term] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
            score += (numerator / denominator)
        return score


def merge_results(title_scores, body_scores, title_weight=0.3, text_weight=0.3, N=3):
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).

    Parameters:
    -----------
    title_scores: list of pairs in the following format:(doc_id,score), build upon the title index

    body_scores: list of pairs in the following format:(doc_id,score), build upon the body/text index
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    list of topN pairs in the following format:(doc_id,score).
    """
    # YOUR CODE HERE
    counter_title = Counter(dict(title_scores))
    counter_body = Counter(dict(body_scores))
    counter_result = {}
    for doc_id in counter_title:
        counter_result[doc_id] = counter_title[doc_id] * title_weight + counter_body[doc_id] * text_weight
    for doc_id in counter_body:
        counter_result[doc_id] = counter_title[doc_id] * title_weight + counter_body[doc_id] * text_weight
    return sorted([(doc_id, score) for doc_id, score in counter_result.items()], key=lambda x: x[1], reverse=True)[:N]

