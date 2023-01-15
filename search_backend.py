
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict,Counter
import re
import nltk
import pickle
import numpy as np
nltk.download('stopwords')
from nltk.corpus import stopwords
from tqdm import tqdm
import operator
from itertools import islice,count
from contextlib import closing
import json
from io import StringIO
from pathlib import Path
from operator import itemgetter
import pickle
import matplotlib.pyplot as plt
import math
import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import os
import re
from operator import itemgetter
from pathlib import Path
import pickle
from contextlib import closing
import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
import math
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from timeit import timeit
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from google.cloud import storage


import pyspark

from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext
from pyspark import SparkConf, SparkFiles
from pyspark.sql import SQLContext
from graphframes import *
from inverted_index_gcp import *
from BM25 import *



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~inverted body~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""Retrieving the index representing the body of articles from the bucket it is stored in"""

client_body=storage.Client()
bucket_body=client_body.get_bucket("shalevandmoria_body")
blob_body=bucket_body.get_blob(f'inverted_index_body/index_body.pkl')
pickle_in_body=blob_body.download_as_string()
inverted_body=pickle.loads(pickle_in_body)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~inverted title~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""Retrieving the index representing the title of articles from the bucket it is stored in"""

client_title=storage.Client()
bucket_title=client_title.get_bucket("shalevandmoria_title")
blob_title=bucket_title.get_blob(f'inverted_index_title/index_title.pkl')
pickle_in_title=blob_title.download_as_string()
inverted_title=pickle.loads(pickle_in_title)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~inverted anchor~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""Retrieving the index representing the anchor text of articles from the bucket it is stored in"""

client_anchor=storage.Client()
bucket_anchor=client_anchor.get_bucket("shalevandmoria_anchor")
blob_anchor=bucket_anchor.get_blob(f'inverted_index_anchor/index_anchor.pkl')
pickle_in_anchor=blob_anchor.download_as_string()
inverted_anchor=pickle.loads(pickle_in_anchor)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ pkl files ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
client_pkl=storage.Client()
bucket_pkl=client_pkl.get_bucket("pkl_files")
# DL
blob_DL=bucket_pkl.get_blob(f'DL.pkl')
pickle_in_DL=blob_DL.download_as_string()
DL=pickle.loads(pickle_in_DL)
# DL_title
blob_DL_title=bucket_pkl.get_blob(f'DL_title.pkl')
pickle_in_DL_title=blob_DL_title.download_as_string()
DL_title=pickle.loads(pickle_in_DL_title)
# norma
blob_norma=bucket_pkl.get_blob(f'norma.pkl')
pickle_in_norma=blob_norma.download_as_string()
norma=pickle.loads(pickle_in_norma)
# pr
blob_pr=bucket_pkl.get_blob(f'pr.pkl')
pickle_in_pr=blob_pr.download_as_string()
pr=pickle.loads(pickle_in_pr)
# pv
blob_pv=bucket_pkl.get_blob(f'pv.pkl')
pickle_in_pv=blob_pv.download_as_string()
pv=pickle.loads(pickle_in_pv)
# id_title
blob_id_and_title=bucket_pkl.get_blob(f'id_and_title.pkl')
pickle_in_id_and_title=blob_id_and_title.download_as_string()
id_title=pickle.loads(pickle_in_id_and_title)


RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stopwords_frozen = frozenset(stopwords.words('english'))
def tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.
    
    Parameters:
    -----------
    text: string , represting the text to tokenize.    
    
    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    return [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in stopwords_frozen]

def generate_query_tfidf_vector(query_to_search,index):
    """ 
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well. 
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.    

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.'). 
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.    
    
    Returns:
    -----------
    vectorized query with tfidf scores
    """
    
    epsilon = .0000001
    total_vocab_size = len(np.unique(query_to_search))
    Q = np.zeros((total_vocab_size))
    counter = Counter(query_to_search)
    i=0
    for token in np.unique(query_to_search):
        if token in index.df.keys(): #avoid terms that do not appear in the index.              
            tf = counter[token]/len(query_to_search) # term frequency divded by the length of the query
            df = index.df[token]  
            idf = math.log((len(DL))/(df+epsilon),10) #smoothing
            Q[i]=tf*idf
            i=i+1
    return Q

def dic_doc_score_cosine(query_to_search,inverted):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list
    and calculate cosine similarity.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                    Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']
    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: doc_id
                                                               value: cosine score.
    """
    unique_q=list(np.unique(query_to_search))
    candidates=defaultdict(lambda: [0]*len(unique_q))
    for term in unique_q:
        if term in inverted.df.keys():  
            list_of_doc =read_posting_list(inverted, term,"body")
            for doc_id, freq in list_of_doc:
                freq_=freq/DL[doc_id]
                logg=math.log(len(DL)/inverted.df[term],10)
                mull=freq_*logg
                candidates[doc_id][unique_q.index(term)]=mull
    Q=generate_query_tfidf_vector(query_to_search,inverted)
    np_q=np.array(list(Q))
    norma_query=np.linalg.norm(np_q)
    for doc_id in candidates.keys():
        candidates[doc_id]=float(np.sum(np.array(candidates[doc_id])*np_q)/norma[doc_id]*norma_query)
    return candidates

def get_top_n(sim_dict,N=3):
    """ 
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores 
   
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
    return sorted([(doc_id,np.round(score,5)) for doc_id, score in sim_dict.items()], key = lambda x: x[1],reverse=True)[:N]

bm25_title = BM25_from_index(inverted_title, DL_title,"title")
bm25_body = BM25_from_index(inverted_body, DL,"body")


def search_backend(query):
    ''' Returns up to a 100 search results for the query using BM25 OF THE BODY AND TITLE OF ARTICLES.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, score).
    '''
    res = []
    if len(query) == 0:
        return res
    # BEGIN SOLUTION
    tokens_query=tokenize(query)
    bm25_queries_score_train_title = bm25_title.search(tokens_query,100)
    bm25_queries_score_train_body = bm25_body.search(tokens_query,100)
    res=merge_results(bm25_queries_score_train_title,bm25_queries_score_train_body,0.48,0.52,N = 100)
    res= list(map(lambda x: (x[0],x[1]+1.5*pr[x[0]]),res))
    res=sorted(res, key=lambda x: x[1], reverse=True)
    # END SOLUTION
    return res

def search_body_backend(query):
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, score).
    '''
    res = []
    if len(query) == 0:
      return res
    # BEGIN SOLUTION
    tokens_query=tokenize(query)
    dic_cosin=dic_doc_score_cosine(tokens_query,inverted_body)
    res=get_top_n(dic_cosin,100)
    # END SOLUTION
    return res

def search_title_backend(query):
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query).
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, score).
    '''
    res = []
    if len(query) == 0:
        return res
    # BEGIN SOLUTION
    tokens_query=tokenize(query)
    unique_q=list(np.unique(tokens_query))
    candidates=defaultdict(lambda: [0]*len(unique_q))
    for term in unique_q:
        if term in inverted_title.df.keys():  
            list_of_doc =read_posting_list(inverted_title, term,"title")
            for doc_id, freq in list_of_doc:
                candidates[doc_id][unique_q.index(term)]=1   
    for doc_id in candidates.keys():
        total = 0
        for number in candidates[doc_id]:
            total += number
        candidates[doc_id]=total
    res=sorted([(doc_id,score) for doc_id, score in candidates.items()], key = lambda x: x[1],reverse=True)
    # END SOLUTION
    return res

def search_anchor_backend(query):
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query).
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, score).
    '''
    res = []
    if len(query) == 0:
      return res
    # BEGIN SOLUTION
    tokens_query=tokenize(query)
    unique_q=list(np.unique(tokens_query))
    candidates=defaultdict(lambda: [0]*len(unique_q))
    for term in unique_q:
        if term in inverted_anchor.df.keys():  
            list_of_doc =read_posting_list(inverted_anchor, term,"anchor")
            for doc_id, freq in list_of_doc:
                candidates[doc_id][unique_q.index(term)]=1   
    for doc_id in candidates.keys():
        total = 0
        for number in candidates[doc_id]:
            total += number
        candidates[doc_id]=total
    res=sorted([(doc_id,score) for doc_id, score in candidates.items()], key = lambda x: x[1],reverse=True)
    # END SOLUTION
    return res

def get_pagerank_backend(wiki_ids):
    ''' Returns PageRank values for a list of provided wiki article IDs.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    if len(wiki_ids) == 0:
      return res
    # BEGIN SOLUTION
    for i in wiki_ids:
        res.append(pr[i])
    # END SOLUTION
    return res

def get_pageview_backend(wiki_ids):
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    if len(wiki_ids) == 0:
      return res
    # BEGIN SOLUTION
    for i in wiki_ids:
      res.append(pv[i])
    # END SOLUTION
    return res


