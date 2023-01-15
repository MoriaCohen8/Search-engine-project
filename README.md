# Search-engine-project
# Information recovery on Wikipedia
In this project we implemented a search engine for all the documents in the English Wikipedia.
The project was carried out under the "Information Retrieval" course at Ben Gurion University

## The Code Structure

| py file | Description | Methods |
| --- | --- | --- |
| search_frotened | Search is used here in the search functions by using the functions that contain the implementation. | *search-search results for the query using BM25. <br> *search_body-search results for the query of the body of articles only. <br> *search_title-search results that contain a query word in the title of articles. <br> *search_anchor- earch results that contain a query word in the anchor text of articles. <br> *get_pagerank-  PageRank values for a list of provided wiki article IDs.  <br> *get_pageview-number of page views that each of the provide wiki articles had in August 2021.|
| search_backned | Here the search is carried out by calling auxiliary functions and calculations. | *tokenize-list of tokens to text <br> *generate_query_tfidf_vector- Generate a vector representing the query. <br>*dic_doc_score_cosine- Returns its cosine score for each document that matches the query. <br> *get_top_n- Sort and return the highest N documents according to the score. <br>*search_backend-Returns up to a 100 of best search results for the query using BM25. <br>*search_body_backend- Returns up to a 100 search results for the query using tfidf and cosine similarity of the body of articles only. <br>*search_title_backend- Returns all search results that contain a query word in the title of articles, ordered in descending order of the number of distinct query words that appear in the title. <br> *search_anchor_backend- Returns all search results that contain a query word in the anchor text of articles, ordered in descending order of the number of query words that appear in anchor text linking to the page. <br>*get_pagerank_backend-  Returns PageRank values for a list of provided wiki article IDs.  <br> *get_pagerank_backend-Returns the number of page views that each of the provide wiki articles had in August 2021.|
| inverted_index_gcp | A file containing the classes MultiFileWriter, MultiFileReader, InvertedIndex with which we created the indexes and help us use them | There are the functions of the classes and the read_posting_list function that returns the posting list of a word in the received index |
| BM25 | calculate of the BM25 | There are the functions of the BM25 class and the functions: <br>*get_top_n- Sort and return the highest N documents according to the score. <br> *merge_results - merge and sort documents retrieved by its weight score|


| Classes | Methods |
| --- | --- |
| BM25_from_index | *calc_idf <br> *search <br> *_score|
| InvertedIndex | *add_doc <br> *write_index <br> *_write_globals <br> *read_index <br> *delete_index <br> *write_a_posting_list <br>*upload_posting_locs |
| MultiFileWriter | *write <br> *close <br> *_upload_to_gcp |
| MultiFileReader | *read <br> *close <br> *exit |


| Pickle File's Name | The Information Inside |
| --- | --- |
| id_title | A dictionary that maps a document ID to the title of that document |
| pv | A dictionary that maps between a document ID and its page view |
| pr | A dictionary that maps between a document ID and its page rank |
| DL_Title | A dictionary that maps a document ID to the length of its title |
| DL | A dictionary that maps a document ID to the length of its body |
| norma | A dictionary that maps between a document ID and the size of the vector representing that document |
