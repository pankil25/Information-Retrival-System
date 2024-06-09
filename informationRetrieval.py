
# Add your import statements here

import math

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import gensim.downloader as api




class InformationRetrieval():

    def __init__(self):
        self.index = None
        self.types = None
        self.docIDs = None
        self.document_term_freq = None
        self.word_embeddings_model=None

    # -------------------------------------------------------------------------------------------------------------------

    # Vector Space Model:

    # -------------------------------------------------------------------------------------------------------------------
    def buildIndex(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        arg2 : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """

        index = {}

        # Initialize variables to store documents, inverted index, and unique terms
        merged_documents = []
        inverted_index = {}
        term_types = set()  # Use a set to store unique terms efficiently.

        document_term_freq = {}  # For counting frequency of word for each document

        # Merge sentences into documents and build inverted index
        for i in range(len(docs)):
            merged_doc = []
            term_freq = {}  # Initialize a dictionary to track term frequencies for the current document
            for sentence_list in docs[i]:
                merged_doc += sentence_list

                for term in sentence_list:
                    term_types.add(term)
                    # Update term frequency for the current document
                    term_freq[term] = term_freq.get(term, 0) + 1

                    # Build inverted index.
                    if term not in inverted_index:
                        inverted_index[term] = [docIDs[i]]
                    elif docIDs[i] not in inverted_index[term]:
                        inverted_index[term].append(docIDs[i])
            merged_documents.append(merged_doc)
            # Store term frequencies for the current document
            document_term_freq[i] = term_freq

        # Store the inverted index, unique terms, and document IDs in class variables

        self.index = inverted_index
        self.types = list(term_types)  # Convert set to list for consistency.
        self.docIDs = docIDs
        self.document_term_freq = document_term_freq



    def tf_idf_matrix_maker(self):
        """
        Calculate the TF-IDF matrix for the document collection.

        Returns
        -------
        np.array
            The TF-IDF matrix where each row represents a document and each column represents a term.
        """

        # Calculate the total number of documents (N) and total number of unique terms (n)
        N = len(self.docIDs)
        n = len(self.types)

        # Initialize a matrix to store TF-IDF scores for each term-document pair
        tf_idf_matrix = np.zeros((N, n))

        # Calculate TF-IDF scores for each term-document pair
        for i, term in enumerate(self.index):
            for doc_id in self.index[term]:
                # Calculate TF-IDF score for the current term and document
                tf_idf_matrix[self.docIDs.index(doc_id)][i] = self.document_term_freq[doc_id - 1][term] * math.log10(
                    N / len(self.index[term]))

        return tf_idf_matrix




    def query_inv_index_maker(self, queries):
        """
        Create an inverted index for the queries and calculate term frequencies.

        Parameters
        ----------
        queries : list
            A list of lists of lists where each sub-list is a query and each sub-sub-list is a sentence of the query.

        Returns
        -------
        tuple
            A tuple containing:
                - The inverted index for the queries.
                - Term frequencies for each query.
                - Merged queries.
        """

        Queries = []  # Initialize list to store merged queries.
        q_inv_idx = {}  # Initialize inverted index for queries.
        query_term_freq = {}  # Initialize a dictionary to store word frequency in each query.

        # Create inverted index and calculate term frequencies for queries
        for i in range(len(queries)):
            merged_queries = []  # Initialize list to store merged queries for current query
            term_freq = {}  # Initialize a dictionary to store term frequencies for current query

            # Merge sentences into queries and calculate term frequencies
            for query in queries[i]:
                merged_queries += query
                for term in self.index:
                    for word in query:
                        if term == word:
                            # Increment term frequency for the current query
                            term_freq[term] = term_freq.get(term, 0) + 1

                            # Update inverted index for the current term and query
                            if term not in q_inv_idx:
                                q_inv_idx[term] = [i]
                            elif i not in q_inv_idx[term]:
                                q_inv_idx[term].append(i) # for no repeatation of query ids

                        else:
                            # Set term frequency to 0 for terms not present in the current query
                            term_freq[term] = term_freq.get(term, 0)

                            # Initialize an empty list for terms not present in the current query in the inverted index
                            if term not in q_inv_idx:
                                q_inv_idx[term] = [] # for other words put null list to make easy multiplication

            Queries.append(merged_queries)  # Append merged queries to the list
            query_term_freq[i] = term_freq  # Store term frequencies for the current query

        return q_inv_idx, query_term_freq, Queries



    def query_tf_idf_maker(self, num_queries, q_inv_idx, query_term_freq):
        """
        Calculate TF-IDF scores for each term-query pair.

        Parameters
        ----------
        num_queries : int
            The total number of queries.
        q_inv_idx : dict
            The inverted index for the queries.
        query_term_freq : dict
            A dictionary containing term frequencies for each query.

        Returns
        -------
        numpy.ndarray
            An array containing TF-IDF scores for each term-query pair.
        """

        N = len(self.docIDs)  # Total number of documents
        n = len(self.types)  # Total number of unique terms
        query_tf_idf = np.zeros((num_queries, n))  # Initialize matrix to store TF-IDF scores for queries.

        # Calculate TF-IDF scores for each term-query pair
        for i, term in enumerate(q_inv_idx):
            for query_id in q_inv_idx[term]:
                # Calculate TF-IDF score for the current term-query pair
                query_tf_idf[query_id][i] = query_term_freq[query_id][term] * math.log10(N / len(self.index[term]))

        return query_tf_idf







    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of mthe query


        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """

        # Initialize list to store ordered document IDs.
        doc_IDs_ordered = []

        tf_idf_matrix = self.tf_idf_matrix_maker()


        q_inv_idx, query_term_freq, Queries = self.query_inv_index_maker(queries)

        num_queries = len(Queries)


        query_tf_idf = self.query_tf_idf_maker(num_queries, q_inv_idx,query_term_freq)

        # Iterate over each query
        for query_index in range(len(query_tf_idf)):
            # Create dictionary to store cosine similarity scores for the current query
            cosine_similarity_scores = {}
            ordered_document_ids = []

            for document_index in range(len(tf_idf_matrix)):
                # Calculate the dot product of tf-idf vectors for the current query and document
                dot_product = np.dot(tf_idf_matrix[document_index], query_tf_idf[query_index])

                # Calculate the norms of the tf-idf vectors for the current document and query
                document_norm = np.linalg.norm(tf_idf_matrix[document_index])
                query_norm = np.linalg.norm(query_tf_idf[query_index])

                # Check for zero or NaN values before division
                if np.isnan(dot_product) or np.isnan(document_norm) or np.isnan(query_norm):
                    cosine_similarity_scores[self.docIDs[document_index]] = np.nan
                elif dot_product == 0 or document_norm == 0 or query_norm == 0:
                    cosine_similarity_scores[self.docIDs[document_index]] = 0.0
                else:
                    # Calculate cosine similarity
                    cosine_similarity_scores[self.docIDs[document_index]] = dot_product / (document_norm * query_norm)

            # Sort the documents based on cosine similarity

            cosine_similarity_scores = sorted(cosine_similarity_scores.items(), key=lambda x: x[1], reverse=True)

            # Store the document IDs in the order of similarity
            for doc in cosine_similarity_scores:
                ordered_document_ids.append(doc[0])

            doc_IDs_ordered.append(ordered_document_ids)

        return doc_IDs_ordered




    # -------------------------------------------------------------------------------------------------------------------

    # LSA using SVD Model:

    # -------------------------------------------------------------------------------------------------------------------



    def buildIndex_svd(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        docs : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        docIDs : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """

        # Convert documents to corpus format
        corpus=[' '.join([' '.join(sentence) for sentence in doc]) for doc in docs]


        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(corpus)


        # Normalize document vectors
        X_normalized = normalize(X, norm='l2', axis=1)
        self.num_features = X.shape[1]



        # Apply SVD to reduce dimensionality
        self.svd_model = TruncatedSVD(n_components=500, random_state=42)
        self.index = self.svd_model.fit_transform(X_normalized)



    def rank_svd(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        queries : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query
        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """

        ranked_docs = []

        for query in queries:

            # Convert query to string
            query_str =' '.join([' '.join(term) for term in query])


            # Vectorize query
            query_vector = self.vectorizer.transform([query_str])


            # Transform query vector using SVD
            query_lsa = self.svd_model.transform(query_vector)

            # Compute cosine similarity between query and documents
            similarities = cosine_similarity(query_lsa, self.index)

            # Get document IDs sorted by relevance
            ranked_doc_ids = np.argsort(similarities[0])[::-1]

            ranked_doc_ids += 1

            # Append ranked document IDs to the result list
            ranked_docs.append(list(ranked_doc_ids))

        return ranked_docs





    # -------------------------------------------------------------------------------------------------------------------

    # Word2vec Model:

    # -------------------------------------------------------------------------------------------------------------------


    def buildIndex_word2vec(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        docs : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        docIDs : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """
        # Load pre-trained word embeddings model (e.g., Word2Vec)
        self.word_embeddings_model = api.load("word2vec-google-news-300")

        corpus = [' '.join([' '.join(sentence) for sentence in doc]) for doc in docs]


        # Tokenize the corpus into lists of words
        tokenized_corpus = [doc.split() for doc in corpus]

        # Compute document embeddings
        doc_embeddings = []
        for doc in tokenized_corpus:
            # Compute average word embedding for each document
            doc_embedding_temp =[self.word_embeddings_model[word] for word in doc if word in self.word_embeddings_model]
            if not doc_embedding_temp:
                # If there are no valid embeddings, skip this query
                continue
            doc_embedding = np.mean(doc_embedding_temp,axis=0)

            doc_embeddings.append(doc_embedding)

        self.index=doc_embeddings



    def rank_word2vec(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        queries : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query
        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """

        ranked_docs = []
        for query in queries:
            query = ' '.join([' '.join(term) for term in query])
            # Compute average word embedding for the query

            # Filter out words without embeddings
            query_embeddings = [self.word_embeddings_model[word] for word in query if word in self.word_embeddings_model]
            if not query_embeddings:
                # If there are no valid embeddings, skip this query
                continue

            # Compute the mean of the valid embeddings
            query_embedding = np.mean(query_embeddings, axis=0)*100


            if np.any(np.isnan(query_embedding)):
                # Skip this query if the embedding contains NaN values
                continue

            # Flatten the query embedding to a 1D array
            query_embedding = query_embedding.flatten()

            # Compute cosine similarities
            doc_similarities = []

            for doc_embedding in self.index:
                if np.any(np.isnan(doc_embedding)):
                    # Skip this document if the embedding contains NaN values
                    continue

                # Flatten the document embedding to a 1D array
                doc_embedding = doc_embedding.flatten()

                similarity = cosine_similarity([query_embedding], [doc_embedding])
                doc_similarities.append(similarity[0][0])






            # Get document IDs sorted by relevance
            ranked_doc_ids = np.argsort(doc_similarities)[::-1]

            ranked_doc_ids += 1

            # Append ranked document IDs to the result list
            ranked_docs.append(list(ranked_doc_ids))

        return ranked_docs






