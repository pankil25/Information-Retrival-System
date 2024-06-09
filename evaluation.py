
# Add your import statements here
import math




class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):

		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = 0  # Initialize precision.

		# Initialize variables to count relevant documents and retrieved documents
		relevant_retrieved = 0
		retrieved = 0

		# Iterate over the top k retrieved document IDs
		for doc_id in query_doc_IDs_ordered[:k]:
			retrieved += 1

			# Check if the retrieved document is relevant
			if doc_id in true_doc_IDs:
				relevant_retrieved += 1

		# Calculate precision
		precision = relevant_retrieved / retrieved

		return precision




	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = 0  # Initialize mean precision.

		# Initialize variable to store the sum of precision values and count of queries
		total_precision = 0.0
		num_queries = len(query_ids)

		# Iterate over each query and corresponding document IDs ordered
		for i in range(num_queries):
			query_id = query_ids[i]
			query_doc_IDs = doc_IDs_ordered[i]

			# Get relevant document IDs for the current query from qrels
			relevant_docs = []
			for qrel in qrels:
				if int(qrel['query_num']) == query_id:
					relevant_docs.append(int(qrel['id']))
				if len(relevant_docs) >= k:
					break

			# Calculate precision for the current query
			precision = self.queryPrecision(query_doc_IDs, query_id, relevant_docs, k)

			# Add precision value to the total sum
			total_precision += precision

		# Calculate mean precision
		meanPrecision = total_precision / num_queries

		return meanPrecision




	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""
		recall = 0  # Initialize recall.

		# Initialize variables to count relevant documents and total relevant documents
		relevant_retrieved = 0
		total_relevant = len(true_doc_IDs)

		# Iterate over the top k retrieved document IDs
		for doc_id in query_doc_IDs_ordered[:k]:
			# Check if the retrieved document is relevant
			if doc_id in true_doc_IDs:
				relevant_retrieved += 1

		# Calculate recall
		recall = relevant_retrieved / total_relevant

		return recall



	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = 0  # Initialize mean recall.

		# Initialize variable to store the sum of recall values and count of queries
		total_recall = 0.0
		num_queries = len(query_ids)

		# Iterate over each query and corresponding document IDs ordered
		for i in range(num_queries):
			query_id = query_ids[i]
			query_doc_IDs = doc_IDs_ordered[i]

			# Get relevant document IDs for the current query from qrels
			relevant_docs = []
			for qrel in qrels:
				if int(qrel['query_num']) == query_id:
					relevant_docs.append(int(qrel['id']))
				if len(relevant_docs) >= k:
					break

			total_relevant = len(relevant_docs)

			# Calculate recall for the current query
			recall = 0.0
			if total_relevant > 0:
				recall = self.queryRecall(query_doc_IDs, query_id, relevant_docs, k)

			# Add recall value to the total sum
			total_recall += recall

		# Calculate mean recall
		meanRecall = total_recall / num_queries

		return meanRecall

	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""
		fscore = 0  # Initialize F-score.

		# Calculate precision and recall for the given query and k
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

		# Calculate F-score using the formula: 2 * (precision * recall) / (precision + recall)
		if precision + recall > 0:
			fscore = 2 * (precision * recall) / (precision + recall)
		else:
			fscore = 0.0

		return fscore




	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = 0  # Initialize mean F-score.

		# Initialize variable to store the sum of F-score values and count of queries
		total_fscore = 0.0
		num_queries = len(query_ids)

		# Iterate over each query and corresponding document IDs ordered
		for i in range(num_queries):
			query_id = query_ids[i]
			query_doc_IDs = doc_IDs_ordered[i]

			# Get relevant document IDs for the current query from qrels
			relevant_docs = []
			for qrel in qrels:
				if int(qrel['query_num']) == query_id:
					relevant_docs.append(int(qrel['id']))
				if len(relevant_docs) >= k:
					break

			# Calculate F-score for the current query
			fscore = self.queryFscore(query_doc_IDs, query_id, relevant_docs, k)

			# Add F-score value to the total sum
			total_fscore += fscore

		# Calculate mean F-score
		meanFscore = total_fscore / num_queries

		return meanFscore

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""


		nDCG = 0  # Initialize nDCG.


		# Collect relevant documents up to rank k
		relevant_docs = []
		for doc_id in query_doc_IDs_ordered[:k]:
			if doc_id in true_doc_IDs:
				relevant_docs.append(doc_id)


		# Compute relevance scores for each document
		relevance_scores = []
		for doc_id in query_doc_IDs_ordered[:k]:
			relevance_score = 0  # Default relevance score is 0

			if doc_id in relevant_docs:

				# In many relevance assessment scenarios, relevance is often graded on a scale, commonly ranging from 0 to 4 or 0 to 5: where higher numbers indicate greater relevance.
				# For example: 0: Not relevant 1: Weakly relevant 2: Fairly relevant 3: Highly relevant 4: Perfectly relevant

				relevance_score = 5 - relevant_docs.index(doc_id)  # Compute relevance score for the document
			relevance_scores.append(relevance_score)  # Append relevance score to the list

		# Compute ideal relevance scores
		sorted_relevance_scores = sorted(relevance_scores, reverse=True)

		# Compute DCG
		DCG = sum([relevance_scores[i] / math.log2(i + 2) for i in range(min(k, len(query_doc_IDs_ordered)))])

		# Compute ideal DCG
		IDCG = sum([sorted_relevance_scores[i] / math.log2(i + 2) for i in range(min(k, len(query_doc_IDs_ordered)))])

		# Compute nDCG
		nDCG = DCG / IDCG if IDCG > 0 else 0  # Avoid division by zero


		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = 0.0  # Initialize mean nDCG

		# Iterate over each query and compute nDCG
		for i in range(len(query_ids)):
			query_id = query_ids[i]
			query_doc_IDs = doc_IDs_ordered[i]

			# Extract relevant document IDs for the current query from qrels
			true_doc_IDs = []
			for qrel in qrels:
				if int(qrel['query_num']) == query_id:

					true_doc_IDs.append(int(qrel['id']))

				if len(true_doc_IDs) >= k:
					break

			# Calculate nDCG for the current query and add to meanNDCG
			nDCG = self.queryNDCG(query_doc_IDs, query_id, true_doc_IDs, k)
			meanNDCG += nDCG

		# Compute the mean nDCG over all queries
		if len(query_ids) > 0:
			meanNDCG /= len(query_ids)

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""


		avgPrecision = 0.0  # Initialize average precision

		# Initialize variables to compute precision@i and relevant_count
		precision_at_i_sum = 0.0
		relevant_count = 0

		# Iterate over the top k documents
		for i in range(len(query_doc_IDs_ordered)):
			doc_id = query_doc_IDs_ordered[i]
			if doc_id in true_doc_IDs:
				relevant_count += 1
				precision_at_i = relevant_count / (i + 1)  # Calculate precision@i
				precision_at_i_sum += precision_at_i
			if relevant_count >=k :
				break

		# Compute average precision if there are relevant documents
		if relevant_count > 0:
			avgPrecision = precision_at_i_sum / k


		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = 0.0  # Initialize mean average precision

		# Iterate over each query and compute average precision
		for i in range(len(query_ids)):
			query_id = query_ids[i]
			query_doc_IDs_ordered = doc_IDs_ordered[i]

			# Extract relevant document IDs for the current query from qrels
			true_doc_IDs = []
			for qrel in q_rels:
				if int(qrel['query_num']) == query_id:

					true_doc_IDs.append(int(qrel['id']))

				if len(true_doc_IDs) >= k:
					break

			# Compute average precision for the current query
			avg_precision = self.queryAveragePrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
			meanAveragePrecision += avg_precision

		# Compute the mean average precision over all queries
		if len(query_ids) > 0:
			meanAveragePrecision /= len(query_ids)

		return meanAveragePrecision




