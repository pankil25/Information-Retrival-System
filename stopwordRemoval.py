

# Add your import statements here

import nltk
from nltk.corpus import stopwords

#for bottomup
from collections import Counter


class StopwordRemoval():

	def fromList(self, text):

		# Initialize an empty list to store text with stopwords removed
		stopwordRemovedText = []
		# Get the list of stopwords from the NLTK library
		stop_words = list(stopwords.words('english'))





		# Iterate over each word in the input text
		for word in text:
			# Initialize an empty list to store the filtered document
			filtered_doc=[]
			# Check if the word is not a stopword
			if word not in stop_words:
				# If the word is not a stopword, add it to the filtered document list
				filtered_doc.append(word)

				# Append the filtered document list to the stopwordRemovedText list
				stopwordRemovedText.append(filtered_doc)

		# Return the text with stopwords removed
		return stopwordRemovedText


#------------------------------------------------------------------------------
# Bottom Up Approach
#-------------------------------------------------------------------------------



	def bottomup(self,text):
		# Tokenization
		tokenized_corpus = [word for document in text for word in nltk.word_tokenize(document.lower())]

		# Count word frequencies
		word_freq = Counter(tokenized_corpus)

		# Identify first 20 common words this list will be treated as stop words
		common_words = [word for word, freq in word_freq.most_common(20)]

		# Initialize an empty list to store text with stopwords removed
		stopwordRemovedText_2 = []
		# Iterate over each document in the input text
		for doc in text:
			filtered_doc = []
			# Iterate over each token in the document
			for token in doc:
				# Check if the token is not in the list of common words
				if token not in common_words:
					# If the token is not in the list of common words, add it to the filtered document
					filtered_doc.append(token)
			# Append the filtered document to the stopwordRemovedText_2 list
			stopwordRemovedText_2.append(filtered_doc)
		# Return the text with stopwords removed
		return stopwordRemovedText_2











	