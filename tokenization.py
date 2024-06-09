
import nltk.data
import re
from nltk.tokenize.treebank import TreebankWordTokenizer






class Tokenization():

	def naive(self, text):


		tokenizedText = None

		# Define pattern to split sentences based on punctuation and whitespace characters
		pattern = r'[@_!#$%^&*()<>?/\\|}{~: |,. \s]'

		# Tokenize each document in the input text
		tokenizedText = [[token for token in re.split(pattern, doc) if token] for doc in text]

		return tokenizedText



	def pennTreeBank(self, text):

		tokenizedText = None

		# Initialize TreebankWordTokenizer
		t = TreebankWordTokenizer()

		# Tokenize each document in the input text using TreebankWordTokenizer
		tokenizedText = [t.tokenize(doc) for doc in text]

		# Return the tokenized text
		return tokenizedText