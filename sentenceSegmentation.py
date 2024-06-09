
import nltk
import re
from nltk.tokenize import sent_tokenize
class SentenceSegmentation():

	def naive(self, text):

		segmentedText = None

		# defining regular expression as pattern
		pattern = r'(?<=[.!?])+'

		# Split text into sentences
		segmentedText = re.split(pattern, text)


		return segmentedText





	def punkt(self, text):


		segmentedText = None

		# Remove periods from the input text


		# Load the Punkt tokenizer for English
		sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

		# Tokenize the text into sentences using the Punkt tokenizer
		segmentedText = sent_detector.tokenize(text.strip())



		
		return segmentedText