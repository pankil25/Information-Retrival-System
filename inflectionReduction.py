

# Add your import statements here
import nltk
from nltk.stem import PorterStemmer




class InflectionReduction:

	def reduce(self, text):
		reducedText = []

		# Initialize Porter stemmer
		porter = PorterStemmer()

		# Perform stemming  for each word
		for docs in text:
			for word in docs:

				# Apply Porter stemming to the word
				stemmed_word = porter.stem(word)
				# Append the stemmed word to the reduced text list
				reducedText.append(stemmed_word)

		
		return reducedText


