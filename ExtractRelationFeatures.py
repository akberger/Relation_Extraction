"""
Extract features for each relation
Format data for MaxEnt classifer
"""
import sys

class RelationFeatureExtractor(object):
	"""
	Extract features for each relation in the corpus
	Input files are read with each line as a document representing
	a relation. 
	Output: A file where each line is a list of features for a relation
	"""

	def __init__(self, corpus, outfile, train=False):
		self.relations = list()
		self.train = train
		self.corpus = corpus
		self.outfile = outfile

		self.feat_fns = [self.words,
						 self.word_types
						 #self.between_words
						 ]

	def words(self, rel):
			"""extracts the words in the relation"""
			return ["w1={0}".format(rel[7]), "w2={0}".format(rel[13])]

	def word_types(self, rel):
		"""extracts the type of the words in the relation"""
		return ["t1={0}".format(rel[5]), "t2={0}".format(rel[11])]

	def postagged_files(self):
		pass

	def parsed_files(self):
		pass

	def featurize(self, parsed_files=None, postagged_files=None):
		"""
		Generate all features requested in self.feat_fns for each
		relation in the corpus
		"""

		if parsed_files:
			self.feat_fns.append(self.parsed_files)
		if postagged_files:
			self.feat_fns.append(self.postagged_files)

		relations = open(self.corpus)
		for i, line in enumerate(relations):
			line = line.split()
			rel_feats = list()
			if self.train:
				rel_feats.append(line[0])
			for f in self.feat_fns:
				# feature fcns return a list containing the feature(s) it extracts
				rel_feats += f(line)
			self.relations.append(rel_feats)

	def write_output(self):
		out = open(self.outfile, 'w')
		for r in self.relations:
			print r
			out.write(' '.join(r) + "\n")

def main(corpus, outfile, train):
	erf = RelationFeatureExtractor(corpus, outfile, train=train)
	erf.featurize()
	erf.write_output()


if __name__ == '__main__':
	corpus = sys.argv[1]
	outfile = sys.argv[2]
	train = False
	if len(sys.argv) > 2:
		train = True
	main(corpus, outfile, train)


