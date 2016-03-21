"""
Extract features for each relation
Format data for MaxEnt classifer
"""
import sys
import os
from collections import defaultdict
from nltk.corpus import gazetteers, names

class RelationFeatureExtractor(object):
    """
    Extract features for each relation in the corpus
    Input files are read with each line as a document representing
    a relation. 
    Output: A file where each line is a list of features for a relation
    """

    def __init__(self, corpus, outfile, tokens_dir, train=False):
        self.relations = list()
        self.train = train
        self.corpus = corpus
        self.outfile = outfile
        self.tokenized_sents, self.tok_sents_pos = self.process_tokens_dir(tokens_dir)
        
        self.pronouns = ["I", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
                        "he", "him", "his", "his", "himself", "she", "her", "hers", "herself", 
                        "it", "its", "itself", "we", "us", "our", "ours", "ourselves", "you", "your", 
                        "yours", "yourselves", "they", "them", "their", "theirs", "themselves"]

        self.locations = set([c.lower() for c in gazetteers.words('countries.txt')] + 
                             [s.lower() for s in gazetteers.words('usstates.txt')])
        self.names = set([name.lower() for name in names.words('male.txt')] +
                 [name.lower() for name in names.words('female.txt')])

        self.feat_fns = [self.words,    #good
                         self.word_types, #good
                         self.pronoun, #good
                         self.name, #good
                         #self.place, #look to get a better list
                         self.num_words_between, #good
                         self.words_between_words, #good
                         self.prev_word, #good
                         #self.post_word, #really bad feature
                         #self.prev_word_pos, #bad
                         self.post_word_pos, #good
                         self.first_word_after_w1, #good
                         self.words_between_POSs, #good 
                         ]
    
    def process_tokens_dir(self, tokens_dir):
        """
        read in all tokenized files. Arrange in dictionary. 
        Key: is fileID
        Value: list of lists. Each nested list is a tokenized sentence
        """
        d_words = defaultdict(list)
        d_pos = defaultdict(list)

        for filename in os.listdir(tokens_dir):
            f = open(os.path.join(tokens_dir, filename))
            fID = ""
            for i, line in enumerate(f):
                line = line.split()
                words = [word.split('_')[0] for word in line]
                POSs = [word.split('_')[1] for word in line]
                if line:
                    if i == 0:
                        fID = words[0]  #first line contains the file ID
                        d_words[fID].append([]) #works with the indexing in .tag files
                        d_pos[fID].append([])
                    else:
                        d_words[fID].append(words)
                        d_pos[fID].append(POSs)

        return d_words, d_pos

    def get_indices(self, rel):
        """relation ID, relation sentence index, word1 index, word2 index"""
        return rel[-13], int(rel[-12]), int(rel[-11]), int(rel[-5])

    def words(self, rel):
        """
        extracts the words in the relation

        I used reverse indexing because that always matches.
        .gold files have relation type as first index, but .raw don't
        """
        return ["w1={0}".format(rel[-7]), "w2={0}".format(rel[-1])]
        #return ["wds={0}-{1}".format(rel[-7], rel[-1])]

    def word_types(self, rel):
        """extracts the type of the words in the relation"""
        return ["t1={0}".format(rel[-9]), "t2={0}".format(rel[-3])]
        #return ["wdtyps={0}-{1}".format(rel[-9], rel[-3])]

    def pronoun(self, rel):
        p = []
        if rel[-7] in self.pronouns:
            p.append("w1=PRN")
        if rel[-7] in self.pronouns:
            p.append("w2=PRN")
        return p

    def name(self, rel):
        n = []
        w1 = rel[-7].split('_')
        w2 = rel[-1].split('_')
        for i, w in enumerate([w1, w2]):
            if w[0] in self.names:
                n.append("{0}=NAME".format(str(i)))
        return n

    def place(self, rel):
        p = []
        w1 = ' '.join(rel[-7].split('_'))
        w2 = ' '.join(rel[-1].split('_'))
        for i, w in enumerate([w1, w2]):
            if w in self.locations:
                p.append('{0}=PLACE'.format(str(i)))
        return p

    def num_words_between(self, rel):
        """find the amount of words between the entities in the relation"""
        dist = abs(int(rel[-4]) - int(rel[-10]))
        return ["dist={0}".format(str(dist))]

    def words_between_words(self, rel):
        relID, rel_index, w1_index, w2_index = self.get_indices(rel)
        btwn_wds = self.tokenized_sents[relID][rel_index][w1_index + 1 : w2_index]
        return ["bwtnwds={0}".format(''.join(btwn_wds))]

    def words_between_POSs(self, rel):
        relID, rel_index, w1_index, w2_index = self.get_indices(rel)
        btwn_pos = self.tok_sents_pos[relID][rel_index][w1_index + 1 : w2_index]
        return ["btwnpos={0}".format(''.join(btwn_pos))]

    def prev_word(self, rel):
        relID, rel_index, w1_index, w2_index = self.get_indices(rel)
        w1prev = ""
        if w1_index is not 0:
            w1prev = self.tokenized_sents[relID][rel_index][w1_index-1]

        return ["w1prv={0}".format(w1prev)]

    def post_word(self, rel):
        relID, rel_index, w1_index, w2_index = self.get_indices(rel)
        w2post = ""
        if w2_index is not len(self.tokenized_sents[relID][rel_index]):
            w2post = self.tokenized_sents[relID][rel_index][w2_index+1]

        return ["w2pst={0}".format(w2post)]

    def prev_word_pos(self,rel):
        relID, rel_index, w1_index, w2_index = self.get_indices(rel)
        prvpos = ""
        if w1_index is not 0:
            prvpos = self.tok_sents_pos[relID][rel_index][w1_index-1]

        return ["prvpos={0}".format(prvpos)]

    def post_word_pos(self, rel):
        relID, rel_index, w1_index, w2_index = self.get_indices(rel)
        pstpos = ""
        if w1_index is not len(self.tokenized_sents[relID][rel_index]):
            pst = self.tok_sents_pos[relID][rel_index][w1_index+1]

        return ["pstpos={0}".format(pstpos)]

    def first_word_after_w1(self, rel):
        relID, rel_index, w1_index, w2_index = self.get_indices(rel)
        word = self.tokenized_sents[relID][rel_index][w1_index+1]
        return ["w1pst={0}".format(word)]

    def mentions_between(self, rel, sent_mentions):
        relID, rel_index, w1_index, w2_index = self.get_indices(rel)
        mtns_bwtn = abs(sent_mentions.index(w2_index) - sent_mentions.index(w1_index))
        return ["mtnsbtwn={0}".format(str(mtns_bwtn))]

    def featurize(self, parsed_files=None, postagged_files=None):
        """
        Generate all features requested in self.feat_fns for each
        relation in the corpus
        """

        if parsed_files:
            self.feat_fns.append(self.parsed_files)
        if postagged_files:
            self.feat_fns.append(self.postagged_files)

        rels = open(self.corpus)
        sent_num = 0
        sent_rels = []
        sent_mentions = []

        for i, line in enumerate(rels):
            line = line.split()
            sent = line[-12]
            if sent is not sent_num: #we are on a new sentence, get feats from prev
                sent_num = sent
                if sent_num:
                    #get features from prev sentence
                    sent_mentions = sorted(set(sent_mentions))
                    self.get_rel_features(sent_rels, sent_mentions)
                    #clear for next sentence
                    sent_rels = []
                    sent_mentions = []
                    #add first line in new sentence
                    sent_rels.append(line)
                    sent_mentions.append(int(line[-11]))
                    sent_mentions.append(int(line[-5]))
            else:
                sent_rels.append(line)
                sent_mentions.append(int(line[-11]))
                sent_mentions.append(int(line[-5]))
        self.get_rel_features(sent_rels, sent_mentions)

    def get_rel_features(self, sent_rels, sent_mentions):
        """
        get the feature list for each relation in sent_rels
        add feature list to set of corpus relations
        """
        for rel in sent_rels:
            rel_feats = []
            if self.train:
                rel_feats.append(rel[0])
            for f in self.feat_fns:
                rel_feats += f(rel)
            rel_feats += self.mentions_between(rel, sent_mentions)
            #print rel_feats
            self.relations.append(rel_feats)


    def write_output(self):
        out = open(self.outfile, 'w')
        for r in self.relations:
            out.write(' '.join(r) + "\n")

if __name__ == '__main__':
    corpus = sys.argv[1]
    outfile = sys.argv[2]
    tokens_dir = sys.argv[3]
    train = False
    if len(sys.argv) > 3:
        train = True

    erf = RelationFeatureExtractor(corpus, outfile, tokens_dir, train)
    erf.featurize()
    erf.write_output()


