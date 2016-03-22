"""
Extract features for each relation
Format data for MaxEnt classifer
"""
import sys
import os
from itertools import groupby
from collections import defaultdict
from nltk.corpus import gazetteers, names
from nltk.tree import ParentedTree

class RelationFeatureExtractor(object):
    """
    Extract features for each relation in the corpus
    Input files are read with each line as a document representing
    a relation. 
    Output: A file where each line is a list of features for a relation
    """

    def __init__(self, corpus, outfile, tokens_dir, parses_dir, train=False):
        self.relations = list()
        self.train = train
        self.corpus = corpus
        self.outfile = outfile
        self.tokenized_sents, self.tok_sents_pos = self.process_tokens_dir(tokens_dir)
        self.parses = self.process_parses_dir(parses_dir)
        self.clusterdict = self.make_cluster_dict('50mpaths2')
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
                         #self.last_word_before_w2
                         self.w1clust, #good
                         self.w2clust, #good
                         self.tree_path,
                         ]

    def make_cluster_dict(self, cfile):
        clusterdict = {}
        with open(cfile) as clusters:
            for line in clusters:
                clusterdict[line.split()[1]] = line.split()[0]
        return clusterdict

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

    def process_parses_dir(self, parses_dir):
        """ Create dictionary mapping file ids to list of parse trees."""
        d_parses = defaultdict(list)
        for filename in os.listdir(parses_dir):
            fID = '.'.join(os.path.basename(filename).split('.')[:-5])
            pfile = open(os.path.join(parses_dir, filename))
            for parse in pfile:
                parse = parse.strip()
                if parse:
                    d_parses[fID].append(parse)
        return d_parses

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

    def last_word_before_w2(self, rel):
        relID, rel_index, w1_index, w2_index = self.get_indices(rel)
        word = self.tokenized_sents[relID][rel_index][w2_index-1]
        return ["w2prv={0}".format(word)]

    def mentions_between(self, rel, sent_mentions):
        relID, rel_index, w1_index, w2_index = self.get_indices(rel)
        mtns_bwtn = abs(sent_mentions.index(w2_index) - sent_mentions.index(w1_index))
        return ["mtnsbtwn={0}".format(str(mtns_bwtn))]

    def w1clust(self, rel):
        word = rel[-7].lower().split('_')[-1]
        if word in self.clusterdict:
            return ['w1clust={0}'.format(self.clusterdict[word])]
        else:
            return ['w1clust=NA']

    def w2clust(self, rel):
        word = rel[-1].lower().split('_')[-1]
        if word in self.clusterdict:
            return ['w2clust={0}'.format(self.clusterdict[word])]
        else:
            return ['w2clust=NA']

    def tree_path(self, rel):
        """Return the string of labels traversed to get from entity 1 to
        entity 2 in the parse tree. Ignores consecutive duplicates."""
        relID, rel_index, w1_index, w2_index = self.get_indices(rel)
        word1 = rel[-7]
        word2 = rel[-1]
        word1 = word1.split('_')[-1]
        word2 = word2.split('_')[-1]
        tree = self.parses[relID][rel_index]
        ptree = ParentedTree.fromstring(tree)
        if word1.startswith('('):
            word1 = word1[1:]
        if word2.startswith('('):
            word2 = word2[1:]
        if word1.startswith('``'):
            word1 = word1[2:]
        if word2.startswith('``'):
            word2 = word2[2:]
        if word1 == "let's":
            word1 = "'s"
        if word2 == "let's":
            word2 = "'s"
        if word1.endswith("'s"):
            word1 = word1[:-2]
        if word2.endswith("'s"):
            word2 = word2[:-2]
        if len(word1) > 1:
            if word1[1] == "'":
                word1 = word1[2:]
        if len(word2) > 1:
            if word2[1] == "'":
                word2 = word2[2:]
        if "&AMP;" in word1:
            word1 = word1[:word1.index(';')]
        if "&AMP;" in word2:
            word2 = word2[:word2.index(';')]
        if word1.endswith('!'):
            word1 = word1[:-1]
        if word2.endswith('!'):
            word2 = word2[:-1]
        path = [label[0] for label in groupby(find_path(ptree, word1, word2))]
        return ['path={0}'.format(''.join(path))]

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

#http://stackoverflow.com/a/28750205/5818736
def get_lca_length(location1, location2):
    i = 0
    while i < len(location1) and i < len(location2) and location1[i] == location2[i]:
        i+=1
    return i

def get_labels_from_lca(ptree, lca_len, location):
    labels = []
    for i in range(lca_len, len(location)):
        labels.append(ptree[location[:i]].label())
    return labels

def find_path(ptree, text1, text2):
    leaf_values = ptree.leaves()
    leaf_index1 = leaf_values.index(text1)
    leaf_index2 = leaf_values.index(text2)
    location1 = ptree.leaf_treeposition(leaf_index1)
    location2 = ptree.leaf_treeposition(leaf_index2)
    #find length of least common ancestor (lca)
    lca_len = get_lca_length(location1, location2)
    #find path from the node1 to lca
    labels1 = get_labels_from_lca(ptree, lca_len, location1)
    #ignore the first element, because it will be counted in the second part of the path
    result = labels1[1:]
    #inverse, because we want to go from the node to least common ancestor
    result = result[::-1]
    #add path from lca to node2
    result = result + get_labels_from_lca(ptree, lca_len, location2)
    return result


    def write_output(self):
        out = open(self.outfile, 'w')
        for r in self.relations:
            out.write(' '.join(r) + "\n")


if __name__ == '__main__':
    corpus = sys.argv[1]
    outfile = sys.argv[2]
    tokens_dir = sys.argv[3]
    parses_dir = sys.argv[4]
    dparses_dir = sys.argv[5]
    train = False
    if len(sys.argv) > 3:
        train = True

    erf = RelationFeatureExtractor(corpus, outfile, tokens_dir, parses_dir, train)
    erf.featurize()
    erf.write_output()


