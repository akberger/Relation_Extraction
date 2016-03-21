import numpy as np
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from ExtractRelationFeatures import RelationFeatureExtractor

def build_dataset(rfe, vectorizer=None, label_map=None):
    """Given a RelationFeatureExtractor object, create numpy
    arrays for use as the dataset for SVM."""
    feat_dicts, y = make_feat_dicts_and_labels(rfe)
    if vectorizer:
        X = vectorizer.transform(feat_dicts)
    else:
        vectorizer = DictVectorizer(sparse=False)
        X = vectorizer.fit_transform(feat_dicts)
    if not label_map:
        label_map = make_label_map(y)
    y = encode_labels(label_map, y)
    return X, y, vectorizer, label_map

def make_feat_dicts_and_labels(rfe):
    """Transform RelationFeatureExtractor object to list of
    feature dicts and labels to be made into sklearn datasets."""
    feat_dicts = []
    y = []
    for feature_list in rfe.relations:
        label = feature_list[0]
        y.append(label)
        d = {}
        for feature in feature_list[1:]:
            if '=' in feature:
                f = feature.split('=')
                d[f[0]] = f[1]
            else:
                d[feature] = 1
        feat_dicts.append(d)
    return feat_dicts, y

def make_label_map(y):
    """Return dictionary mapping string labels to ints"""
    label_map = {}
    label_map['no_rel'] = 0
    i = 1
    for label in y:
        if label in label_map:
            continue
        else:
            label_map[label] = i
            i += 1
    return label_map

def encode_labels(label_map, y):
    """Transform a list of string labels to integers."""
    return np.array([label_map[l] for l in y], dtype='int8')

def train_model(X, y, decision_function_shape='ovr', kernel='linear'):
    clf = svm.SVC(decision_function_shape=decision_function_shape,
                  kernel=kernel, verbose=True)
    clf.fit(X, y)
    return clf

def write_predictions(clf, label_map, X, y, outfile):
    """Write classifier output to file."""
    predictions = clf.predict(X, y)
    inv_label_map = {v:k for k,v in label_map.items()}
    with open(outfile, 'w+') as outf:
        for pred in predictions:
            outf.write(inv_label_map[pred] + '\n')
