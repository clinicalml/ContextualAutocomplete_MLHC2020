import pickle
import numpy as np
import string
from textblob import Word
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn
import pandas as pd

import os

from hpi_extraction_utils import parse_omr_note
from umls_data import UMLS, lookup, trie

BASE_FP = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

class HPIAutocompleteRanker(nn.Module):
    def __init__(self, tf_idf_size, num_buckets):
        super(HPIAutocompleteRanker, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Linear(tf_idf_size, 64),
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Linear(num_buckets, 64),
            nn.ReLU(),
        )
        self.final = nn.Sequential(
            nn.Linear(128, num_buckets),
        )

    def forward(self, x, y):
        b1 = self.branch1(x)
        b2 = self.branch2(y)
        concat = torch.cat((b1, b2), dim=1)
        output = self.final(concat)
        return output

class HPIAutocomplete:
    """
    Predicts terms that will be mentioned in the HPI for contextual autocomplete.
    Use dual-branch shallow MLP with (1) binary tf-idf on triage text and (2) binary presence of OMR buckets.
    """
    def __init__(self):
        with open(os.path.join(BASE_FP, 'models/hpi_autocomplete/taber_acronym.pkl')) as h:
            self.acronyms = pickle.load(h)
        with open(os.path.join(BASE_FP, 'models/hpi_autocomplete/allowable_umls_lookups.pkl'), 'r') as h:
            self.allowable_umls_terms = pickle.load(h)
        self.stopwords = set(nltk_stopwords.words("english")).union(list(string.punctuation) + [' ', '\n'])
        self.triage_lemmatizer = WordNetLemmatizer()
        with open(os.path.join(BASE_FP, 'models/hpi_autocomplete/hpi_autocomplete_vectorizer.pkl'), 'rb') as h:
            self.triage_vectorizer = pickle.load(h)
        with open(os.path.join(BASE_FP, 'models/hpi_autocomplete/umls_to_history_bucket.pkl'), 'rb') as h:
            self.umls_to_hist_bucket = pickle.load(h)
            self.max_bucket = max(self.umls_to_hist_bucket.values()) + 1
        self.nn_ranker = HPIAutocompleteRanker(len(self.triage_vectorizer.get_feature_names()), self.max_bucket)
        self.nn_ranker.load_state_dict(torch.load(os.path.join(BASE_FP, 'models/hpi_autocomplete/hpi_autocomplete_dual_branch_nn.model')))
        self.nn_ranker.eval()
        self.autocomplete_ontology = pd.read_csv(os.path.join(BASE_FP, 'ontologies/hpi_autocomplete_ontology.csv'), index_col=0)
        self.autocomplete_ontology['uids_eval'] = self.autocomplete_ontology['uids'].apply(pd.eval)
        self.autocomplete_ontology['synonyms'] = self.autocomplete_ontology['synonyms'].apply(pd.eval)
        self.autocomplete_ontology_buckets = {}
        self.autocomplete_to_hist_buckets = {i : row['model_relevance_bucket'] for i, row in self.autocomplete_ontology.iterrows()}
        self.empirical_probabilities = {i : row['freq'] for i, row in self.autocomplete_ontology.iterrows()}
        for i, uids in enumerate(self.autocomplete_ontology['uids_eval']):
            for u in uids:
                self.autocomplete_ontology_buckets[int(u)] = i

    def spell_correct(self, note):
        final = []
        for word in note.split():
            if (len(word) < 5 or not word.islower() or not word.isalpha()) or word.lower() in lookup() or word.lower() in self.acronyms:
                final.append(word.lower())
            else:
                w = Word(word.lower())
                new_word, conf = w.spellcheck()[0]
                word_to_add = word if conf < 0.7 else new_word
                final.append(self.triage_lemmatizer.lemmatize(word_to_add))
        return ' '.join(final)

    def get_omr_buckets(self, omr_notes):
        omr_buckets = set()
        omr_terms = set()
        for note in omr_notes:
            pos, neg = parse_omr_note(note)
            if not pos:
                continue
            for term, cui in pos:
                if cui in self.umls_to_hist_bucket:
                    omr_buckets.add(self.umls_to_hist_bucket[cui])
                    if cui in self.autocomplete_ontology_buckets:
                        omr_terms.add(self.autocomplete_ontology_buckets[cui])
        return omr_buckets, omr_terms

    def get_bucket_ranking(self, corpus, omr_buckets):
        X1 = torch.FloatTensor(self.triage_vectorizer.transform([corpus]).toarray())
        X2 = np.zeros((1, self.max_bucket))
        X2[:, omr_buckets] = 1
        X2 = torch.FloatTensor(X2)
        outputs = torch.squeeze(nn.Sigmoid()(self.nn_ranker(X1, X2))).data.numpy()
        ixs_omr = np.array([i for i in np.argsort(outputs)[::-1] if i in omr_buckets])
        ixs_non_omr = np.array([i for i in np.argsort(outputs)[::-1] if i not in omr_buckets])
        return ixs_omr, ixs_non_omr

    def get_lime_data(self, corpus, omr_buckets):
        corpus = self.spell_correct(corpus)
        X1 = torch.FloatTensor(self.triage_vectorizer.transform([corpus]).toarray())
        X2 = np.zeros((1, self.max_bucket))
        X2[:, omr_buckets] = 1
        X2 = torch.FloatTensor(X2)
        outputs = torch.squeeze(nn.Sigmoid()(self.nn_ranker(X1, X2))).data.numpy()
        return X1.unsqueeze(dim=0).numpy(), X2.unsqueeze(dim=0).numpy(), outputs
        
    def get_spell_ranking(self):
        return list(np.argsort(list(self.autocomplete_ontology['common_name'])))

    def get_frequency_ranking(self):
        return list(range(len(self.autocomplete_ontology))) 

    def get_ranking(self, corpus, omr_buckets, omr_terms, last_mentioned):
        if not corpus:
            omr_ranking, non_omr_ranking = [], []
            omr_probs, non_omr_probs = [], []
            term_ranking = sorted(self.empirical_probabilities, key=self.empirical_probabilities.get, reverse=True)
            for t in term_ranking:
                if self.autocomplete_to_hist_buckets.get(t) in omr_buckets:
                    omr_ranking.append(t)
                    omr_probs.append([0, self.empirical_probabilities.get(t, 0)])
                else:
                    non_omr_ranking.append(t)
                    non_omr_probs.append([0, self.empirical_probabilities.get(t, 0)])
            return omr_ranking + non_omr_ranking
        corpus = self.spell_correct(corpus)
        omr_bucket_ranking, non_omr_bucket_ranking = self.get_bucket_ranking(corpus, omr_buckets)
        autocomplete_terms = set([t for t in self.autocomplete_to_hist_buckets if t in omr_terms and not self.autocomplete_ontology.loc[t]['ignore']])
        non_omr_terms = [
            autocomplete_b for autocomplete_b in self.empirical_probabilities
            if autocomplete_b not in autocomplete_terms
            and not self.autocomplete_ontology.loc[autocomplete_b]['ignore']
        ]
        bucket_reverse_map = {j : i for i,j in enumerate(list(omr_bucket_ranking) + list(non_omr_bucket_ranking))}
        def get_rank(term):
            return (bucket_reverse_map.get(self.autocomplete_to_hist_buckets.get(term), len(self.empirical_probabilities)),
                    -self.empirical_probabilities.get(term, 0))
        omr_term_ranking = sorted(autocomplete_terms, key=get_rank)
        non_omr_term_ranking = sorted(non_omr_terms, key=get_rank)
        return omr_term_ranking + non_omr_term_ranking
