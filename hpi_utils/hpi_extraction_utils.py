from string import punctuation as punct; punct += ' '
import pickle
from functools32 import lru_cache
import re
from umls_data import UMLS, lookup, trie

import os


BASE_FP = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))


@lru_cache(maxsize=1)
def allowed_umls_terms():
    with open(os.path.join(BASE_FP, 'models/hpi_autocomplete/allowable_umls_lookups.pkl'), 'rb') as h:
        return pickle.load(h)


@lru_cache(maxsize=1)
def allowed_umls_trie():
    with open(os.path.join(BASE_FP, 'models/hpi_autocomplete/allowed_umls_trie.pkl'), 'rb') as h:
        return pickle.load(h)


@lru_cache(maxsize=1)
def acronyms():
    with open(os.path.join(BASE_FP, 'models/hpi_autocomplete/taber_acronym.pkl'), 'rb') as h:
        return pickle.load(h)


fullstops = ['.', '-', ';']
expected_delim = [',' , '.', ' ', ';', '+', '\n', '(', ')', '/']
midstops = ['+', 'but', 'and', 'pt', '.', ';', 'except', 'reports', 'alert', 'complains', 'has', 'states', 'secondary', 'per', 'did', 'aox3']
negwords = ['no', 'not', 'denies', 'without', 'non']

def remove_sub_strings(match_list):
    res = []
    for (match, cuis, start, ending) in match_list:
        sub_string = False
        for (match_1, cuis_1, start_1, ending_1) in res:
            if start_1 <= start and ending_1 >= ending:
                sub_string = True
        if not sub_string:
            res += [(match, cuis, start, ending)]
    return res

def insert_acronym(clause):
    split_template = "([{}])".format(''.join(re.escape(i) for i in expected_delim))
    split_template = re.compile(split_template)
    final_words = [acronyms()[word] if word in acronyms() else word for word in re.split(split_template, clause)]
    return ''.join(final_words)

def negation_detection(words):
    flag = 0
    res = []
    for i, w in enumerate(words):
        neg_start_condition = (flag == 1)
        neg_stop_condition =  (w in fullstops + midstops + negwords) or (i > 0 and words[i-1][-1] in fullstops)
        neg_end_of_list = (i==(len(words)-1) )
        if neg_start_condition and neg_stop_condition:
            flag = 0
            res += [(start_index, i-1)]
        elif neg_start_condition and neg_end_of_list:
            flag = 0
            res += [(start_index, i)]
        if w in negwords:
            flag = 1
            start_index = i
    return res

def filter_concepts(match_results, whitelist=None):
    blacklisted = lambda res : len(set(UMLS()[res[1][0]][3]).intersection(blacklisted_concepts)) > 0
    whitelisted = lambda res : res[1][0] in whitelisted_umls_terms
    if not whitelist:
        return [res for res in match_results if whitelisted(res) or not blacklisted(res)]
    else:
        res = []
        for r in match_results:
            cui_match = [c for c in r[1] if c in whitelist]
            if len(cui_match) > 0:
                r_new = (r[0], cui_match, r[2], r[3])
                res.append(r_new)
        return res

def find_concepts(note_text, window_size=150, whitelist=None, default_trie=None):
    if default_trie is None:
        default_trie = trie()
    concept_delim = expected_delim + ['-']
    idx = 0
    res = []
    for i in range(len(note_text)):
        if note_text[i] in ' \r\t\n\f' and i > 0 and note_text[i-1] not in ' \r\t\n\f':
            idx += 1
        if i == 0 or note_text[i-1] in punct:
            matches = set(default_trie.prefixes(unicode((note_text[i:i + window_size]).lower())))
            matches = set([m for m in matches if ',' not in m]) # disallow commas, e.g. disease, coronary
            if len(matches) == 0:
                continue
            longest_match = max(matches, key=len)
            if len(longest_match) <= 3:
                continue
            if (i+len(longest_match)==len(note_text)) or (note_text[i+len(longest_match)] in concept_delim):
                res += [(longest_match, lookup()[longest_match], idx, idx + len(longest_match.split()) - 1)]
            else:
                continue
    res = filter_concepts(res, whitelist=whitelist)
    res_no_substrings = remove_sub_strings(res)
    return res_no_substrings

def filter_negatives_omr(extracted_concepts, detected_negations):
    pos = []
    neg = []
    for (term, cui, start_ix, end_ix) in extracted_concepts:
        cui_converts = [UMLS()[c][1] for c in cui]
        sole_cui = cui[cui_converts.index(term)] if term in cui_converts else cui[0]
        found_neg = False
        for neg_start_ix, neg_end_ix in detected_negations:
            if start_ix >= neg_start_ix and end_ix <= neg_end_ix:
                found_neg = True
                break
        if found_neg:
            neg.append((term, sole_cui))
        else:
            pos.append((term, sole_cui))
    return (pos, neg)

def parse_omr_note(omr_note):
    try:
        text = omr_note['text']
        if text is None:
            return ([], [])
        text = text.lower()
        text = insert_acronym(text)
        detected_negations = negation_detection(text.split())
        extracted_concepts = find_concepts(text, whitelist=allowed_umls_terms(), default_trie=allowed_umls_trie())
        pos, neg = filter_negatives_omr(extracted_concepts, detected_negations)
        return (pos, neg)
    except UnicodeDecodeError:
        return ([], [])
