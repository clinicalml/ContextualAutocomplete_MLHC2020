import pickle
import numpy as np
import pandas as pd
import os
from vital_parser_utils import *
import scipy.sparse
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

import os

MODEL_FP = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))
BASE_FP = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

class SymptomAutocompleteChiefComplaint:
    """
    Predicts symptoms that will be mentioned in a clinical note based on a chief complaint
    and list of vitals.
    """
    def __init__(self):
        self.symptom_ontology = pd.read_csv(os.path.join(BASE_FP, 'ontologies/symptom_autocomplete_ontology.csv'), index_col=0)
        with open(os.path.join(MODEL_FP, 'hpi_symptoms/population_vitals.pkl')) as h:
            self.population_vitals = pickle.load(h)
        with open(os.path.join(MODEL_FP, 'hpi_symptoms/symptom_empirical_info.pkl')) as h:
            self.chief_complaints, self.vital_buckets, self.symptom_umls_terms, self.cc_symptom_freqs, self.cc_vital_freqs = pickle.load(h)
    
    def get_spell_ranking(self):
        return [t for t in np.argsort(list(self.symptom_ontology['common_name'])) if not self.symptom_ontology.iloc[t]['ignore']]
    
    def get_frequency_ranking(self):
        return [t for t in range(len(self.symptom_umls_terms)) if not self.symptom_ontology.iloc[t]['ignore']]

    def get_ranking(self, chief_complaint, vitals):
        if chief_complaint not in self.chief_complaints:
            return [t for t in range(len(self.symptom_umls_terms)) if not self.symptom_ontology.iloc[t]['ignore']]
        
        cc_ix = self.chief_complaints.index(chief_complaint)
        ranking = np.argsort(self.cc_symptom_freqs[:, cc_ix])[::-1]
        return [t for t in ranking if not self.symptom_ontology.iloc[t]['ignore']]

class SymptomAutocompleteLR:
    """
    Predicts symptoms that will be mentioned in a clinical note based on a chief complaint
    and list of vitals.
    """
    def __init__(self):
        self.symptom_ontology = pd.read_csv(os.path.join(BASE_FP, 'ontologies/symptom_autocomplete_ontology.csv'), index_col=0)
        with open(os.path.join(MODEL_FP, 'hpi_symptoms/logistic_regression_cc_vitals_MLHC.pkl'), 'rb') as h:
            self.cc_encoder, self.vitals_encoder, self.models = pickle.load(h)
        self.cc_possibilities = set(self.cc_encoder.classes_)
    
    def parse_vital_values(self, vitals):
        age = parse_age(vitals.get('Age'))
        temp = parse_temp(vitals.get('Temp'))
        resp_rate = parse_resp_rate(vitals.get('RR'))
        pulse_ox = parse_pulse_ox(vitals.get('O2Sat'))
        heart_rate = parse_heart_rate(vitals.get('Pulse'))
        systolic_bp, diastolic_bp = parse_blood_pressure(vitals.get('BP'))
        vitals_parsed = {
            'age' : age,
            'systolic_bp' : systolic_bp,
            'diastolic_bp' : diastolic_bp,
            'heart_rate' : heart_rate,
            'resp_rate' : resp_rate,
            'pulse_ox' : pulse_ox,
            'temp' : temp,
            'sex' : vitals.get('Sex'),
            'acuity' : vitals.get('Acuity')
        }
        return vitals_parsed

    def get_vital_buckets(self, vitals):
        age_bucket = bucketize_age(vitals.get('age'))
        temp_bucket = bucketize_temp(vitals.get('temp'))
        resp_rate_bucket = bucketize_resp_rate(vitals.get('resp_rate'))
        pulse_ox_bucket = bucketize_pulse_ox(vitals.get('pulse_ox'))
        heart_rate_bucket = bucketize_heart_rate(vitals.get('heart_rate'))
        blood_pressure_bucket = bucketize_blood_pressure(vitals.get('systolic_bp'), vitals.get('diastolic_bp'))
        vital_buckets = {
            'age' : age_bucket,
            'systolic_bp' : blood_pressure_bucket,
            'diastolic_bp' : blood_pressure_bucket,
            'heart_rate' : heart_rate_bucket,
            'resp_rate' : resp_rate_bucket,
            'pulse_ox' : pulse_ox_bucket,
            'temp' : temp_bucket,
            'sex' : vitals.get('sex'),
            'acuity' : vitals.get('acuity')
        }
        return vital_buckets
        
    def get_spell_ranking(self):
        return [t for t in np.argsort(list(self.symptom_ontology['common_name'])) if not self.symptom_ontology.iloc[t]['ignore']]
    
    def get_frequency_ranking(self):
        return [t for t in range(len(self.symptom_ontology)) if not self.symptom_ontology.iloc[t]['ignore']]

    def get_ranking(self, chief_complaint, vitals):
        if chief_complaint not in self.cc_possibilities or not vitals:
            return [t for t in range(len(self.symptom_ontology)) if not self.symptom_ontology.iloc[t]['ignore']]
        ccs_encoded = self.cc_encoder.transform([[chief_complaint]])
        vital_values = self.parse_vital_values(vitals)
        vital_buckets = self.get_vital_buckets(vital_values)
        if None in vital_values.values():
            return [t for t in range(len(self.symptom_ontology)) if not self.symptom_ontology.iloc[t]['ignore']]
        vitals_to_transform = [[vital_buckets['age'], vital_buckets['sex'], vital_buckets['acuity'], vital_buckets['systolic_bp'], vital_buckets['heart_rate'], vital_buckets['resp_rate'], vital_buckets['pulse_ox'], vital_buckets['temp']]]
        vitals_encoded = self.vitals_encoder.transform(vitals_to_transform)
        all_features_encoded = scipy.sparse.hstack((ccs_encoded, vitals_encoded)).tocsr()
        output_probs = np.array([model[0].predict_proba(all_features_encoded)[0, 1] for model in self.models])
        return list(np.argsort(output_probs)[::-1])


class SymptomAutocompleteNB:
    """
    Predicts symptoms that will be mentioned in a clinical note based on a chief complaint
    and list of vitals.
    """
    def __init__(self):
        self.symptom_ontology = pd.read_csv(os.path.join(BASE_FP, 'ontologies/symptom_autocomplete_ontology.csv'), index_col=0)
        with open(os.path.join(MODEL_FP, 'hpi_symptoms/naive_bayes_cc_vitals_MLHC.pkl'), 'rb') as h:
            self.cc_encoder, self.vitals_encoder, self.models = pickle.load(h)
        self.cc_possibilities = set(self.cc_encoder.classes_)
    
    def parse_vital_values(self, vitals):
        age = parse_age(vitals.get('Age'))
        temp = parse_temp(vitals.get('Temp'))
        resp_rate = parse_resp_rate(vitals.get('RR'))
        pulse_ox = parse_pulse_ox(vitals.get('O2Sat'))
        heart_rate = parse_heart_rate(vitals.get('Pulse'))
        systolic_bp, diastolic_bp = parse_blood_pressure(vitals.get('BP'))
        vitals_parsed = {
            'age' : age,
            'systolic_bp' : systolic_bp,
            'diastolic_bp' : diastolic_bp,
            'heart_rate' : heart_rate,
            'resp_rate' : resp_rate,
            'pulse_ox' : pulse_ox,
            'temp' : temp,
            'sex' : vitals.get('Sex'),
            'acuity' : vitals.get('Acuity')
        }
        return vitals_parsed

    def get_vital_buckets(self, vitals):
        age_bucket = bucketize_age(vitals.get('age'))
        temp_bucket = bucketize_temp(vitals.get('temp'))
        resp_rate_bucket = bucketize_resp_rate(vitals.get('resp_rate'))
        pulse_ox_bucket = bucketize_pulse_ox(vitals.get('pulse_ox'))
        heart_rate_bucket = bucketize_heart_rate(vitals.get('heart_rate'))
        blood_pressure_bucket = bucketize_blood_pressure(vitals.get('systolic_bp'), vitals.get('diastolic_bp'))
        vital_buckets = {
            'age' : age_bucket,
            'systolic_bp' : blood_pressure_bucket,
            'diastolic_bp' : blood_pressure_bucket,
            'heart_rate' : heart_rate_bucket,
            'resp_rate' : resp_rate_bucket,
            'pulse_ox' : pulse_ox_bucket,
            'temp' : temp_bucket,
            'sex' : vitals.get('sex'),
            'acuity' : vitals.get('acuity')
        }
        return vital_buckets
        
    def get_spell_ranking(self):
        return [t for t in np.argsort(list(self.symptom_ontology['common_name'])) if not self.symptom_ontology.iloc[t]['ignore']]
    
    def get_frequency_ranking(self):
        return [t for t in range(len(self.symptom_ontology)) if not self.symptom_ontology.iloc[t]['ignore']]

    def get_ranking(self, chief_complaint, vitals):
        if chief_complaint not in self.cc_possibilities or not vitals:
            return [t for t in range(len(self.symptom_ontology)) if not self.symptom_ontology.iloc[t]['ignore']]
        ccs_encoded = self.cc_encoder.transform([[chief_complaint]])
        vital_values = self.parse_vital_values(vitals)
        vital_buckets = self.get_vital_buckets(vital_values)
        if None in vital_values.values():
            return [t for t in range(len(self.symptom_ontology)) if not self.symptom_ontology.iloc[t]['ignore']]
        vitals_to_transform = [[vital_buckets['age'], vital_buckets['sex'], vital_buckets['acuity'], vital_buckets['systolic_bp'], vital_buckets['heart_rate'], vital_buckets['resp_rate'], vital_buckets['pulse_ox'], vital_buckets['temp']]]
        vitals_encoded = self.vitals_encoder.transform(vitals_to_transform)
        all_features_encoded = scipy.sparse.hstack((ccs_encoded, vitals_encoded)).tocsr()
        output_probs = np.array([model[0].predict_proba(all_features_encoded)[0, 1] for model in self.models])
        return list(np.argsort(output_probs)[::-1])
        
        
        
