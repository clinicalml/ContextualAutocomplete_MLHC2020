import pickle
import numpy as np
import pandas as pd
import os
from scipy.stats import percentileofscore
import vital_parser_utils
import os

MODEL_FP = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))
BASE_FP = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

class SymptomAutocomplete:
    """
    Predicts symptoms that will be mentioned in a clinical note based on a chief complaint
    and list of vitals.
    """
    def __init__(self):
        self.symptom_ontology = pd.read_csv(os.path.join(BASE_FP, 'ontologies/symptom_autocomplete_ontology.csv'), index_col=0)
        self.vital_names = ['age', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'resp_rate', 'pulse_ox', 'temp']
        with open(os.path.join(MODEL_FP, 'hpi_symptoms/population_vitals.pkl')) as h:
            self.population_vitals = pickle.load(h)
        with open(os.path.join(MODEL_FP, 'hpi_symptoms/symptom_empirical_info.pkl')) as h:
            self.chief_complaints, self.vital_buckets, self.symptom_umls_terms, self.cc_symptom_freqs, self.cc_vital_freqs = pickle.load(h)

    def parse_vital_values(self, vitals):
        age = vital_parser_utils.parse_age(vitals.get('Age'))
        temp = vital_parser_utils.parse_temp(vitals.get('Temp'))
        resp_rate = vital_parser_utils.parse_resp_rate(vitals.get('RR'))
        pulse_ox = vital_parser_utils.parse_pulse_ox(vitals.get('O2Sat'))
        heart_rate = vital_parser_utils.parse_heart_rate(vitals.get('Pulse'))
        systolic_bp, diastolic_bp = vital_parser_utils.parse_blood_pressure(vitals.get('BP'))
        vitals_parsed = {
            'age' : age,
            'systolic_bp' : systolic_bp,
            'diastolic_bp' : diastolic_bp,
            'heart_rate' : heart_rate,
            'resp_rate' : resp_rate,
            'pulse_ox' : pulse_ox,
            'temp' : temp
        }
        return vitals_parsed

    def get_vital_buckets(self, vitals):
        age_bucket = vital_parser_utils.bucketize_age(vitals.get('age'))
        temp_bucket = vital_parser_utils.bucketize_temp(vitals.get('temp'))
        resp_rate_bucket = vital_parser_utils.bucketize_resp_rate(vitals.get('resp_rate'))
        pulse_ox_bucket = vital_parser_utils.bucketize_pulse_ox(vitals.get('pulse_ox'))
        heart_rate_bucket = vital_parser_utils.bucketize_heart_rate(vitals.get('heart_rate'))
        blood_pressure_bucket = vital_parser_utils.bucketize_blood_pressure(vitals.get('systolic_bp'), vitals.get('diastolic_bp'))
        vital_buckets = {
            'age' : age_bucket,
            'systolic_bp' : blood_pressure_bucket,
            'diastolic_bp' : blood_pressure_bucket,
            'heart_rate' : heart_rate_bucket,
            'resp_rate' : resp_rate_bucket,
            'pulse_ox' : pulse_ox_bucket,
            'temp' : temp_bucket
        }
        return vital_buckets
    
    def get_spell_ranking(self):
        return [t for t in np.argsort(list(self.symptom_ontology['common_name'])) if not self.symptom_ontology.iloc[t]['ignore']]
    
    def get_frequency_ranking(self):
        return [t for t in range(len(self.symptom_umls_terms)) if not self.symptom_ontology.iloc[t]['ignore']]

    def get_ranking(self, chief_complaint, vitals):
        if chief_complaint not in self.chief_complaints:
            return [t for t in range(len(self.symptom_umls_terms)) if not self.symptom_ontology.iloc[t]['ignore']]
        cc_ix = self.chief_complaints.index(chief_complaint)
        if not vitals:
            ranking = np.argsort(self.cc_symptom_freqs[:, cc_ix])[::-1]
            return [t for t in ranking if not self.symptom_ontology.iloc[t]['ignore']]
        vital_values = self.parse_vital_values(vitals)
        vital_buckets = self.get_vital_buckets(vital_values)
        if None in vital_values.values():
            ranking = np.argsort(self.cc_symptom_freqs[:, cc_ix])[::-1]
            return [t for t in ranking if not self.symptom_ontology.iloc[t]['ignore']]
        vitals_percentile = np.array([percentileofscore(self.population_vitals[v], vital_values[v]) for v in self.vital_names])
        abnormal_vital_bucket = vital_buckets[self.vital_names[np.argmax(np.abs(vitals_percentile - 50))]]
        vital_ix = self.vital_buckets.index(abnormal_vital_bucket)
        cc_vital_ix = (len(self.vital_buckets) * cc_ix) + vital_ix
        if self.cc_vital_freqs[cc_vital_ix, :].sum() < 100:
            ranking = np.argsort(self.cc_symptom_freqs[:, cc_ix])[::-1]
        else:
            ranking = np.argsort(self.cc_vital_freqs[cc_vital_ix, :])[::-1]
        return [t for t in ranking if not self.symptom_ontology.iloc[t]['ignore']]
