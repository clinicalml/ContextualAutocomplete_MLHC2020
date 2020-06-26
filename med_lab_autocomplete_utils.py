import pandas as pd
import json
import numpy as np


class MedAutocomplete:
    def __init__(self):
        with open('ontologies/medication_ontology.json', 'r') as h:
            med_json = json.load(h)
        self.med_freqs = med_json['freq']
        self.med_spell = med_json['spell']
    
    def get_spell_ranking(self):
        return list(np.argsort(self.med_freqs))

    def get_frequency_ranking(self):
        return list(range(len(self.med_freqs)))
    
    def get_ranking(self):
        return self.get_frequency_ranking()

class LabAutocomplete:
    def __init__(self):
        with open('ontologies/lab_ontology.json', 'r') as h:
            lab_json = json.load(h)
        self.lab_freqs = lab_json['freq']
        self.lab_spell = lab_json['spell']
    
    def get_spell_ranking(self):
        return list(np.argsort(self.lab_freqs))

    def get_frequency_ranking(self):
        return list(range(len(self.lab_freqs)))
    
    def get_ranking(self):
        return self.get_frequency_ranking()