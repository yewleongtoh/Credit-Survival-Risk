#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class PipelineFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Class for performing column selection within an sklearn pipeline
    """
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        # do nothing
        return self

    def transform(self, X, y=None, **fit_params):
        def map_season(x):
            # Create new features Month, Year, Season
            labels = {"Spring": [3, 4, 5], 
              "Summer": [6, 7, 8],
              "Autumn": [9, 10, 11],
              "Winter": [12, 1, 2]}
            """
            Map Months into their respective season

            returns: label (str)
            """
            for k,v in labels.items():
                if x in v:
                    return k
            
        def map_states(x):
            # Create new features Month, Year, Season
            labels = {5: ["AK", "HI"], 
                      4: ["CO"],
                      3: ["MS", "RI", "LA", "VA"],
                      2: ["OK"],
                     1:["AL", "KY"]}
            """
            Rank states by default rate

            returns: label (str)
            """
            for k,v in labels.items():
                if x in v:
                    return k
                else:
                    return 0

        def map_pay_frequency(x):
            # Mapping for payFrequency Category column
            labels = {'B': 14,  # Biweekly = 14 days
                      'I': 1,   # Irregular = assume 1 days
                      'M': 30,  # Monthly = 30 days or 31 days
                      'S': 15,  # Semi Monthly = 15 days
                      'W': 7}   # Weekly = 7 days
            """
            Check if payFrequency present in the labels dict
            params:
            x = payFrequency, str

            returns: label (int)
            """
            for k,v in labels.items():
                if x in k:
                    return v
                else:
                    assert "Couldn't find right mapping for " + x

        def map_month(x):
            """
            Map Months into ordinal feature

            returns: weightage of month (float)
            """
            if x==2 | x==3:
                return x
            else:
                return 1

        def map_year(x):
            """
            Map Year into ordinal feature

            returns: weightage of year (float)
            """
            if x < 2017:
                return 2016 - x + 1
            else:
                return 4

        def create_isSeason(x):
            """
            Create Boolean feature

            returns: 0/1 (float)
            """
            if x == 'Spring':
                return 1
            else:
                return 0
        X_copy = X.copy()
        # Seasonal Feature
        X_copy['lastPaymentMonth'] = X_copy['lastPaymentDate'].dt.month
        X_copy['lastPaymentYear'] = X_copy['lastPaymentDate'].dt.year
        X_copy['lastPaymentSeason'] = X_copy['lastPaymentMonth'].map(map_season)
        X_copy['lastPaymentMonth'] = X_copy['lastPaymentMonth'].map(map_month)
        X_copy['lastPaymentYear'] = X_copy['lastPaymentYear'].map(map_year)
        X_copy['lastPaymentSeason'] = X_copy['lastPaymentSeason'].map(create_isSeason)
        # Geographical Feature
        X_copy['state'] = X_copy['state'].map(map_states)
        # Payment Behaviour
        X_copy['totalPayFrequency'] = X_copy['originallyScheduledPaymentAmount'] / X_copy['avgInstallmentAmountPerPayFrequency']
        X_copy.loc[~np.isfinite(X_copy['totalPayFrequency']), 'totalPayFrequency'] = 0 
        X_copy['payFrequency'] = X_copy['payFrequency'].apply(map_pay_frequency)
        X_copy['expectedDaysUntilPayoff'] = X_copy['payFrequency'] * X_copy['totalPayFrequency']
        # Lead Type
        X_copy['leadType'] = X_copy['leadType'].apply(lambda x: 1 if x=='lead' else 0)
        X_copy['fpStatus'] = X_copy['fpStatus'].apply(lambda x: 1 if x=='Checked' else 0)
        X_copy = X_copy.drop(['originated', 'approved', 'isFunded', # single occurence
                             'loanId', 'anon_ssn', 'underwritingid', # IDs
                             'applicationDate', 'originatedDate', 'lastPaymentDate'], # Dates
                             axis=1)
        numeric_features = ['payFrequency', 'apr', 'nPaidOff', 'loanAmount',
                           'originallyScheduledPaymentAmount', 'state', 'leadType', 'leadCost',
                           'hasCF', 'noOfInstallmentMade', 'avgInstallmentAmountPerPayFrequency',
                           'clearfraudscore', 'lastPaymentMonth',
                           'lastPaymentYear', 'lastPaymentSeason', 'totalPayFrequency',
                           'expectedDaysUntilPayoff', 'fpStatus']
        return X_copy[numeric_features]

