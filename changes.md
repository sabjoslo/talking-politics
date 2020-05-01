An evolving list of changes to the calculation of the CRec metrics that would result in a discrepancy between current data and data used to generate experimental stimuli for studies 1 & 2.

- Include an L1 norming scheme (impute 1 observation for every term for every party). Previously, the BitCounter() imputed `sys.float_info.epsilon` on-the-fly (as it encountered observations of 0).
- Cut stopwords **before** counting words. Previously, stopwords were manually excluded from lists of candidate survey items.
- Use the *q* distribution in the denominator when calculating *PKL* and *logodds*. Previously, the *m* distribution was used.
