from ucimlrepo import fetch_ucirepo
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols 
import numpy as np

# Variables of Interest
dv = 'time_in_hospital'
iv = 'race'
iv2 = 'A1Cresult'

Diabetes = fetch_ucirepo(id=296)

X = Diabetes.data.features
y = Diabetes.data.target


df = pd.DataFrame(X)
df2 = df[['race', 'A1Cresult', 'time_in_hospital']]
df2 = df2.dropna()
df2['race'] = df2['race'].astype(str)
df2['A1Cresult'] = df2['A1Cresult'].astype(str)
df2['time_in_hospital'] = df2['time_in_hospital'].astype('int64')
print(df2.columns)
print(df2.head())
print(df2.dtypes)

plt.hist(df2['time_in_hospital'], bins = 20, edgecolor = 'k', alpha = 0.7)
plt.title('Distribution of Time in Hospital')
plt.xlabel('Time in Hospital')
plt.ylabel('Count')
plt.show()

def test_normality(df2, alpha = 0.05):
    stat, p = stats.shapiro(df2['time_in_hospital'])
    if p > alpha:
        print("Data looks normally distributed (fail to reject H0)")
    else:
        print("Data does not look normally distributed (reject H0)")

print(test_normality(df2))

### The normality showed an abnormally high number and the data is not normally distributed.

def test_homogeniety(*args):
    stat, p = stats.levene(*args)
    if p > 0.05:
        print("Variances are equal (fail to reject H0)")
    else:
        print("Variances are not equal (reject H0)")

groups = df2.groupby(['race', 'A1Cresult'])

for (races, A1Cresults), group_df2 in groups:
    _, p = stats.shapiro(group_df2['time_in_hospital'])

    print(f"Group ({races}, {A1Cresults})")
    print(f"P-value from Shapiro-Wilk test: {p}\n")

### All of these showed statstically significant values except the group fo Caucasian and >8 which showed 0.0. The fartherst from 0.05 was the Caucasian Norm group
### showing 1.9e-41.

levene_test = stats.levene(
    df2['time_in_hospital'][df2['race'] == 'AfricanAmerican'][df2['A1Cresult'] == '>8'],
    df2['time_in_hospital'][df2['race'] == 'Asian'][df2['A1Cresult'] == '>8'],
    df2['time_in_hospital'][df2['race'] == 'Caucasian'][df2['A1Cresult'] == '>8'],
    df2['time_in_hospital'][df2['race'] == 'Hispanic'][df2['A1Cresult'] == '>8'],
    df2['time_in_hospital'][df2['race'] == 'Other'][df2['A1Cresult'] == '>8'],
    df2['time_in_hospital'][df2['race'] == 'AfricanAmerican'][df2['A1Cresult'] == '>7'],
    df2['time_in_hospital'][df2['race'] == 'Asian'][df2['A1Cresult'] == '>7'],
    df2['time_in_hospital'][df2['race'] == 'Caucasian'][df2['A1Cresult'] == '>7'],
    df2['time_in_hospital'][df2['race'] == 'Hispanic'][df2['A1Cresult'] == '>7'],
    df2['time_in_hospital'][df2['race'] == 'Other'][df2['A1Cresult'] == '>7'],
    df2['time_in_hospital'][df2['race'] == 'AfricanAmerican'][df2['A1Cresult'] == 'Norm'],
    df2['time_in_hospital'][df2['race'] == 'Asian'][df2['A1Cresult'] == 'Norm'],
    df2['time_in_hospital'][df2['race'] == 'Caucasian'][df2['A1Cresult'] == 'Norm'],
    df2['time_in_hospital'][df2['race'] == 'Hispanic'][df2['A1Cresult'] == 'Norm'],
    df2['time_in_hospital'][df2['race'] == 'Other'][df2['A1Cresult'] == 'Norm'],
)

model = ols('time_in_hospital ~ C(race) * C(A1Cresult)', data=df2).fit()

anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

###                             sum_sq       df         F    PR(>F)
### C(race)                  167.770289      4.0  4.335662  0.001664
### C(A1Cresult)             110.390357      2.0  5.705602  0.003334
### C(race):C(A1Cresult)      59.979436      8.0  0.775020  0.624827
### Residual              160402.223857  16581.0       NaN       NaN

###