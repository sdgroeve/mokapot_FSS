import os
import sys
import mokapot
import sklearn
import pandas as pd

print(mokapot.__version__)
print(sklearn.__version__)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def parse_result():
    num_target = 0
    num_entrap = 0
    with open("mokapot.psms.txt") as f:
        f.readline()
        for row in f:
            l = row.rstrip().split("\t")
            if float(l[5]) <= 0.01:
                entrap = True
                for pr in l[7:]:
                    if not "ENTRAPMENT" in pr:
                        entrap = False
                if entrap == True:
                    num_entrap += 1
                else:
                    num_target += 1
    num_decoys = 0
    with open("mokapot.decoy.psms.txt") as f:
        f.readline()
        for row in f:
            l = row.rstrip().split("\t")
            if float(l[5]) <= 0.01:
                num_decoys += 1
    return (num_target,num_entrap/num_target)

#features required in pin file
pre_f = ["SpecId","Label","ScanNr"]
post_f = ["Peptide","Proteins"]

#MaxQuant feaures
search_engine_features = [
    "absdM",
    "Charge1",
    "Charge2",
    "Charge3",
    "Charge4",
    "Charge5",
    "Charge6",
    "Charge7",
    "ChargeN",
    "dM",
    "enzInt",
    "lnCTermIonCurrentRatio",
    "lnExplainedIonCurrent",
    "lnMS2IonCurrent",
    "lnNTermIonCurrentRatio",
    "Mass",
    "MeanErrorTop7",
    "PepLen",
    "RawDeltaScore",
    "RawModLocProb",
    "RawScore",
    "sqMeanErrorTop7",
    "StdevErrorTop7"
]

#selected AI features
ai_features = ["rt_diff","rt_diff_best","cos","dotprod","spec_pearson_norm","spec_mse","spec_pearson"]

#Reading pin file
#Can't do this with Pandas as number of columns is not equal
X_cols = []
X = []
with open(sys.argv[1]) as f:
    X_cols = f.readline().rstrip().split("\t")
    for row in f:
        l = row.rstrip().split("\t")
        l = l[:len(X_cols)]
        X.append(l)
pin = pd.DataFrame(X,columns=X_cols)
pin = pin[pre_f+search_engine_features+post_f]

#Mokapot reads pin from file only
pin.to_csv(sys.argv[1]+".filtered",sep="\t")
psms = mokapot.read_pin(sys.argv[1]+".filtered")

results, models = mokapot.brew(psms,max_workers=25)
results.to_txt(decoys=True)

result = []

print("Running baseline.")
(num_target, entrapment_fdr) = parse_result()
result.append(["LSVM",0,num_target,entrapment_fdr])

print("Running XGBoost.")
hyper_params = {'scale_pos_weight': [1, 10, 100],
                'max_depth': [1, 3, 6],
                'min_child_weight': [1, 10, 100],
                'gamma': [0, 1, 10], }    
gs = GridSearchCV(
    estimator=xgb.XGBClassifier(),
    param_grid=hyper_params,
    # scoring="roc_auc",
    n_jobs=25,  # -1 means all processors
    verbose=0,
    cv=5
)
mcls = mokapot.model.Model(gs)
results, models = mokapot.brew(psms,model=mcls,max_workers=10)
results.to_txt(decoys=True)
(num_target, entrapment_fdr) = parse_result()
result.append(["XGboost",0,num_target,entrapment_fdr])

print("Running Random Forest.")
for m in [2,6,12,24,48]:
    mcls = mokapot.model.Model(RandomForestClassifier(n_estimators=100,max_depth=m,n_jobs=25))
    results, models = mokapot.brew(psms,model=mcls,max_workers=10)
    sys.stdout.write("RF %i "%m)
    results.to_txt(decoys=True)
    (num_target, entrapment_fdr) = parse_result()
    result.append(["RF",m,num_target,entrapment_fdr])

result = pd.DataFrame(result,columns=["algorithm","max_depth","#PSMs","entrapment-FDR"])
result.to_csv(sys.argv[1]+".result.csv",index=False)
print("Result written to %s"%sys.argv[1]+".result.csv")

