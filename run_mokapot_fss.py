import os
import sys
import mokapot
import sklearn
import pandas as pd

print(mokapot.__version__)
print(sklearn.__version__)

from sklearn.ensemble import RandomForestClassifier

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

#set type of FSS
FSS = "backward" #back or forward

#set number of cpus mokapot can use
#note that cpus need tobe distributed over both mokapot and the classifier
num_mokapot_cpus = 10

#fixed model for FSS
cls = RandomForestClassifier(n_estimators=100,max_depth=48,n_jobs=25)

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

#features required in pin file
pre_f = ["SpecId","Label","ScanNr"]
post_f = ["Peptide","Proteins"]

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

result = []

if FSS == "backward":
    #backwards FSS
    removed_features = []
    for i in range(len(search_engine_features)-1):
        min_entrapment_fdr = 9999
        feat_to_remove = ""
        for feat in search_engine_features:
            if (feat == "RawScore") | (feat in removed_features):
                continue
            search_engine_features_tmp = []
            for f in search_engine_features:
                if (f in removed_features) | (f == feat):
                    continue
                search_engine_features_tmp.append(f)
            pin[pre_f+search_engine_features_tmp+post_f].to_csv(sys.argv[1]+".filtered",sep="\t")
            psms = mokapot.read_pin(sys.argv[1]+".filtered")
            results, models = mokapot.brew(psms,model=mokapot.model.Model(cls),max_workers=num_mokapot_cpus)
            results.to_txt(decoys=True)
            (num_target, entrapment_fdr) = parse_result()
            print([feat,num_target,entrapment_fdr])
            sys.stdout.flush()
            if entrapment_fdr < min_entrapment_fdr:
                min_entrapment_fdr = entrapment_fdr
                feat_to_remove = feat       
        print("Removed feature %s (%f)"%(feat_to_remove,min_entrapment_fdr)) 
        sys.stdout.flush()
        removed_features.append(feat_to_remove)
        result.append([i,feat_to_remove,num_target,entrapment_fdr])

if FSS == "forward":
    current_features = ["RawScore"]
    search_engine_features_tmp = []
    for f in current_features:
        search_engine_features_tmp.append(f)
    pin[pre_f+search_engine_features_tmp+post_f].to_csv(sys.argv[1]+".filtered",sep="\t")
    psms = mokapot.read_pin(sys.argv[1]+".filtered")
    results, models = mokapot.brew(psms,model=mokapot.model.Model(cls),max_workers=num_mokapot_cpus)
    results.to_txt(decoys=True)
    (num_target, entrapment_fdr) = parse_result()
    print(["RawScore",num_target,entrapment_fdr])
    result.append([-1,"RawScore",num_target,entrapment_fdr])
    for i in range(len(search_engine_features)-1):
        min_entrapment_fdr = 9999
        feat_to_add = ""
        for feat in search_engine_features:
            if (feat == "RawScore") | ( feat in current_features):
                continue
            search_engine_features_tmp = []
            for f in current_features:
                search_engine_features_tmp.append(f)
            search_engine_features_tmp.append(feat)
            pin[pre_f+search_engine_features_tmp+post_f].to_csv(sys.argv[1]+".filtered",sep="\t")
            psms = mokapot.read_pin(sys.argv[1]+".filtered")
            results, models = mokapot.brew(psms,model=mokapot.model.Model(cls),max_workers=num_mokapot_cpus)
            results.to_txt(decoys=True)
            (num_target, entrapment_fdr) = parse_result()
            print([feat,num_target,entrapment_fdr])
            sys.stdout.flush()
            if entrapment_fdr < min_entrapment_fdr:
                min_entrapment_fdr = entrapment_fdr
                feat_to_add = feat       
        print("Added feature %s (%f)"%(feat_to_add,min_entrapment_fdr)) 
        sys.stdout.flush()
        current_features.append(feat_to_add)
        result.append([i,feat_to_add,num_target,entrapment_fdr])

result = pd.DataFrame(result,columns=["iteration","feature","#PSMs","entrapment-FDR"])
result.to_csv(sys.argv[1]+".fss.result",index=False)
print("Result written to %s"%sys.argv[1]+".fss.result")
