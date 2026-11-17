from glob import glob
import json

import numpy as np

from prepare_sets import *


from models import *

def collect_all_rel(directory):
    #C:\Users\ydls\Desktop\rel\full2\results
    all_js=glob(directory+'/*.json')
    #print(all_js)
    schemes=[x for x in all_js if 'bow' not in x]
    schemes=list(schemes)
    schemes.sort()

    rel=['None']

    for s in schemes:
        with open(s) as f:
            s=json.load(f)
        r=s['outputs']
        r_k=list(r.keys())
        r_v=[r[x] for x in r_k]
        rel.extend(r_v)

    return rel


def one_test_xy(directory,heads=False):
    #balanced_data\3_y_train.npy"
    all_tst=glob(directory+'/*_y_test.npy')
    if heads:
        all_tst=[x for x in all_tst if 'heads' in x]
    else:
        all_tst=[x for x in all_tst if 'heads' not in x]

    all_tst=list(all_tst)
    all_tst.sort()
    #2. load X_train, y_train
    X_test=[]
    y_test=[]
    pred_n=[]

    for i in all_tst:
        x_loc=i.replace('y_test','X_test')
        y_loc=i
        X=np.load(x_loc,mmap_mode='r')
        y=np.load(y_loc,mmap_mode='r')
        X=list(X)
        #3. y to global index
        pr=i.split('/')[-1].replace('_y_test.npy','').replace('_heads','')
        pr=int(pr)
        if not heads:
            
            y=list(y)
            #y=[t+pr if t!=0 else 0 for t in y ]
        X_test.extend(X)
        y_test.extend(y)
        pred_n.extend([pr]*len(y))
    
    X_test=np.asarray(X_test)
    y_test=np.asarray(y_test)
        
    return X_test, y_test,pred_n

import joblib


def pred_one_group(models_dir,scalers_dir,heads=True):
    preds=glob(models_dir+"/*_model.pth")
    models=[x for x in preds if 'head' not in x]
    if heads:
        models_h=[x for x in preds if 'head' in x]

    prefs=[int(x.split('/')[-1].replace('_model.pth','').replace('_heads','')) for x in preds]

    prefs=list(set(prefs))
    prefs.sort()

    X_test, y_test,pred_n=one_test_xy(scalers_dir) #pred_n indicates where it sjould stop
    X_test=X_test[:,:-1]

    #print(len(X_test),len(y_test),len(pred_n))

    if heads:
        X_test_h, y_test_h,pred_n_h=one_test_xy(scalers_dir,heads=True)
        X_test_h=np.asarray(X_test_h)
        y_test_h=np.asarray(y_test_h)
        pred_n_h=np.asarray(pred_n_h)

    X_test=np.asarray(X_test)
    y_test=np.asarray(y_test)
    y_test_old=np.asarray([0]*len(y_test))
    pred_n=np.asarray(pred_n)

    y=[]
    y_gt=[]

    n=len(X_test)
    y=[0]*n
    y_gt=[0]*n
    pred_mask=[False]*n
    gt_mask=[False]*n

    for p in prefs:
        X_test_c=X_test
        y_test_c=y_test

        scaler=joblib.load(scalers_dir+f'/{p}_scaler.pkl')

        if heads:
            X_test_h_c=X_test_h
            h_model=joblib.load(models_dir+f'/{p}_heads_model.pth')
            h_model.scaler=scaler
            h_pred=h_model.predict(X_test_h_c)
            X_test_c=np.column_stack((X_test_c,h_pred))

        model=joblib.load(models_dir + f'/{p}_model.pth')
        model.scaler=scaler
        y_pred=model.predict(X_test_c)
        y_pred.tolist()
        y_pred=[round(x) for x in y_pred]

        if int(p)==0:
            add=0
        else:
            add=0
            for u in range(0,int(p)):
                with open(models_dir+f'/{u}.json') as f:
                    s=json.load(f)
                add+=len(s['outputs'])

        
        ch_idx=[i for i in range(n) if pred_mask[i]==False and int(y_pred[i])!=0]
        for i in ch_idx:
            y[i]=int(y_pred[i])+add
            pred_mask[i]=True

        ch_idx_gt=[i for i in range(n) if gt_mask[i]==False and int(pred_n[i])==int(p) and y_test[i]!=0]
        for i in ch_idx_gt:
            y_gt[i]=int(y_test[i])+add
            gt_mask[i]=True

    return y, y_gt


def compute_m(pred,test,mapping):

    pred=np.asarray([round(x) for x in pred])

    acc=accuracy_score(test, pred)
    report=classification_report(test, pred, zero_division=0)

    report=report.split('\n')
    new_re='idx   dep                             precision              recall                  f-score\n\n'
    for line in report[2:-5]:
        line=line.rsplit()
        #           5       0.83      0.28      0.42      9385
        #print(line)
        no=line[0]
        pre=line[1]
        rec=line[2]
        f=line[3]
        new_re+=f'{no:<6}{mapping[int(no)]:<22}\t\t{pre:<8}\t\t{rec:<8}\t\t{f}\n'

    rr=report[-4].rstrip().rsplit()
    new_re+=f'\n{rr[0]}\t\t{rr[1]}\n'
    return new_re

#print(collect_all_rel('full3/results'))
#one_test_xy('full2/balanced_data')[-1]

#pred,test=eval('full3/results','full3/balanced_data')

y,y_gt=pred_one_group('full3/results','full3/balanced_data')

all_r=collect_all_rel('full3/results')

print(len(all_r))
print(all_r)

new_re=compute_m(y,y_gt,all_r)
with open('report.txt','w') as f:
    f.write(new_re)
print(new_re)