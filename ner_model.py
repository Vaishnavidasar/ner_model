import os
os.getcwd()
import pandas as pd
import spacy
from tqdm import tqdm
from spacy.tokens import DocBin
import random
import numpy as np
import csv
import pre_processing
import model

def ner_label(data):
    lst=[]
    lst2=[]
    data2 = data.loc[data['label'] ==1 ]
    data2 = data2['Sentence']
    df = data2.to_string()
    nlp = spacy.load("output/model-best")
    doc = nlp(df)
    for ent in doc.ents:
        lst.append(ent.text)
        lst2.append(ent.label_)
        final = pd.DataFrame(list(zip(lst, lst2)),columns =['Name', 'val'])
        final.to_csv('ner.csv',index = False)
    return('NER performed')