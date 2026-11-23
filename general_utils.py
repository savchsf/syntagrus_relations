from glob import glob
import os

def get_prefixes(directory):
    paths=glob(directory+'/*.json') #counting prefixes by json
    prefixes=[os.path.split(x)[-1].replace('.json','') for x in list(paths) if '_bow' not in x]
    return prefixes