from glob import glob


import json

from conllu import *

from syntagrusbin import *
from scalers import *
from prepare_sets import *
from models import *

def main():
    config=input(".json setup config path: ")

    with open(config,encoding='utf-8') as f:
        config=json.load(f)

    exp_d=config["exp_dir"]

    #1. parse
    if "1" in config["stages"]:
        print("1. Parsing conllu...")
        dataset=ConlluDataset(config["unk_token"])
        dataset.compute(config["dataset_path"])
        datas=ConlluDataset.split_into_chunks(dataset,config['max_sentences'],exp_d+'/conllu_parts')
        print('Conllu parsed!')

    #2. Binarize
    if "2" in config["stages"]:
        print("2. Binarizing...")
        bin_=SynTagRusBinarizer(use_bert=config["use_bert"],bert_name=config["bert_name"], batch_size= config["batch_size"],
                                unk_token= config["unk_token"], use_head= config["use_head"], predict_head = config["predict_head"],
                                use_pos= config["use_pos"], use_morph= config["use_morph"], use_sentenceemb= config["use_sentenceemb"],
                                use_s_no= config["use_s_no"], use_distance= config["use_distance"], binary= config["binary"], sentence_batch= config["batch_sentence"]

                                )
        DatasetSplit.make_multiple_datasets(dataset_configuration=bin_,n=config["n_predictors"], export_path=exp_d+'/binarized_data', chunks_dir=exp_d+'/conllu_parts')
        print('Binarized!')

    #3. Balance
    if "3" in config["stages"]:
        print("3. Balancing...")

        scaler=config["scaler"]
        match(scaler):
            case "MinMaxScaler":
                scaler=MinMaxScaler()
            case "StandardScaler":
                scaler=StandardScaler()
            case _:
                raise ValueError("Select a valid scaler")


        DatasetBalance.process_all(exp_d+'/binarized_data', exp_d+'/balanced_data', scaler=scaler, test_size=config["test_size"], k=config["k"], balance=config["methods"])
        print("Balanced!")

    if "4" in config["stages"]:
        print("4. Training...")
        TrainModels.process_all(exp_d+'/balanced_data',config["model"],exp_d+'/results')
        print('Trained!')



if __name__=="__main__":
    main()
