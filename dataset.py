from tokenizers import Tokenizer
from tokenizers.normalizers import Lowercase, Strip, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer

import os
import json, glob
import pandas as pd
import numpy as np

from utils import merge_jsonl
from object_detection import extract_features
from ocr import detect_text

import torch
from torch.utils.data import random_split, DataLoader, Dataset

class MemesDataset(Dataset):
        def __init__(self, data_list, tokenizer, txt_seq_len, obj_seq_len, data_dir, num_image_features, num_classes) -> None:
                super().__init__()
                self.data_list = data_list
                self.tokenizer = tokenizer
                self.txt_seq_len = txt_seq_len
                self.obj_seq_len = obj_seq_len
                self.data_dir = data_dir
                self.num_classes = num_classes

                self.text_input_format = '[SOS] {sen} [EOS]'
                self.pad_token = tokenizer.token_to_id('[PAD]')
                self.num_image_features = num_image_features
                self.img_pad = np.array([0 for _ in range(num_image_features)])

        def __len__(self):
                return len(self.data_list)
        
        def __getitem__(self, index):
                data = self.data_list[index]
                
                # text encoder input 
                # text = detect_text(os.path.join(self.data_dir, data['img']))
                text = data['text']
                text_sentence = self.text_input_format.format(sen = text)
                txt_encoder_input = self.tokenizer.encode(text_sentence)
                txt_encoder_input.pad(pad_id=self.pad_token, length=self.txt_seq_len)
                txt_encoder_input = torch.tensor(txt_encoder_input.ids, dtype=torch.int64) # (txt_seq_len, )
                del text_sentence

                # image encoder input
                object_features = extract_features(os.path.join(self.data_dir, data['img']))
                padding_count = self.obj_seq_len-len(object_features)
                # print(object_features.shape, np.zeros((padding_count,self.num_image_features), np.int64).shape)
                img_encoder_input = np.concatenate([object_features, np.zeros((padding_count,self.num_image_features), np.int64)])
                img_encoder_input = torch.tensor(img_encoder_input, dtype=torch.float32)# (obj_seq_len, num_image_features)
                mask_len = object_features.shape[0]
                del object_features
                
                # label output
                label = np.eye(self.num_classes)[data['label']]
                label = torch.tensor(label, dtype=torch.float32) # (num_classes)
                
                # print(txt_encoder_input.size(0), self.txt_seq_len)
                assert txt_encoder_input.size(0)==self.txt_seq_len
                assert img_encoder_input.size(0)==self.obj_seq_len
                assert label.size(0)==self.num_classes

                return {
                        'txt_encoder_input': txt_encoder_input, # (txt_seq_len, )
                        'img_encoder_input': img_encoder_input, # (1, obj_seq_len, num_image_features)
                        'label': label, # (1,num_classes)
                        'text_encoder_mask': (txt_encoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len)
                        'img_encoder_mask': torch.tensor(np.concatenate([torch.ones(mask_len), np.zeros(padding_count)])).unsqueeze(0).unsqueeze(0).int(),
                        'text': data['text'],
                        'detected_text': text
                }

def get_data(f):
        with open(f, 'r') as file:
                for line in file:
                        yield json.loads(line)          
                
def get_tokenizer(config, lower=True):
        path = os.path.join(os.getcwd(),config['data_path'],config['tokenizer_file'])
        text = config['annotation_file'].format(name='text')
        print('>> Building tokenizer..')
        if not os.path.exists(path):
                print('>> Tokenizer not found, building from scratch.')
                tokenizer = Tokenizer(WordPiece(unk_token='[UNK]'))
                if lower:
                        tokenizer.normalizer =  Sequence([Strip(), Lowercase()])
                else:
                        tokenizer.normalizer = Strip()
                tokenizer.pre_tokenizer = Whitespace()
                trainer = WordPieceTrainer(show_progress=True, min_frequency=1, special_tokens=['[UNK]','[PAD]','[SOS]','[EOS]'])
                if not os.path.exists(text):
                        print('>> Generating the entire text corpos..')
                        jsonl_files = glob.glob(config['annotation_file'].format(name='*'))
                        merge_jsonl(jsonl_files, text)

                tokenizer.train_from_iterator(get_data(text), trainer)
                tokenizer.save(path, pretty=True)
        else:
                print(f'>> Tokenizer found at {path}.')
                tokenizer = Tokenizer.from_file(path)
        return tokenizer

def get_dataset(config):
        data_list = config['data_list'].format(name='data_list')
        print('\nData Preprocessing..')
        if not os.path.exists(data_list):
                ls = []
                print(f">> Data list file not found. Downloading and saving..")
                with open(config['annotation_file'].format(name='train'), 'r') as file:
                        for line in file:
                                ls.append(json.loads(line)) 
                df = pd.DataFrame(ls)
                df.to_csv(data_list, index=False)
                del ls
        else:
                print(f'>> Data list found at {data_list}')
                print(">> Loading the data list file..")
                df = pd.read_csv(data_list)
        
        tokenizer = get_tokenizer(config, lower=True)
        
        max_seq_len = 0
        # for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Count text max seq. length'):
        #         src_seq_len = len(tokenizer.encode(row.text))
        #         max_seq_len = max(max_seq_len, src_seq_len)
        
        max_seq_len = df.text.apply(lambda x: len(tokenizer.encode(x))).max()
        print(f'>> Maximum text sequence length:', max_seq_len)
        config['seq_len'] = max_seq_len

        train_size = int(config['dataset_split_ratio']*df.shape[0])
        val_size = df.shape[0]-train_size
        train_dataset_raw, val_dataset_raw = random_split(df.to_dict(orient='records'), lengths=[train_size, val_size])
        del df

        params = {
                'tokenizer':tokenizer, 
                'txt_seq_len':max_seq_len, 
                'obj_seq_len':config['obj_seq_len'], 
                'data_dir':config['memes_data_dir'], 
                'num_image_features':config['num_image_features'], 
                'num_classes':config['num_classes']
        }

        train_dataset = MemesDataset(data_list=train_dataset_raw, **params)
        val_dataset = MemesDataset(data_list=val_dataset_raw, **params)

        train_dataset = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_dataset = DataLoader(val_dataset, batch_size=1)

        return {
                'train_dataset': train_dataset,
                'val_dataset': val_dataset,
                'tokenizer': tokenizer
        }
        

