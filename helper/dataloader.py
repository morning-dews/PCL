'''
    created by zhangming chan
    for the dataloader
'''
import os
import queue

import pdb
import json
import time
import torch

import numpy as np

from torch.utils.data import Dataset, DataLoader
from collections import Counter
from threading import Thread
from tqdm import tqdm

import random
random.seed(2333)

class me_dict(dict):
    def __init__(self, default=None):
        dict.__init__(self)
        self.default = default

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            return self.default


class DealDataset(Dataset):
    '''
        deal the data for posting
        build the vocabulary

        in raw data, there are six kinds of data:
        id: the unique id for every record; str
        title: the title of the product; (title str, seg str)
        longtext: the 设计亮点 of product; (text str, seg str)
        shorttext: the 短亮点 of product; [(text str, seg str) * N]
        middletext: the 卖点 of product; [(text str, seg str) * N]
        knowledge: the knowledge of product; [(key str, value str) * M]
    '''

    def __init__(self, 
                 path, 
                 batch_size, 
                 enc_w2i=None, 
                 enc_i2w=None, 
                 dec_w2i=None, 
                 dec_i2w=None, 
                 min_freq=20, 
                 max_lens={'title': 20, 'subtitle': 3, 'long_text': 15, 'short_text': 6}, 
                 data_type='train'):

        self.data_type = data_type

        self.queue_raw = queue.Queue(10000)
        self.queue_record = queue.Queue(5000)
        self.queue_batch = queue.Queue(100)

        self.use_type = [
                            'item_id', 
                            'title', 
                            # 'subtitle', 
                            # 'long_text', 
                            # 'shorttext', 
                            'short_text', 
                            # 'knowledge',
                        ]

        self.batch_size = batch_size
        # the vocabulary
        self.enc_w2i = enc_w2i
        self.enc_i2w = enc_i2w
        self.dec_w2i = dec_w2i
        self.dec_i2w = dec_i2w

        self.shuffle = True    # shuffle the data
        self.limits = max_lens # the max length of all datas

        # file_path
        self.data_path = path
        if not os.path.exists(self.data_path): assert False, 'Please check the path of data!'

        ### Start the Thread for read raw data from the file
        self.thread_raw = Thread(target=self._read_file)
        self.thread_raw.daemon = True
        self.thread_raw.start()

        ### Start the Thread(s) for make the record from raw data
        self.thread_rec = Thread(target=self._make_record)
        self.thread_rec.daemon = True
        self.thread_rec.start()

        ### Start the Thread(s) for make the batch from record data
        self.thread_bat = Thread(target=self._make_batch)
        self.thread_bat.daemon = True
        self.thread_bat.start()


        self.enc_vocab_size = len(self.enc_w2i)
        self.dec_vocab_size = len(self.dec_w2i)

        self.vocab_pad = self.dec_w2i['<pad>']
        self.vocab_sos = self.dec_w2i['<sos>']
        self.vocab_eos = self.dec_w2i['<eos>']


    def _watch_thread(self):
        ### provide the Threads wokring
        while True:
            time.sleep(60)
            if self.thread_raw.is_alive():
                self.thread_raw = Thread(target=self._read_file)
                self.thread_raw.daemon = True
                self.thread_raw.start()
            if self.thread_rec.is_alive():
                self.thread_rec = Thread(target=self._make_record)
                self.thread_rec.daemon = True
                self.thread_rec.start()
            if self.thread_bat.is_alive():
                self.thread_bat = Thread(target=self._make_batch)
                self.thread_bat.daemon = True
                self.thread_bat.start()


    def _read_file(self):
        ### read the data from file
        print('Loading data from \'{}\' ...'.format(os.path.abspath(self.data_path)))
        while True:
            with open(self.data_path, 'r', encoding='utf-8') as reader:
                for record in reader:
                    record = json.loads(record)
                    self.queue_raw.put(record)

    def _make_record(self):
        ### raw data to record
        def _to_token(data_text, token_name, is_dec=False):
            token2id = self.enc_w2i if not is_dec else self.dec_w2i

            cur_sample = []
            if is_dec:
                data_text = data_text[:self.limits[token_name]-2]
            else:
                data_text = data_text[:self.limits[token_name]]

            for word in data_text:
                cur_sample.append(token2id[word])

            sample_len = len(cur_sample)

            if is_dec:
                sample = [token2id['<sos>']] + cur_sample + [token2id['<eos>']] + \
                          [token2id['<pad>']] * (self.limits[token_name]-sample_len)
            else:
                sample = cur_sample + [token2id['<pad>']] * (self.limits[token_name]-sample_len)

            return sample, sample_len

        while True:
            record = self.queue_raw.get()
            data_record = {}
            for type_key in self.use_type:
                if type_key in ['item_id']: 
                    data_record['item_id'] = record[type_key]
                elif type_key in ['title', 'subtitle', 'short_text', 'long_text']:
                    is_dec = True if type_key == 'long_text' else False
                    if type_key == 'long_text':
                        sample_return, sample_len = _to_token(''.join(record[type_key]), type_key, is_dec)
                    else:
                        sample_return, sample_len = _to_token(record[type_key], type_key, is_dec)
                    data_record[type_key] = sample_return
                    data_record[type_key+'_len'] = sample_len
                else:
                    assert False, 'Some error in convering data to number.'
                data_record['old_short'] = record['short_text']

            self.queue_record.put(data_record)


    def _make_batch(self):
        max_len = self.limits['long_text'] + 1
        while True:
            datas = {}
            for _ in range(self.batch_size):
                record = self.queue_record.get()
                for key, value in record.items():
                    if key in datas:
                        datas[key].append(value)
                    else:
                        datas[key] = [value]

                if 'long_text_len' in record:
                    cur_len = record['long_text_len'] + 1
                    cur_pos_f = np.arange(1, cur_len+1)
                    cur_pos_b = np.arange(cur_len, 0, -1)
                    for _ in range(max_len-cur_len):
                        cur_pos_f = np.append(cur_pos_f, 0)
                        cur_pos_b = np.append(cur_pos_b, 0)

                    cur_pos_f = cur_pos_f.tolist()
                    cur_pos_b = cur_pos_b.tolist()

                    if 'pos_enc_f' in datas:
                        datas['pos_enc_f'].append(cur_pos_f)
                    else:
                        datas['pos_enc_f'] = [cur_pos_f]

                    if 'pos_enc_b' in datas:
                        datas['pos_enc_b'].append(cur_pos_b)
                    else:
                        datas['pos_enc_b'] = [cur_pos_b]

            for key in datas.keys():
                if key not in ['item_id', 'old_short']:
                    datas[key] = torch.tensor(datas[key])

            self.queue_batch.put(datas)

    def __next__(self):
        return self.queue_batch.get()

    def __iter__(self):
        return self

    def __len__(self):
        # if self.data_type in ['dev', 'test']:
        #     return 5000 // self.batch_size
        # else:
        #     return 490000 // self.batch_size

        if self.data_type in ['dev', 'test']:
            # return 1240322 // self.batch_size
            return 26958 // self.batch_size
        else:
            return 3497984 // self.batch_size


def get_dataloader(path, 
                   batch_size=64, 
                   data_type=['train', 'dev', 'test']):

    min_freq = 15
    # make the vocabulary
    enc_w2i = me_dict(1)
    enc_w2i['<pad>'] = 0
    enc_w2i['<unk>'] = 1
    with open(path + 'encoder_words', encoding='utf-8') as reader:
        for idx, line in enumerate(reader):
            line = line.split('\t')
            if idx > 100000: break
            if min_freq > int(line[1]): break
            enc_w2i[line[0]] = len(enc_w2i)

    dec_w2i = me_dict(1)
    dec_w2i['<pad>'] = 0
    dec_w2i['<unk>'] = 1
    dec_w2i['<sos>'] = 2
    dec_w2i['<eos>'] = 3
    with open(path + 'decoder_words', encoding='utf-8') as reader:
        for line in reader:
            line = line.split('\t')
            dec_w2i[line[0]] = len(dec_w2i)

    enc_i2w = {v: k for k, v in enc_w2i.items()}
    dec_i2w = {v: k for k, v in dec_w2i.items()}

    dec_w2i['<sos>'] = dec_w2i['，']

    dataloader_set = {}

    for d_type in data_type:
        dataset = DealDataset(path + '{}.json'.format(d_type), 
                              batch_size=batch_size if d_type not in ['test'] else 1, 
                              data_type=d_type,
                              enc_w2i=enc_w2i,
                              enc_i2w=enc_i2w,
                              dec_w2i=dec_w2i,
                              dec_i2w=dec_i2w)

        dataloader_set[d_type] = dataset

    return dataloader_set


if __name__ == '__main__':
    dataloader_set = get_dataloader('../data/50W/')

    for idx, w in tqdm(enumerate(dataloader_set['train'])):        
        if max(w['short_text_len']) > 15:
            pdb.set_trace()

    pdb.set_trace()
