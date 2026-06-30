from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import random

import logging; init_logger = lambda: logging.getLogger(__name__)
logger = init_logger()

 
def randomReplacePrompt(sentence):
    words = nltk.word_tokenize(sentence)
    replace_count = round(len(words) * 0.15)
    idxs = np.random.permutation(np.arange(len(words)))[:replace_count]
    for index in idxs:
        synonym = words[index]
        synonym_list = wn.synsets(words[index])
        if len(synonym_list) > 0:
            synonym_list = random.choice(synonym_list).lemmas()
            if len(synonym_list) > 0:
                synonym = random.choice(synonym_list).name()
        # print("replace {} with {}".format(words[index], synonym))
        words[index] = synonym
    new_sentence = ' '.join(words).replace(" .", ".")
    # print(new_sentence)
    return new_sentence

def augument(df: pd.DataFrame):
    new_df = df.copy()
    for idx, row in new_df.iterrows():
        sentence = row['prompts']
        new_sentence = randomReplacePrompt(sentence)
        new_df.loc[idx, 'prompts'] = new_sentence
    return new_df

class Conversation(Dataset):
    def __init__(self, filename, response_type, min_len=16, max_len=512, mode='train', resample=1):
        super(Conversation, self).__init__()
        df = pd.read_csv(filename)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"Total samples: {len(df)}")
        train_df = df[:int(len(df) * 0.8)]

        if mode == 'train':
            self.df = train_df
            logger.info(f"Train samples: {len(self.df)}")
            if resample:
                nltk.download('punkt_tab')
                nltk.download('punkt')
                nltk.download('wordnet')
                self.bin_size = 10
                self.bins = np.arange(min_len, max_len+1, (max_len-min_len+1) / self.bin_size)
                self.df.loc[:, 'bin_index'] = np.digitize(self.df['response_lens'].values, self.bins)
                groups = self.df.groupby('bin_index')
                max_group_size = groups.size().max()
                # sampled_groups = groups.apply(lambda x: x.sample(n=max_group_size, random_state=42))
                # self.df = sampled_groups.reset_index(drop=True)
                new_df = []
                for group_name, group_df in groups:
                    num_to_replcaicate = int(0.25 * (max_group_size - len(group_df)))
                    replicate_df = group_df.copy()
                    while(num_to_replcaicate > len(group_df)):
                        new_group_df = augument(group_df)
                        replicate_df = pd.concat([replicate_df, new_group_df])
                        num_to_replcaicate -= len(group_df)
                    if num_to_replcaicate > 0:
                        replicate_df = pd.concat([replicate_df, group_df.sample(n=num_to_replcaicate, random_state=42)])
                    new_df.append(replicate_df)
                self.df = pd.concat(new_df).reset_index(drop=True)
                logger.info(f"Train replcated samples: {len(self.df)}")
        elif mode == 'test':
            self.df = df[int(len(df) * 0.7):].reset_index(drop=True)
            logger.info(f"Test samples: {len(self.df)}")


        self.response_bins = self.create_bins(train_df['response_lens'].values, response_type)
        logger.info(f"classification bins: {self.response_bins}")
    
    
    def create_bins(self, data, num_bins=4):
        sorted_data = np.sort(data)
        unique_data, counts = np.unique(sorted_data, return_counts=True)
        probabilities = counts / len(data)
        cumsum_prob = np.cumsum(probabilities)
        bins_prob = 1.0 / num_bins
        bins_split = []
        for i in range(num_bins-1):
            for j, prob in enumerate(cumsum_prob):
                if prob >= (i+1) * bins_prob:
                    bins_split.append(unique_data[j] + 0.5)
                    break
        return bins_split
    
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        prompt, response_len = self.df.loc[index, 'prompts'], self.df.loc[index, 'response_lens']
        
        response_bin_index = np.sum(response_len > self.response_bins)
        return prompt, response_bin_index, response_len