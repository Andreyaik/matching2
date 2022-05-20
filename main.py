import numpy as np
import pandas as pd
import uuid
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer
from transformers import logging
from matcher import matcher
from prepare_dataloader import dataloader_my
from model import model_matching, out_model
logging.set_verbosity_error()

K_MATCHES = 10
NGRAM_LENGTH = 4
THRESHOLD = 0.4


def nn_match(new_item, all_dataset, model_m, tokenizer_m, threshold=0.5, k_matches=10, ngram_length=4):
    """ function finds a group with identical products for the input product name
    :param new_item: (str) name of the new product in string format
    :param all_dataset: (DataFrame) Data Frame in which we are looking for a group for a new product,
     with the name of the products (column "name_2") and the group ID (column "id_group")
    :param model_m: (object) created object of the pre-trained model
    :param tokenizer_m: (object) created object of the pre-emitted tokenizer
    :param threshold: (float) value of the threshold (the level of confidence after TFIDF matching) below which
    we believe that there is no group of identical products for a new product in the dataset
    :param k_matches: (int) number of the most similar products when matching using TFIDF vectors,
    which is further compared by a neural network
    :param ngram_length: (int) number of characters into which the product name is
    divided when forming the TF IDf vector
    :return: (int) number of the group to which the new product belongs
    (if there is no group, then a new group is created)
    """
    # find k_matches the most similar names by the TFIDF method
    idx_matched = []
    matched_tfidf = matcher(new_item, list(all_dataset['name_2'].values), k_matches=k_matches,
                            ngram_length=ngram_length)

    # if the confidence after TF IDF comparison is lower threshold, then this is a new position
    if matched_tfidf['Lookup 1 Confidence'].values[0] < threshold:
        # print('Dataset have not this position')
        new_id = uuid.uuid1()
        return new_id

    # select indexes of the most similar positions
    for i in range(1, k_matches + 1):
        name_col = f'Lookup {i} Index'
        idx_matched.append(matched_tfidf[name_col].values.reshape(-1, 1))
    idx_matched = np.hstack(idx_matched)

    anchor_vector = []
    anchor_iter = dataloader_my(list(new_item), tokenizer_m, max_length=256, batch_size=32)
    for i, batch in enumerate(anchor_iter):
        out = out_model(model_m, batch).detach().cpu()
        anchor_vector.append(out)
    out_anchor = torch.cat(anchor_vector, dim=0).reshape(-1, 1, 312)

    out_docs = torch.Tensor(np.array([out_model(model_m, tokenizer_m(list(all_dataset.loc[row, 'name_2'].values),
                                                                     padding=True, max_length=256, truncation=True,
                                                                     return_tensors='pt')).detach().cpu().numpy() for
                                      row in idx_matched]))

    result = idx_matched[
        np.arange(matched_tfidf.shape[0]), np.argmax(F.cosine_similarity(out_anchor, out_docs, dim=-1).numpy(), axis=1)]

    return all_dataset.loc[result, 'id_group'].values[0]


if __name__ == '__main__':
    model_match = model_matching()
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")

    print(nn_match([pd.read_csv('data/new_items_test.csv').loc[0, 'name_1']],
                   pd.read_csv('data/dataset_test_100.csv'), model_match,
                   tokenizer, threshold=THRESHOLD, k_matches=K_MATCHES,
                   ngram_length=NGRAM_LENGTH))






