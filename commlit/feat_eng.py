
# -------------------------- # -------------------------- #
# Feature engineering module # -------------------------- #
# -------------------------- # -------------------------- #

import numpy as np
import pandas as pd
from tqdm import tqdm
from commlit.pre_proc import gen_ent_df, gen_sent_df, gen_token_df
from commlit.base_feats import gen_base_features
from commlit.word_vecs import gen_word_vec_feat, gen_word_vec_matrix

def gen_batch_features(df, nlp, freq_df, 
                       tag_df=None, all_ents=None,
                       n_rep=1, noisy_y=False,
                       word_vec_raw=False,
                       word_vec_feat=False, **kwargs):
    """
    """
    
    doc_tups = list(df[["excerpt","id"]].itertuples(index=False, name=None))
    feat_vec = []
    word_vec = []
    y_vec = []
    id_vec = []
    
    for doc, i in tqdm(nlp.pipe(doc_tups, as_tuples=True)):

        # pre-process doc
        token_df = gen_token_df(doc, freq_df) 
        sent_df = gen_sent_df(doc)
        if all_ents is not None:
            ent_df = gen_ent_df(doc)
        else:
            ent_df = None

        # get target value and its st. dev.
        y, sd = df.query("id == @i")[["target", "standard_error"]].values[0]
        
        # extract basic features
        features = gen_base_features(token_df, sent_df, ent_df, 
                                     tag_df=tag_df, all_ents=all_ents)
        features["id"] = i
        
        # add word-vec features if True
        if word_vec_feat:
            features.update(gen_word_vec_feat(doc, token_df, **kwargs))
        
        # repeat feature extraction if desired
        for n in range(n_rep):
            
            # add new set of basic features and ids each time
            id_vec.append(i)
            feat_vec.append(features)
            
            # extract word vectors and re-sample
            if word_vec_raw:
                word_vec.append(gen_word_vec_matrix(doc))
              
            # add noise to y-measuers or append as is   
            if noisy_y:
                y_vec.append(y + sd*np.random.randn(1))
            else:
                y_vec.append(y)
    
    # compile, format and return data    
    feat_df = pd.DataFrame(feat_vec)
    y_vec = np.array(y_vec)
    if word_vec_raw:
        word_vec = np.stack(word_vec)[:, :, :, np.newaxis]
        return feat_df, word_vec, y_vec, id_vec
    else:
        return feat_df, y_vec, id_vec
