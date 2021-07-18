
# -------------------------- # -------------------------- #
# Feature engineering module # -------------------------- #
# -------------------------- # -------------------------- #

import pandas as pd
from tqdm import tqdm
from commlit.pre_proc import gen_ent_df, gen_sent_df, gen_token_df
from commlit.word_vecs import gen_word_vec_feat

def gen_features(token_df, sent_df, ent_df=None, tag_df=None, all_ents=None):
    """
    """
    
    # pre-process
    comm_score_df = token_df.loc[~token_df["comm_score"].isna() & 
                                 (token_df["stop"]==False), 
                                ["word", "comm_score"]]
    comm_score_df = comm_score_df.drop_duplicates().sort_values("comm_score")
    word_df = token_df[token_df["alpha"]==True]
    uniq_df = word_df[["word", "length"]].drop_duplicates().sort_values("length")
    
    # generate main features
    features = {
        "word_len_avg": word_df["length"].mean(),
        "word_len_std": word_df["length"].std(),
        "word_len_top20": uniq_df.tail(20)["length"].mean(),
        "word_len_top10": uniq_df.tail(10)["length"].mean(),
        "word_len_top5": uniq_df.tail(5)["length"].mean(),
        "comm_score_avg": comm_score_df["comm_score"].mean(),
        "comm_score_std": comm_score_df["comm_score"].std(),
        "comm_score_top20": comm_score_df.tail(20)["comm_score"].mean(),
        "comm_score_top10": comm_score_df.tail(10)["comm_score"].mean(),
        "comm_score_top5": comm_score_df.tail(5)["comm_score"].mean(),
        "words_per_sent": sent_df["length"].mean(),
        "noun_chunks_per_sent": sent_df["noun_chunks"].mean(),
        "frac_stop": word_df["stop"].mean()
    }
    
    # add length buckets
    len_lab = ["frac_" + l for l in ["short", "medium", "long", "huge"]]
    len_df = word_df.copy()[word_df["length"] > 3]
    len_df.loc[:, "len_cat"] = pd.cut(len_df["length"], 
                                      [1, 5, 7, 10, 100], 
                                      labels=len_lab)
    features.update(len_df["len_cat"].value_counts(
        normalize=True).sort_index().to_dict())

    # add frequency buckets
    comm_lab = ["frac_" + l for l in 
                ["very_common", "common", "uncommon", "rare", "very_rare"]]
    comm_score_df.loc[:, "comm_cat"] = pd.cut(
        comm_score_df["comm_score"], 
        [0, 0.0005, 0.0025, 0.025, 0.1, 1],
        labels=comm_lab
    )
    features.update(comm_score_df["comm_cat"].value_counts(
        normalize=True).sort_index().to_dict())

    # get tag features
    if tag_df is not None:
        tag_feat = token_df["tag"].value_counts(normalize=True).reset_index()
        tag_feat.columns = ["tag", "tag_count"]
        tag_feat = tag_df.merge(tag_feat, on="tag", how="left").fillna(0)
        tag_dict = dict(zip(tag_feat["tag_adj"], tag_feat["tag_count"]))
        features.update(tag_dict)
        
    # gen ent features
    if ent_df is not None and all_ents is not None:
        
        # ent type concentrations
        ner_df = pd.DataFrame(all_ents, columns=["ent_type"])
        if ent_df.shape[0] > 0:
            ner_feat = ent_df["type"].value_counts(normalize=True).reset_index()
            ner_feat.columns = ["ent_type", "ent_count"]
            ner_df = ner_df.merge(ner_feat, on="ent_type", 
                                  how="left").fillna(0)
        else:
            ner_df.loc[:, "ent_count"] = 0
        ner_dict = dict(zip(ner_df["ent_type"], ner_df["ent_count"]))
        features.update(ner_dict)
        
        # general ent details
        if ent_df.shape[0] > 0:
            features["avg_ent_words"] = ent_df["n_words"].mean()
            features["avg_ent_len"] = ent_df["length"].mean()
        else:
            features["avg_ent_words"] = 0
            features["avg_ent_len"] = 0
        
    return features

def gen_feature_df(df, nlp, freq_df, 
                   tag_df=None, all_ents=None,
                   word_vec_feat=True, **kwargs):
    """
    """
    
    doc_tups = list(df[["excerpt","id"]].itertuples(index=False, name=None))
    feat_list = []
    for doc, i in tqdm(nlp.pipe(doc_tups, as_tuples=True)):

        # pre-process doc
        token_df = gen_token_df(doc, freq_df) 
        sent_df = gen_sent_df(doc)
        if all_ents is not None:
            ent_df = gen_ent_df(doc)
        else:
            ent_df = None

        # extract basic features
        features = gen_features(token_df, sent_df, ent_df, 
                                tag_df=tag_df, 
                                all_ents=all_ents)
        
        # add word-vec features if True
        if word_vec_feat:
            features.update(gen_word_vec_feat(doc, token_df, **kwargs))
            
        # include the doc id then append
        features["id"] = i    
        feat_list.append(features)
        
    feat_df = pd.DataFrame(feat_list).merge(df.drop(columns="excerpt"), 
                                            on= "id")
    return feat_df
