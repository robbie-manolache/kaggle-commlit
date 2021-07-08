
# -------------------------- # -------------------------- #
# Feature engineering module # -------------------------- #
# -------------------------- # -------------------------- #

import pandas as pd
from tqdm import tqdm

def gen_tag_df(all_tags, tag_map=None):
    """
    """
    
    # define tag_map
    if tag_map is None:
        tag_map = [("$", "CASH"), ("''", "QUOTE"), ("``", "QUOTE"), 
                   (",", "COMMA"), ("-LRB-", "BKT"), ("-RRB-", "BKT"), 
                   (":", "COLON"), (".", "PUNCT"), ("_SP", "SP"), 
                   ("WP$", "WPP"), ("PRP$", "PRPP")]
        tag_map = pd.DataFrame(tag_map, columns=["tag_og", "tag_new"])
        
    # adjust tags and create output
    adj_tags = []
    for tag in all_tags:
        if tag in tag_map["tag_og"].tolist():
            adj_tags.append(tag_map.query("tag_og == @tag")["tag_new"].values[0])
        else:
            adj_tags.append(tag)
    tag_df = pd.DataFrame(zip(all_tags, adj_tags), columns=["tag", "tag_adj"])
    
    return tag_df

def gen_token_df(doc, freq_df=None, freq_norm=1e5):
    """
    Generates DataFrame of characteristics for tokens in a spacy doc.
    """

    # Create token dataframe
    token_list = [{
        "word": token.text.lower(),
        "lemma": token.lemma_,
        "length": len(token),
        "pos": token.pos_,
        "tag": token.tag_,
        "alpha": token.is_alpha,
        "stop": token.is_stop,
        "punct": token.is_punct
    } for token in doc]
    token_df = pd.DataFrame(token_list)
    
    # Add word frequencies if available
    if freq_df is not None:
        token_df = token_df.merge(freq_df, on="word", how="left")
        token_df.loc[:, "comm_score"] = freq_norm / token_df["count"]
    
    return token_df

def gen_sent_df(doc):
    """
    Generates DataFrame of characteristics for sentences in a spacy doc.
    """   
    
    sent_list = [{
        "length": len(sent),
        "noun_chunks": len(list(sent.noun_chunks))
    } for sent in doc.sents]
    
    return pd.DataFrame(sent_list)


def gen_ent_df(doc):
    """
    Generates DataFrame of characteristics for entities in a spacy doc.
    """   
    
    ent_list = [{
        "entity": ent.text.lower(),
        "n_words": len(ent),
        "length": len(ent.text),
        "type": ent.label_
    } for ent in doc.ents]
    
    return pd.DataFrame(ent_list)

def gen_features(token_df, sent_df, ent_df=None, tag_df=None,
                 all_ents=None):
    """
    """
    
    # pre-process
    comm_score_df = token_df.loc[~token_df["comm_score"].isna() & 
                                 (token_df["stop"]==False), 
                                ["word", "comm_score"]]
    comm_score_df = comm_score_df.drop_duplicates().sort_values("comm_score")
    word_df = token_df[token_df["alpha"]==True]
    uniq_df = word_df[["word", "length"]].drop_duplicates().sort_values("length")
    n_words = word_df.shape[0]
    n_sents = sent_df.shape[0]
    
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
        "words_per_sent": n_words/n_sents,
        "noun_chunks_per_sent": sent_df["noun_chunks"].mean(),
        "frac_stop": word_df["stop"].mean()
    }
    
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

def gen_feature_df(df, nlp, freq_df, tag_df=None, all_ents=None):
    """
    """
    
    doc_tups = list(df[["excerpt","id"]].itertuples(index=False, name=None))
    feat_list = []
    for doc, i in tqdm(nlp.pipe(doc_tups, as_tuples=True)):

        token_df = gen_token_df(doc, freq_df) 
        sent_df = gen_sent_df(doc)
        ent_df = gen_ent_df(doc)

        features = gen_features(token_df, sent_df, ent_df, 
                                tag_df=tag_df, 
                                all_ents=all_ents)
        features["id"] = i
        
        feat_list.append(features)
        
    feat_df = pd.DataFrame(feat_list).merge(df.drop(columns="excerpt"), 
                                            on= "id")
    return feat_df
