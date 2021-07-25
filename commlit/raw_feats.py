
# -------------------------- # -------------------------- #
# Raw feature extraction --- # -------------------------- #
# -------------------------- # -------------------------- #

import pandas as pd

def gen_raw_word_features(doc, tag_df=None, freq_df=None,
                          len_norm=20, vec_norm=10, len_vec=96,
                          freq_norm=None, unknown_comm_score=0.5):
    """
    """
    
    # get baseline word features
    token_list = [{
        "seq": i,
        "word": token.text.lower(),
        "length": len(token)/len_norm,
        "tag": token.tag_,
        "alpha": token.is_alpha,
        "stop": token.is_stop,
        "punct": token.is_punct,
        "vec": token.vector    
    } for i, token in enumerate(doc)]
    token_df = pd.DataFrame(token_list)
    
    # create word frequency feature
    if freq_df is not None:
        if freq_norm is None:
            freq_norm = freq_df["count"].min()
        token_df = token_df.merge(freq_df, on="word", how="left")
        token_df.loc[:, "comm_score"] = (freq_norm / token_df["count"])
        alpha_nonstop_miss = ((token_df["alpha"]==True) & 
                              (token_df["stop"]==False) & 
                              (token_df["comm_score"].isna()))
        token_df.loc[alpha_nonstop_miss, "comm_score"] = unknown_comm_score
        token_df.loc[:, "comm_score"] = token_df["comm_score"].fillna(0)
        token_df = token_df.drop(["count"], axis=1).sort_values("seq")
    
    # expand vector embedding lists
    vec_cols = ["v"+str(i) for i in range(len_vec)]
    token_df[vec_cols] = pd.DataFrame(token_df["vec"].tolist(), 
                                      index=token_df.index)/vec_norm
    token_df = token_df.drop(["vec"], axis=1)
    
    # generate tag dummies if provided
    if tag_df is not None:
        token_df = token_df.merge(tag_df.astype({"tag_adj": "category"}), 
                                  on="tag").drop(["tag"], axis=1)
        tag_dums = pd.get_dummies(token_df["tag_adj"], 
                                  columns=tag_df["tag_adj"])
        token_df = pd.concat([token_df, tag_dums], axis=1)
        token_df = token_df.drop(["tag_adj"], axis=1).sort_values("seq")
    else:
        token_df = token_df.drop(["tag"], axis=1)
        
    return token_df
    