
# ---------------------------- # ---------------------------- #
# Word vector trasnform module # ---------------------------- #
# ---------------------------- # ---------------------------- #

import pandas as pd
from commlit.feat_eng import gen_token_df

def __quantile_means__(x, nq=3):
    """
    """
    if nq > 1:
        q = pd.qcut(x, nq)
        return x.groupby(q).mean().values
    elif nq == 1:
        return x.mean()
    else:
        print("nq must be 1 or greater!")
        return

def gen_word_vec_df(doc, nq=3):
    """
    """
    
    # gen token metadata df
    token_df = gen_token_df(doc)
    
    # compile word vectors
    vecs = [token.vector for token in doc]
    vec_df = pd.DataFrame(vecs, columns=["vec_" + str(i) for 
                                         i in range(len(vecs[0]))])
    
    # concat token metadata and word vecs
    df = pd.concat([token_df, vec_df], axis=1)
    df = df[(df["alpha"] == True) & 
            (df["stop"] == False) & 
            (df["punct"] == False)]
    df = df.drop(columns=[c for c in df.columns 
                          if not c.startswith("vec_")])
    
    # compute quantile means
    q_df = df.apply(__quantile_means__, axis=0, nq=nq)
    q_df.loc[:, "q_id"] = ["q_" + str(q+1) for q in range(nq)]
    q_df = q_df.melt(id_vars="q_id")
    q_df.loc[:, "variable"] = q_df["variable"] + "_" + q_df["q_id"]
    
    return dict(zip(q_df["variable"], q_df["value"]))
    
