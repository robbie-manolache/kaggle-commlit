
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
    
def gen_word_vec_feat(doc, token_df=None, method="qval_fp",
                      q=[0.2, 0.8], nq=3):
    """
    Methods:
        "qval_fp": quantile values (passed to q), fraction positive
        "qmeans": means of nq quantiles
    """
    
    # gen token metadata df
    if token_df is None:
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
    
    # compute quantile means if selected
    if method == "qmeans":
        q_df = df.apply(__quantile_means__, axis=0, nq=nq)
        q_df.loc[:, "q_id"] = ["q_" + str(q+1) for q in range(nq)]
        q_df = q_df.melt(id_vars="q_id")
        q_df.loc[:, "variable"] = q_df["variable"] + "_" + q_df["q_id"]
        
    # compute quantile values and frac pos if selected
    if method == "qval_fp":
        q_df = df.quantile(q).transpose().join((df > 0).mean().rename("v")
                                               ).reset_index()
        q_df.columns = ["idx"] + ["q"+str(int(q_val*100)) 
                                  for q_val in q] + ["frac_pos"]
        q_df = q_df.melt(id_vars="idx")
        q_df.loc[:, "variable"] = q_df["idx"] + "_" + q_df["variable"]
        
    return dict(zip(q_df["variable"], q_df["value"]))
    
