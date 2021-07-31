
# -------------------------- # -------------------------- #
# Upscaling data size module # -------------------------- #
# -------------------------- # -------------------------- #

import numpy as np
import pandas as pd

def upscale_targets(df, n=10):
    """
    """
    
    og_cols = df.columns
    
    if n > 1:
        up_df = df.copy()[["id", "target", "standard_error"]]
        up_df.loc[:, "noise"] = up_df["standard_error"].apply(
            lambda x: x*np.random.randn(n)
        )
        up_df = up_df.explode("noise")
        up_df.loc[:, "target"] = (up_df["target"] + 
                                up_df["noise"]).astype(float)
        df = df.drop(columns=["target"]
                    ).merge(up_df[["id", "target"]], on="id")
    
    return df[og_cols]

def even_upsample(grp, n_row=100, n_rep=5):
    """
    """
    n_grp = grp.shape[0]
    grp_df = []
    
    for i in range(n_rep):
        if n_grp >= n_row:
            i_df = grp.sample(n_row)
        else:
            i_df = pd.concat([grp]*(n_row // n_grp) + 
                             [grp.sample(n_row % n_grp)], 
                             ignore_index=True)
        i_df.loc[:, "grp_id"] = i
        grp_df.append(i_df)
        
    return pd.concat(grp_df, ignore_index=True)
