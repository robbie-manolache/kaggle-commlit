
# -------------------------- # -------------------------- #
# Upscaling data size module # -------------------------- #
# -------------------------- # -------------------------- #

import numpy as np

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
