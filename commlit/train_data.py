
# -------------------------- # -------------------------- #
# Training data assembly --- # -------------------------- #
# -------------------------- # -------------------------- #

import numpy as np
from commlit.up_scale import even_upsample

def gen_train_data(df, 
                   tgt_df=None,
                   up_sample=True,
                   rem_punct=False, 
                   rem_stop=False,
                   min_stop_len=0.3, 
                   drop_cols=["seq", "word", "alpha"],
                   tgt_noise_var="length",
                   tgt_noise_mult=2,
                   up_sample_param={"n_row": 100,
                                    "n_rep": 10}):
    """
    """
    
    # drop unwanted columns
    x = df.copy().drop(drop_cols, axis=1)
    
    # remove punctuation tokens
    if rem_punct:
        x = x[x["punct"]==False].drop(["punct"], axis=1)
       
    # remove (short) stop words    
    if rem_stop:
        x = x[(x["stop"]==False) | 
              (x["length"]>=min_stop_len)].drop(["stop"], axis=1)
    
    # drop columns with no variance
    x_std = x.std()
    x = x.drop(x_std[x_std == 0].index.to_list(), axis=1)
    
    # upsample even number of rows per id    
    if up_sample:
        x = x.groupby("id").apply(even_upsample,
                                  n_row=up_sample_param["n_row"],
                                  n_rep=up_sample_param["n_rep"]
                                  ).reset_index(drop=True)
        
    # create target variable
    if tgt_df is not None:
        y = x.groupby(["id", "grp_id"])[tgt_noise_var].mean().reset_index()
        y.loc[:, "diff_avg"] = y.groupby("id")[tgt_noise_var].transform("mean")
        y.loc[:, "diff_dev"] = 1 - y[tgt_noise_var]/y["diff_avg"]
        y = y.merge(tgt_df, on="id", how="inner")
        y = y["target"] + y["diff_dev"]*y["standard_error"]*tgt_noise_mult
        y = y.values
    else:
        y = None
        
    # convert x to matrix input
    x = list(x.groupby(["id", "grp_id"]))
    x = [grp[1].drop(["id", "grp_id"], axis=1).astype(float).values 
         for grp in x]
    x = np.stack(x)[:, :, :, np.newaxis]    
        
    return x, y
