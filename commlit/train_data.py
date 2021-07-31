
# -------------------------- # -------------------------- #
# Training data assembly --- # -------------------------- #
# -------------------------- # -------------------------- #

import re
import numpy as np
from commlit.up_scale import even_upsample

def gen_train_data(df, 
                   tgt_df=None,
                   sent_df=None,
                   gen_agg_feat=True,
                   up_sample=True,
                   rem_punct=False, 
                   rem_stop=False,
                   min_stop_len=0.3, 
                   x_cols=None,
                   drop_cols=["seq", "word", "alpha"],
                   agg_excl=["seq", "word"],
                   agg_excl_vec=True,
                   quant_cols=["length", "comm_score"],
                   quantiles=np.arange(0.025, 1, 0.025),
                   tgt_noise_var="length",
                   tgt_noise_mult=2,
                   sent_norm={"sent_length": 50, 
                              "noun_chunks": 10},
                   up_sample_param={"n_row": 125,
                                    "n_rep": 5}):
    """
    """
    
    # take copy
    x = df.copy()
    
    # aggregate selected features
    if gen_agg_feat:
        if agg_excl_vec:
            agg_excl += [f for f in x.columns if 
                         bool(re.match("v[0-9]+$", f))]
        agg_cols = [f for f in x.columns if f not in agg_excl]
        m = x[agg_cols].groupby("id").mean().reset_index()
        
        # add sentence features if available
        if sent_df is not None:
            s = sent_df.groupby("id").mean().reset_index()
            for k, v in sent_norm.items():
                s.loc[:, k] = s[k] / v
            m = m.merge(s, on="id")
    else:
        m = None
    
    # quantile features
    q = {}
    if quant_cols is not None:
        for qc in quant_cols:
            q_df = x[x["alpha"]==True].groupby("id")[qc]
            q_df = q_df.quantile(quantiles).reset_index()
            q_df = q_df.pivot("id", "level_1", qc)
            q_df.columns = ["q" + str(np.round(i, 5)) for i in quantiles]
            q[qc] = q_df.reset_index()  
    
    # drop unwanted columns
    x = x.drop(drop_cols, axis=1)
    
    # remove punctuation tokens
    if rem_punct:
        x = x[x["punct"]==False].drop(["punct"], axis=1)
       
    # remove (short) stop words    
    if rem_stop:
        x = x[(x["stop"]==False) | 
              (x["length"]>=min_stop_len)].drop(["stop"], axis=1)
    
    # drop columns with no variance, unless x_cols pre-specified
    if x_cols is None:
        x_std = x.std()
        x = x.drop(x_std[x_std == 0].index.to_list(), axis=1)
        x_cols = x.columns.to_list()
    else:
        x = x[x_cols]
    
    # upsample even number of rows per id    
    if up_sample:
        x = x.groupby("id").apply(even_upsample,
                                  n_row=up_sample_param["n_row"],
                                  n_rep=up_sample_param["n_rep"]
                                  ).reset_index(drop=True)
    
    # get the frame
    frame = x[["id", "grp_id"]].drop_duplicates()
    
    # update aggregate features
    if gen_agg_feat:
        m = frame.merge(m, on="id")
    if quant_cols is not None:
        for qc in quant_cols:
            q[qc] = frame.merge(q[qc], on="id")
    
        
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
        
    return x, y, m, q, frame, x_cols
