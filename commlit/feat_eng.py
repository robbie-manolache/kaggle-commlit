
# -------------------------- # -------------------------- #
# Feature engineering module # -------------------------- #
# -------------------------- # -------------------------- #

import pandas as pd

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
        "type": ent.label_
    } for ent in doc.ents]
    
    return pd.DataFrame(ent_list)

