
# csat_end_to_end.py (recreated)
# See in-notebook description for steps. Optional BERTopic block included (commented).

import pandas as pd, numpy as np, re
from pathlib import Path

SRC = Path("new_onepager.csv")
OUT = Path("."); OUT.mkdir(exist_ok=True, parents=True)
df = pd.read_csv(SRC)

COL_CSAT="CSAT score" if "CSAT score" in df.columns else None
COL_ASAT=next((c for c in df.columns if re.search(r"how would you rate the service you received from the consultant", c, re.I)), None)
COL_AGENT="OPERATOR" if "OPERATOR" in df.columns else ("Supervisor" if "Supervisor" in df.columns else None)
COL_SUP="Supervisor" if "Supervisor" in df.columns else None
COL_ID="Respondent Id" if "Respondent Id" in df.columns else None
COL_DATE=next((c for c in ["INTERACTION_DATE","Completed Date"] if c in df.columns), None)
COL_CALLTYPE="CALL_TYPE" if "CALL_TYPE" in df.columns else None
COL_SUBCAT="SUBCATEGORY" if "SUBCATEGORY" in df.columns else None
COL_DIV="PRIMARY_DIVISION" if "PRIMARY_DIVISION" in df.columns else None
COL_SUBPROD="Sub-Product" if "Sub-Product" in df.columns else None
COL_WORKTYPE="Work Type" if "Work Type" in df.columns else None
COL_WAVE="Wave" if "Wave" in df.columns else None
COL_FCR_Y="FCR Yes" if "FCR Yes" in df.columns else None
COL_FCR_N="FCR No" if "FCR No" in df.columns else None

TEXT_COLS=[c for c in [
 "What are your most important reasons for giving us that score?",
 "We’d love to know what the consultant did to earn such a rating?",
 "How could the consultant improve how they handled your enquiry?"
] if c in df.columns]

for c in [COL_CSAT, COL_ASAT, COL_FCR_Y, COL_FCR_N]:
    if c: df[c]=pd.to_numeric(df[c], errors="coerce")

if COL_DATE:
    df[COL_DATE]=pd.to_datetime(df[COL_DATE], errors="coerce")
    df["Week"]=df[COL_DATE].dt.isocalendar().week.astype("Int64")
    df["Month"]=df[COL_DATE].dt.month.astype("Int64")
else:
    df["Week"]=pd.NA; df["Month"]=pd.NA

if TEXT_COLS:
    df["TEXT_ALL"]=df[TEXT_COLS].astype(str).agg(" ".join, axis=1).str.replace(r"\s+"," ", regex=True).str.strip()
else:
    df["TEXT_ALL"]=""

def bucket_by_quantiles(series):
    s=pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return (lambda v: np.nan), None, None
    q33,q66=s.quantile([0.33,0.66])
    def b(v):
        if pd.isna(v): return np.nan
        if v<=q33: return "Low"
        if v<=q66: return "Medium"
        return "High"
    return b,q33,q66

if COL_CSAT:
    bcs,_,_=bucket_by_quantiles(df[COL_CSAT]); df["CSAT_Bucket"]=df[COL_CSAT].map(bcs)

if COL_ASAT:
    s=pd.to_numeric(df[COL_ASAT], errors="coerce")
    if s.min()>=0 and s.max()<=10:
        def bas(v):
            if pd.isna(v): return np.nan
            if v<=7: return "Low"
            if v<=9: return "Medium"
            return "High"
        df["ASAT_Bucket"]=s.map(bas)
    else:
        bas,_,_=bucket_by_quantiles(s); df["ASAT_Bucket"]=s.map(bas)

if COL_FCR_Y and COL_FCR_N:
    df["FCR_Total"]=df[COL_FCR_Y].fillna(0)+df[COL_FCR_N].fillna(0)
    df["FCR_Flag"]=np.where(df[COL_FCR_Y].fillna(0)>0,1,np.where(df[COL_FCR_N].fillna(0)>0,0,np.nan))
else:
    df["FCR_Total"]=pd.NA; df["FCR_Flag"]=pd.NA

if COL_AGENT:
    g=df.groupby(COL_AGENT, dropna=False)
    agent_summary=g.agg(
        cases=(COL_ID,"count") if COL_ID else (COL_AGENT,"size"),
        csat_avg=(COL_CSAT,"mean") if COL_CSAT else (COL_AGENT,"size"),
        asat_avg=(COL_ASAT,"mean") if COL_ASAT else (COL_AGENT,"size"),
    ).reset_index()
    if "CSAT_Bucket" in df: 
        agent_summary["csat_low_pct"]=g["CSAT_Bucket"].apply(lambda s:(s=="Low").mean()).values
        agent_summary["csat_high_pct"]=g["CSAT_Bucket"].apply(lambda s:(s=="High").mean()).values
    if "ASAT_Bucket" in df and COL_ASAT:
        agent_summary["asat_low_pct"]=g["ASAT_Bucket"].apply(lambda s:(s=="Low").mean()).values
        agent_summary["asat_high_pct"]=g["ASAT_Bucket"].apply(lambda s:(s=="High").mean()).values
    if COL_FCR_Y and COL_FCR_N:
        g2=g.agg(yes=(COL_FCR_Y,"sum"), no=(COL_FCR_N,"sum")); d=(g2["yes"].fillna(0)+g2["no"].fillna(0)).replace(0,np.nan)
        agent_summary["fcr_rate"]=g2["yes"]/d
else:
    agent_summary=pd.DataFrame()

def simple_clean(s):
    s=s.astype(str).str.lower().str.replace(r"[^a-z0-9\s]"," ", regex=True).str.replace(r"\s+"," ", regex=True).str.strip()
    return s
df["_clean"]=simple_clean(df["TEXT_ALL"])
stop=set("the a an is are am were was be been being i you he she they them we us our your his her its of in on at to for from by with as it and or if but not no yes do did done have has had this that these those then so very just into out about over under more most less few lot much many me my mine yours theirs ours him her hims hers".split())

def top_terms(mask, n=1, topn=25):
    s=df.loc[mask, "_clean"]; counts={}
    for line in s:
        toks=[t for t in line.split() if t not in stop and len(t)>2]
        grams=toks if n==1 else [" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)]
        for g in grams: counts[g]=counts.get(g,0)+1
    ser=pd.Series(counts).sort_values(ascending=False).head(topn) if counts else pd.Series(dtype=int)
    return ser.reset_index().rename(columns={"index":"term",0:"count"})

low_mask=df["CSAT_Bucket"].eq("Low") if "CSAT_Bucket" in df else pd.Series(False, index=df.index)
high_mask=df["CSAT_Bucket"].eq("High") if "CSAT_Bucket" in df else pd.Series(False, index=df.index)
low_uni=top_terms(low_mask,1,40); low_bi=top_terms(low_mask,2,40)
high_uni=top_terms(high_mask,1,40); high_bi=top_terms(high_mask,2,40)

topic_summary=pd.DataFrame()
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    docs=df["_clean"].fillna(""); mask_nonempty=docs.str.len()>0; docs_nz=docs[mask_nonempty]
    if docs_nz.shape[0]>=40:
        tfidf=TfidfVectorizer(max_features=6000, ngram_range=(1,2), min_df=5, stop_words="english")
        X=tfidf.fit_transform(docs_nz)
        n_topics=min(10, max(4, X.shape[0]//120))
        nmf=NMF(n_components=n_topics, random_state=42, init="nndsvd")
        W=nmf.fit_transform(X); H=nmf.components_; vocab=np.array(tfidf.get_feature_names_out())
        dom=W.argmax(axis=1); df["topic_id"]=np.nan; df.loc[docs_nz.index,"topic_id"]=dom
        rows=[]; 
        for k in range(nmf.n_components):
            top_idx=np.argsort(H[k])[::-1][:10]
            rows.append({"topic_id":int(k),"top_terms":", ".join(vocab[top_idx])})
        topic_kw=pd.DataFrame(rows)
        metrics=[c for c in [COL_CSAT,COL_ASAT] if c]
        topic_summary=df.loc[mask_nonempty].groupby("topic_id")[metrics].mean().reset_index()
        topic_summary["count"]=df.loc[mask_nonempty].groupby("topic_id")[metrics[0]].size().values
        if "CSAT_Bucket" in df:
            topic_summary["low_csat_pct"]=df.loc[mask_nonempty].groupby("topic_id")["CSAT_Bucket"].apply(lambda s:(s=="Low").mean()).values
        topic_summary=topic_summary.merge(topic_kw, on="topic_id", how="left").sort_values("count", ascending=False)
except Exception as e:
    topic_summary=pd.DataFrame({"note":[f"NMF topic modeling skipped: {e}"]})

facts_cols=[c for c in [
 COL_ID, COL_DATE, "Week", "Month", COL_AGENT, COL_SUP,
 COL_CALLTYPE, COL_SUBCAT, COL_DIV, COL_SUBPROD, COL_WORKTYPE, COL_WAVE,
 COL_CSAT, "CSAT_Bucket", COL_ASAT, "ASAT_Bucket",
 COL_FCR_Y, COL_FCR_N, "FCR_Flag", "topic_id", "TEXT_ALL"
] if (c in df.columns) or (c in ["Week","Month","CSAT_Bucket","ASAT_Bucket","FCR_Flag","topic_id","TEXT_ALL"])]
facts=df[facts_cols].copy(); facts.to_csv(OUT/"facts_processed.csv", index=False)
agent_summary.to_csv(OUT/"agent_summary.csv", index=False)
topic_summary.to_csv(OUT/"topic_summary.csv", index=False)
terms=pd.DataFrame({
 "segment": (["Low CSAT"]*len(low_uni))+ (["Low CSAT"]*len(low_bi))+ (["High CSAT"]*len(high_uni))+ (["High CSAT"]*len(high_bi)),
 "ngram": (["unigram"]*len(low_uni))+ (["bigram"]*len(low_bi))+ (["unigram"]*len(high_uni))+ (["bigram"]*len(high_bi)),
 "term": pd.concat([low_uni["term"],low_bi["term"],high_uni["term"],high_bi["term"]], ignore_index=True),
 "count": pd.concat([low_uni["count"],low_bi["count"],high_uni["count"],high_bi["count"]], ignore_index=True),
}); terms.to_csv(OUT/"term_drivers.csv", index=False)
data_dict=pd.DataFrame({"column":df.columns,"dtype":[str(t) for t in df.dtypes],"non_null":df.notna().sum().values,"nulls":df.isna().sum().values}).sort_values("column")
data_dict.to_csv(OUT/"data_dictionary.csv", index=False)
print("Done.")
# Optional advanced BERTopic (uncomment if installed):
# from bertopic import BERTopic
# from sentence_transformers import SentenceTransformer
# emb = SentenceTransformer("all-MiniLM-L6-v2").encode(df["_clean"].fillna("").tolist(), show_progress_bar=True)
# topic_model = BERTopic(min_topic_size=25, calculate_probabilities=False, verbose=True)
# topics, _ = topic_model.fit_transform(df["_clean"].fillna("").tolist(), emb)
# df["topic_id_bertopic"] = topics
# df.to_csv(OUT/"facts_processed.csv", index=False)  # overwrite with BERTopic topics
