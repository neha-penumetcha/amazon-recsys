import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

@st.cache_resource
def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet',   quiet=True)
    return True
setup_nltk()

@st.cache_resource
def load_model():
    with open('models/svd_model.pkl','rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv('outputs/clean_reviews_with_sentiment.csv')
    ps = pd.read_csv('outputs/product_sentiment.csv')
    return df, ps

model    = load_model()
df, ps   = load_data()
analyzer = SentimentIntensityAnalyzer()

def hybrid_recs(user_id, w_cf, w_nlp, n):
    all_prods = df['product_id'].unique()
    seen      = set(df[df['user_id']==user_id]['product_id'])
    lookup    = ps.set_index('product_id')[
        ['sentiment_scaled','avg_sentiment',
         'pct_positive','review_count']
    ].to_dict('index')
    rows = []
    for pid in all_prods:
        if pid in seen: continue
        cf  = model.predict(user_id, pid).est
        sl  = lookup.get(pid,{}).get('sentiment_scaled', 3.0)
        rows.append({
            'Product ID':      pid,
            'CF Score':        round(cf, 2),
            'Sentiment Score': round(sl, 2),
            'Hybrid Score':    round(w_cf*cf + w_nlp*sl, 2),
            '% Positive':      round(lookup.get(pid,{}).get('pct_positive',0)*100,1),
            'Review Count':    lookup.get(pid,{}).get('review_count',0)
        })
    return pd.DataFrame(rows).sort_values(
        'Hybrid Score', ascending=False).head(n).reset_index(drop=True)

st.set_page_config(page_title="Amazon Recommender", page_icon="🎸", layout="wide")
st.title("Amazon Product Recommendation System")
st.caption("Hybrid: SVD Collaborative Filtering + VADER NLP Sentiment")

with st.sidebar:
    st.header("Controls")
    user_id = st.selectbox("Select User", df['user_id'].unique()[:300])
    n       = st.slider("Recommendations", 5, 20, 10)
    st.markdown("---")
    st.subheader("Hybrid Weights")
    w_cf  = st.slider("CF weight", 0.0, 1.0, 0.7, 0.05)
    w_nlp = round(1 - w_cf, 2)
    st.write(f"NLP weight: **{w_nlp}**")
    st.markdown("---")
    st.caption(f"{df.shape[0]:,} reviews · {df['user_id'].nunique():,} users · {df['product_id'].nunique():,} products")

tab1,tab2,tab3,tab4 = st.tabs(["Recommendations","User Profile","Sentiment Explorer","Model Info"])

with tab1:
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Computing..."):
            recs = hybrid_recs(user_id, w_cf, w_nlp, n)
        st.success(f"Top {n} products for {user_id[:25]}...")
        st.dataframe(recs, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(recs.set_index('Product ID')['Hybrid Score'])
        with col2:
            fig, ax = plt.subplots(figsize=(5,3))
            sc = ax.scatter(recs['CF Score'], recs['Sentiment Score'],
                            c=recs['Hybrid Score'], cmap='viridis', s=60)
            plt.colorbar(sc, ax=ax, label='Hybrid Score')
            ax.set_xlabel('CF Score')
            ax.set_ylabel('Sentiment Score')
            ax.set_title('CF vs Sentiment')
            st.pyplot(fig)

with tab2:
    user_df = df[df['user_id']==user_id]
    c1,c2,c3 = st.columns(3)
    c1.metric("Reviews given",   len(user_df))
    c2.metric("Avg rating",      round(user_df['rating'].mean(),2))
    c3.metric("Avg VADER score", round(user_df['vader_compound'].mean(),3))
    st.dataframe(user_df[['product_id','rating','sentiment_label','review_text']], use_container_width=True)

with tab3:
    st.subheader("Analyse any review text")
    sample = st.text_area("Paste a review here:", placeholder="This guitar is amazing...")
    if sample:
        sc = analyzer.polarity_scores(sample)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Positive", sc['pos'])
        c2.metric("Negative", sc['neg'])
        c3.metric("Neutral",  sc['neu'])
        c4.metric("Compound", sc['compound'])
        label = ("Positive" if sc['compound']>=0.05 else "Negative" if sc['compound']<=-0.05 else "Neutral")
        st.info(f"Classification: **{label}**")

with tab4:
    st.markdown("""
**Collaborative Filtering** — SVD (scikit-surprise) · n_factors=100 · n_epochs=30

**NLP Sentiment** — VADER on raw review text · score: -1 → +1

**Hybrid Formula** — `hybrid = (CF_weight × predicted_rating) + (NLP_weight × sentiment_scaled)`

**Dataset** — Amazon Musical Instruments 5-core · 10,261 reviews
    """)
