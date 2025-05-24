

import streamlit as st
import pandas as pd
import joblib
from scraper import scrape_karkidi_jobs
from model_training import preprocess_and_cluster

st.set_page_config(page_title="Job Classification App", layout="wide")

st.title("🔍 Job Posting Classification App")
st.markdown("Scrape jobs from Karkidi, classify them using ML, and find best-fit jobs for your skills!")

# Step 1: Scrape
if st.button("Scrape Jobs"):
    with st.spinner("Scraping jobs..."):
        df = scrape_karkidi_jobs(pages=2)
        st.session_state.df = df
        st.success("Scraping completed!")

# Step 2: Show & Cluster
if 'df' in st.session_state:
    df = st.session_state.df
    st.dataframe(df.head())

    if st.button("Preprocess & Cluster"):
        with st.spinner("Clustering jobs..."):
            clustered_df = preprocess_and_cluster(df)
            st.session_state.clustered_df = clustered_df
            st.success("Clustering done!")

# Step 3: Skill Match
if 'clustered_df' in st.session_state:
    st.subheader("🔎 Find Jobs Based on Your Skills")
    user_skills = st.text_input("Enter your skills (comma-separated)", "Python, SQL, Machine Learning")
    if st.button("Recommend Jobs"):
        vectorizer = joblib.load('vectorizer.pkl')
        kmeans = joblib.load('model.pkl')
        X_user = vectorizer.transform([user_skills])
        cluster_label = kmeans.predict(X_user)[0]
        matched_jobs = st.session_state.clustered_df[st.session_state.clustered_df['Cluster'] == cluster_label]
        st.write(f"Jobs matching your skill cluster ({cluster_label}):")
        st.dataframe(matched_jobs)

        csv = matched_jobs.to_csv(index=False).encode('utf-8')
        st.download_button("Download as CSV", csv, "matched_jobs.csv", "text/csv")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, BeautifulSoup, and Scikit-Learn")