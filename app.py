import os
import streamlit as st
import pandas as pd
from datetime import datetime
import nltk
import random
import time
import logging
from textblob import TextBlob
from typing import List, Dict, Any
import altair as alt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download("punkt", quiet=True)

try:
    from apify_client import ApifyClient
except ModuleNotFoundError:
    st.error("Install `apify-client`: pip install apify-client")
    st.stop()

try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

ONCOLOGY_TERMS = ["oncology", "cancer", "monoclonal", "checkpoint", "immunotherapy"]
GI_TERMS = ["biliary tract", "gastric", "gea", "gi", "adenocarcinoma"]
RESEARCH_TERMS = ["biomarker", "clinical trial", "abstract", "network", "congress"]
BRAND_TERMS = ["ziihera", "zanidatamab", "brandA", "pd-l1"]

def keyword_hits(text: str) -> Dict[str, bool]:
    t = (text or "").lower()
    return {
        "onco": any(k in t for k in ONCOLOGY_TERMS),
        "gi": any(k in t for k in GI_TERMS),
        "res": any(k in t for k in RESEARCH_TERMS),
        "brand": any(k in t for k in BRAND_TERMS),
    }

def classify_sentiment(score: float) -> str:
    if score > 0.15:
        return "Positive"
    elif score < -0.15:
        return "Negative"
    return "Neutral"

def classify_kol_dol(score: float) -> str:
    if score >= 8:
        return "KOL"
    elif score >= 5:
        return "DOL"
    return "Not Suitable"

def retry_with_backoff(func=None, *, max_retries=3, base_delay=2):
    def decorator(f):
        def wrapper(*args, **kwargs):
            attempt, last_exception = 0, None
            while attempt < max_retries:
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    attempt += 1
                    delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                    logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator(func) if func else decorator

@st.cache_resource
def get_apify_client(api_token: str) -> ApifyClient:
    return ApifyClient(api_token)

@st.cache_resource
def get_openai_client(api_key: str):
    if not openai:
        raise RuntimeError("OpenAI SDK not installed.")
    return openai.OpenAI(api_key=api_key)

@retry_with_backoff
def call_openai(prompt: str, api_key: str, model: str, temperature: float, max_tokens: int) -> str:
    api_key = api_key.strip() if api_key else ""
    if not api_key:
        raise ValueError("OpenAI API key is empty or invalid.")
    client = get_openai_client(api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

@retry_with_backoff
def call_gemini(prompt: str, api_key: str, model: str, temperature: float, max_tokens: int) -> str:
    if not genai:
        raise RuntimeError("Gemini SDK not installed.")
    api_key = api_key.strip() if api_key else ""
    if not api_key:
        raise ValueError("Gemini API key is empty or invalid.")
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model)
    resp = m.generate_content(
        prompt=prompt,
        temperature=temperature,
        max_output_tokens=max_tokens if max_tokens > 0 else None,
    )
    # Safely extract text response
    if hasattr(resp, "text") and resp.text:
        return resp.text.strip()
    elif hasattr(resp, "candidates") and resp.candidates:
        return resp.candidates[0].content.parts[0].text.strip()
    return str(resp).strip()

@st.cache_data(show_spinner=False)
def run_threads_scraper(api_key: str, usernames: List[str], max_posts: int = 50) -> List[Dict[str, Any]]:
    client = get_apify_client(api_key)
    run_input = {
        "urls": [f"https://www.threads.net/@{u}" for u in usernames],
        "maxPosts": max_posts,
        "includeComments": False,
    }
    run = client.actor("red.cars/threads-scraper").call(run_input=run_input)
    dsid = run.get("defaultDatasetId")
    if not dsid:
        return []
    return list(client.dataset(dsid).iterate_items())

def process_profiles_and_posts(data: List[Dict[str, Any]]):
    profiles, posts_list = {}, []
    for item in data:
        if item.get("type") == "profile":
            u = item.get("username", "")
            profiles[u] = {
                "Username": u,
                "Full Name": item.get("displayName", ""),
                "Bio": item.get("bio", ""),
                "Follower Count": item.get("followers", 0),
                "Profile URL": f"https://www.threads.net/@{u}",
                "Is Verified": item.get("isVerified", False),
                "Relevant Posts": 0,
                "Total Engagement (Relevant)": 0,
                "Avg Engagement (Relevant)": 0,
                "Keyword Hits Bio": keyword_hits(item.get("bio", "")),
            }
    for item in data:
        if item.get("type") == "post":
            uname = item.get("ownerUsername", "")
            txt = item.get("caption", "")
            s_score = TextBlob(txt).sentiment.polarity if txt else 0.0
            s_label = classify_sentiment(s_score)
            tags = keyword_hits(txt)
            engagement = (item.get("likesCount", 0) or 0) + (item.get("commentsCount", 0) or 0)
            posts_list.append({
                "Username": uname,
                "Post Text": txt,
                "Likes": item.get("likesCount", 0),
                "Comments": item.get("commentsCount", 0),
                "Engagement": engagement,
                "Sentiment": s_label,
                "Keywords": ", ".join([k for k, v in tags.items() if v]),
            })
            if any(tags.values()) and uname in profiles:
                profiles[uname]["Relevant Posts"] += 1
                profiles[uname]["Total Engagement (Relevant)"] += engagement
    profile_rows = []
    for uname, p in profiles.items():
        rel = p["Relevant Posts"]
        avg_eng = (p["Total Engagement (Relevant)"] / rel) if rel else 0
        p["Avg Engagement (Relevant)"] = round(avg_eng, 1)
        bio_score = TextBlob(p["Bio"]).sentiment.polarity if p["Bio"] else 0.0
        score = (bio_score * 10) + min(rel, 5) + min(avg_eng / 50, 3)
        score = max(min(round(score), 10), 1)
        rationale = [
            f"Bio sentiment scaled: {round(bio_score * 10, 1)}",
            f"Relevant posts(capped5): {min(rel, 5)}",
            f"Avg engagement scaled(max3): {round(min(avg_eng / 50, 3), 2)}",
        ]
        tags_bio = p.get("Keyword Hits Bio", {})
        extra = []
        if tags_bio.get("onco"):
            extra.append("Oncology")
        if tags_bio.get("gi"):
            extra.append("GI")
        if tags_bio.get("res"):
            extra.append("Research")
        if tags_bio.get("brand"):
            extra.append("Brand/Drugs")
        if extra:
            rationale += ["Bio mentions: " + ", ".join(extra)]
        p["DOL Score"] = score
        p["Sentiment"] = classify_sentiment(bio_score)
        p["KOL/DOL"] = classify_kol_dol(score)
        p["Rationale"] = "; ".join(rationale)
        profile_rows.append(p)
    return pd.DataFrame(profile_rows), pd.DataFrame(posts_list)

# Sidebar Inputs

st.sidebar.header("Scraper Settings")

api_key = st.sidebar.text_input("Apify API Token", type="password", value=(st.secrets.get("APIFY_API_TOKEN") if "APIFY_API_TOKEN" in st.secrets else os.getenv("APIFY_API_TOKEN", "")))
if api_key:
    api_key = api_key.strip()

usernames_str = st.sidebar.text_area("Threads Usernames (comma separated)", "elonmusk")
usernames = [u.strip().lstrip("@") for u in usernames_str.split(",") if u.strip()]

max_posts = st.sidebar.slider("Max posts per profile", 1, 200, 50)

st.sidebar.header("Top Target Thresholds")

high_score_threshold = st.sidebar.slider("High Score Threshold", 5, 10, 8)
high_engagement_threshold = st.sidebar.number_input("High Engagement Threshold", 0, 500, 50)

st.sidebar.header("LLM Vetting Settings")

default_prompt = (
    "Summary:\nRelevance:\nStrengths:\nWeaknesses:\nRed Flags:\nBrand Mentions:\nResearch Notes:\n"
)
prompt_template = st.sidebar.text_area("LLM Prompt Template", value=default_prompt, height=180)

llm_provider = st.sidebar.selectbox("Provider", ["OpenAI GPT", "Google Gemini"])
temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.6)
max_tokens = st.sidebar.number_input("Max Tokens", 0, 4096, 512)

openai_api_key = None
gemini_api_key = None
openai_model = None
gemini_model = None

if llm_provider == "OpenAI GPT":
    openai_api_key = (
        st.sidebar.text_input("OpenAI API Key", type="password", value=(st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY", "")))
    )
    if openai_api_key:
        openai_api_key = openai_api_key.strip()
    openai_model = st.sidebar.selectbox("OpenAI Model", ["gpt-4", "gpt-3.5-turbo"])
else:
    gemini_api_key = (
        st.sidebar.text_input("Gemini API Key", type="password", value=(st.secrets.get("GEMINI_API_KEY") if "GEMINI_API_KEY" in st.secrets else os.getenv("GEMINI_API_KEY", "")))
    )
    if gemini_api_key:
        gemini_api_key = gemini_api_key.strip()
    gemini_model = st.sidebar.selectbox("Gemini Model", ["gemini-2.5-pro", "gemini-2.5-flash"])

if "last_scrape_time" not in st.session_state:
    st.session_state.last_scrape_time = None

st.sidebar.markdown(f"**Last Scrape:** {st.session_state.last_scrape_time or 'Never'}")

# Button to Start Scraping and Analysis

if st.sidebar.button("Scrape & Analyze 🚀"):
    if not api_key or not usernames:
        st.error("Missing API key or usernames.")
    else:
        with st.spinner("Scraping..."):
            data = run_threads_scraper(api_key, usernames, max_posts=max_posts)
        if not data:
            st.warning("No data.")
        else:
            profiles_df, posts_df = process_profiles_and_posts(data)
            st.session_state.profiles_df = profiles_df
            st.session_state.posts_df = posts_df
            st.session_state.last_scrape_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Profiles Overview & Visualization

if "profiles_df" in st.session_state and not st.session_state.profiles_df.empty:
    profiles_df = st.session_state.profiles_df
    st.subheader("Profiles Overview & Charts")
    df_plot = profiles_df.dropna(subset=["DOL Score", "KOL/DOL"]).copy()
    df_plot["DOL Score"] = df_plot["DOL Score"].astype(str)

    c1, c2 = st.columns(2)
    with c1:
        score_chart = (
            alt.Chart(df_plot)
            .mark_bar()
            .encode(
                x=alt.X("DOL Score:N", title="DOL Score"),
                y=alt.Y("count()", title="Count"),
                color=alt.Color("KOL/DOL:N", title="KOL/DOL Status"),
            )
            .properties(title="KOL/DOL Score Distribution")
        )
        st.altair_chart(score_chart, use_container_width=True)
    with c2:
        sentiment_data = profiles_df.groupby("Sentiment").size().reset_index(name="count")
        pie = (
            alt.Chart(sentiment_data)
            .mark_arc()
            .encode(
                theta="count:Q",
                color="Sentiment:N",
                tooltip=["Sentiment", "count"],
            )
            .properties(title="Bio Sentiment Breakdown")
        )
        st.altair_chart(pie, use_container_width=True)

    st.markdown("#### Engagement vs DOL Score (Top Targets Highlighted)")
    profiles_df["Top Target"] = (
        (profiles_df["DOL Score"] >= high_score_threshold)
        & (profiles_df["Avg Engagement (Relevant)"] >= high_engagement_threshold)
    )
    scatter = (
        alt.Chart(profiles_df)
        .mark_circle(size=80)
        .encode(
            x=alt.X("Avg Engagement (Relevant):Q", title="Average Engagement per Relevant Post"),
            y=alt.Y("DOL Score:Q", title="DOL Score"),
            color=alt.Color("Top Target:N", title="Top Target"),
            tooltip=[
                "Username",
                "DOL Score",
                "Avg Engagement (Relevant)",
                "Relevant Posts",
                "Follower Count",
            ],
        )
        .interactive()
    )
    st.altair_chart(scatter, use_container_width=True)

    top_targets_df = profiles_df[profiles_df["Top Target"]]
    st.markdown(f"**Top Targets (Score ≥ {high_score_threshold}, Engagement ≥ {high_engagement_threshold})**")
    st.dataframe(top_targets_df, use_container_width=True)
    st.download_button(
        "Download Top Targets CSV", top_targets_df.to_csv(index=False), "top_targets.csv", "text/csv"
    )

    f_status = st.multiselect("Filter by KOL/DOL", ["KOL", "DOL", "Not Suitable"], ["KOL", "DOL", "Not Suitable"])
    f_sent = st.multiselect("Filter by Sentiment", ["Positive", "Neutral", "Negative"], ["Positive", "Neutral", "Negative"])
    f_df = profiles_df[(profiles_df["KOL/DOL"].isin(f_status)) & (profiles_df["Sentiment"].isin(f_sent))]
    st.dataframe(f_df, use_container_width=True)
    st.download_button("Download Profiles CSV", f_df.to_csv(index=False), "profiles.csv", "text/csv")

    if st.button("Generate Vetting Notes with LLM"):
        data_text = f_df.to_string(index=False)
        prompt = prompt_template + "\n\nProfiles Data:\n" + data_text
        with st.spinner("Calling LLM..."):
            try:
                if llm_provider == "OpenAI GPT":
                    notes = call_openai(prompt, openai_api_key, openai_model, temperature, max_tokens)
                else:
                    notes = call_gemini(prompt, gemini_api_key, gemini_model, temperature, max_tokens)
                st.session_state.llm_notes = notes
            except Exception as e:
                st.error(f"LLM call failed: {e}")

if "llm_notes" in st.session_state:
    st.subheader("LLM Vetting Notes")
    st.markdown(st.session_state.llm_notes)
    st.download_button("Download Vetting Notes", st.session_state.llm_notes, "vetting_notes.txt", "text/plain")

# Posts Overview & Charts

if "posts_df" in st.session_state and not st.session_state.posts_df.empty:
    posts_df = st.session_state.posts_df
    st.subheader("Posts Overview & Chart")

    all_keywords = []
    for kws in posts_df["Keywords"]:
        if kws:
            all_keywords.extend([k.strip() for k in kws.split(",") if k.strip()])
    kw_freq = pd.Series(all_keywords).value_counts().reset_index()
    kw_freq.columns = ["Keyword", "Count"]

    if not kw_freq.empty:
        kw_chart = alt.Chart(kw_freq).mark_bar().encode(
            x="Keyword:N", y="Count:Q", tooltip=["Keyword", "Count"]
        )
        st.altair_chart(kw_chart, use_container_width=True)

    kw_filter = st.multiselect("Filter by Keywords", sorted(kw_freq["Keyword"]), sorted(kw_freq["Keyword"]))
    f_sent_posts = st.multiselect("Filter by Post Sentiment", ["Positive", "Neutral", "Negative"], ["Positive", "Neutral", "Negative"])

    fp_df = posts_df[
        (posts_df["Sentiment"].isin(f_sent_posts)) & (posts_df["Keywords"].apply(lambda x: any(k in x for k in kw_filter)))
    ]

    st.dataframe(fp_df, use_container_width=True)
    st.download_button("Download Posts CSV", fp_df.to_csv(index=False), "posts.csv", "text/csv")

