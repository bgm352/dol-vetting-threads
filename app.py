import os
import streamlit as st
import pandas as pd
import time
from datetime import datetime
import nltk
import random
import logging
from textblob import TextBlob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from apify_client import ApifyClient
except ModuleNotFoundError:
    st.error("Install `apify-client`: pip install apify-client")
    st.stop()
nltk.download("punkt")
try: import openai
except ImportError: openai = None
try: import google.generativeai as genai
except ImportError: genai = None

st.set_page_config("Meta Threads DOL/KOL Vetting Tool", layout="wide", page_icon="🩺")
st.title("🩺 Meta Threads DOL/KOL Vetting Tool")

ONCOLOGY_TERMS = ["oncology","cancer","monoclonal","checkpoint","immunotherapy"]
GI_TERMS = ["biliary tract","gastric","gea","gi","adenocarcinoma"]
RESEARCH_TERMS = ["biomarker","clinical trial","abstract","network","congress"]
BRAND_TERMS = ["ziihera","zanidatamab","brandA","pd-l1"]

def retry_with_backoff(func=None, *, max_retries=3, base_delay=2):
    def decorator(f):
        def wrapper(*args, **kwargs):
            attempt, last_exception = 0, None
            while attempt < max_retries:
                try: return f(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    attempt += 1
                    time.sleep(base_delay * (2**(attempt-1)) + random.uniform(0,1))
            raise last_exception
        return wrapper
    if func is None: return decorator
    else: return decorator(func)

@st.cache_resource
def get_apify_client(api_token: str): return ApifyClient(api_token)
@st.cache_resource
def get_openai_client(api_key: str):
    if not openai: raise RuntimeError("OpenAI SDK not installed")
    return openai.OpenAI(api_key=api_key)

@retry_with_backoff
def call_openai(prompt, client, model, temperature, max_tokens):
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    if max_tokens > 0: kwargs["max_tokens"] = max_tokens
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()

@retry_with_backoff
def call_gemini(prompt, api_key, model, temperature, max_tokens, reason_effort, reason_summary):
    if not genai: raise RuntimeError("Google Gemini SDK not installed")
    genai.configure(api_key=api_key)
    model_obj = genai.GenerativeModel(model)
    params = dict(prompt=prompt)
    if max_tokens > 0: params["max_tokens"] = max_tokens
    if temperature is not None: params["temperature"] = temperature
    if reason_effort and reason_effort != "None": params["reasoning_effort"] = reason_effort.lower()
    if reason_summary and reason_summary != "None": params["reasoning_summary"] = reason_summary.lower()
    resp = model_obj.generate_content(**params)
    return getattr(resp, "text", str(resp)).strip()

def get_llm_response(prompt, provider, openai_api_key=None, gemini_api_key=None, openai_model=None, openai_temperature=0.6, openai_max_tokens=512, gemini_model=None, gemini_temperature=0.6, gemini_max_tokens=512, gemini_reasoning_effort=None, gemini_reasoning_summary=None):
    try:
        if provider == "OpenAI GPT":
            client = get_openai_client(openai_api_key)
            return call_openai(prompt, client, openai_model, openai_temperature, openai_max_tokens)
        elif provider == "Google Gemini":
            return call_gemini(prompt, gemini_api_key, gemini_model, gemini_temperature, gemini_max_tokens, gemini_reasoning_effort, gemini_reasoning_summary)
        else:
            return f"Unknown LLM provider: {provider}"
    except Exception as e:
        logger.error(f"LLM call error: {e}")
        return f"{provider} call failed: {e}"

@st.cache_data(show_spinner=False, persist="disk")
def run_meta_threads_scraper_batched(api_key, query, target_total, batch_size):
    MAX_WAIT_SECONDS = 180
    client = get_apify_client(api_key)
    results, offset, failures = [], 0, 0
    pbar = st.progress(0.0, text="Scraping Meta Threads...")
    while len(results) < target_total and failures < 5:
        st.info(f"Batch {1 + offset//batch_size}: {len(results)} of {target_total}")
        try:
            run = client.actor("futurizerush/meta-threads-scraper").call(
                run_input={
                    "search": query,
                    "mode": "keyword",
                    "resultsPerPage": batch_size,
                    "maxPosts": batch_size
                }
            )
            run_id = run.get("id") or run.get("data", {}).get("id")
            batch_start = time.time()
            dataset_id = None
            while run_id and (time.time()-batch_start) < MAX_WAIT_SECONDS:
                run_status = client.actor("futurizerush/meta-threads-scraper").run(run_id)
                if run_status.get("status") == "SUCCEEDED":
                    dataset_id = run_status.get("defaultDatasetId")
                    break
                time.sleep(5)
            if not dataset_id:
                failures += 1; st.error("Batch timeout/failure, retrying..."); continue
            batch_items = list(client.dataset(dataset_id).iterate_items())
            if not batch_items:
                failures += 1; st.error("No posts in batch, retrying..."); continue
            for item in batch_items:
                if item not in results: results.append(item)
            offset += batch_size
            pbar.progress(min(1.0, len(results)/float(target_total)))
            if len(batch_items) < batch_size:
                break
        except Exception as e:
            st.error(f"Scraping failed: {e}"); failures += 1
    pbar.progress(1.0)
    return results[:target_total]

def classify_kol_dol(score):
    if score >= 8: return "KOL"
    if score >= 5: return "DOL"
    return "Not Suitable"
def classify_sentiment(score):
    if score > 0.15: return "Positive"
    if score < -0.15: return "Negative"
    return "Neutral"
def generate_rationale(text, transcript, author, score, sentiment, mode):
    all_text = f"{text or ''} {transcript or ''}".lower()
    tags = {
        "onco": any(t in all_text for t in ONCOLOGY_TERMS),
        "gi": any(t in all_text for t in GI_TERMS),
        "res": any(t in all_text for t in RESEARCH_TERMS),
        "brand": any(t in all_text for t in BRAND_TERMS),
    }
    name = author or "This creator"
    rationale = ""
    if "Doctor" in mode:
        if score >= 8: rationale = f"{name} is highly influential,"
        elif score >= 5: rationale = f"{name} has moderate relevance,"
        else: rationale = f"{name} does not actively discuss core campaign topics,"
        if tags["onco"]: rationale += " frequently engaging in oncology content"
        if tags["gi"]: rationale += ", particularly in GI-focused diseases"
        if tags["res"]: rationale += " and demonstrating strong research credibility"
        if tags["brand"]: rationale += ", mentioning monoclonal therapies or campaign drugs specifically"
        if transcript and "not found" not in transcript.lower():
            rationale += f'. Transcript: "{transcript[:90].strip()}..."'
        else:
            rationale += f". {transcript}"
        rationale += f" (Score: {score}/10)"
    else:
        rationale = f"{name} appears to express {sentiment.lower()} brand sentiment."
        if transcript and "not found" not in transcript.lower():
            rationale += f' Transcript: "{transcript[:90].strip()}..."'
        else:
            rationale += f". {transcript}"
    return rationale

def process_threads_posts(posts, fetch_time=None, last_fetch_time=None):
    results = []
    for post in posts:
        try:
            author = post.get("authorUsername") or post.get("authorName", "")
            text = post.get("text","")
            post_id = post.get("id","")
            url = post.get("threadUrl","")
            ts = pd.to_datetime(post.get("createdAt","")) if post.get("createdAt") else datetime.now()
            body = text or ""
            sentiment_score = TextBlob(body).sentiment.polarity
            sentiment = classify_sentiment(sentiment_score)
            dol_score = max(min(round((sentiment_score * 10) + 5), 10), 1)
            kol_dol_label = classify_kol_dol(dol_score)
            rationale = generate_rationale(text, text, author, dol_score, sentiment, run_mode)
            is_new = "🟢 New" if last_fetch_time is None or ts > last_fetch_time else "Old"
            results.append({
                "Author": author,
                "Text": text.strip(),
                "Likes": post.get("likesCount", 0),
                "Comments": post.get("commentsCount", 0),
                "Timestamp": ts,
                "Post URL": url,
                "DOL Score": dol_score,
                "Sentiment Score": sentiment_score,
                "KOL/DOL Status": kol_dol_label,
                "Brand Sentiment Label": sentiment,
                "LLM DOL Score Rationale": rationale,
                "Data Fetched At": fetch_time,
                "Is New": is_new,
            })
        except Exception as e:
            st.warning(f"Skipped post due to error: {e}")
    return pd.DataFrame(results)

def main():
    global run_mode
    apify_api_key = st.sidebar.text_input("Apify API Token", value=os.getenv("APIFY_API_TOKEN", ""), type="password")
    llm_provider = st.sidebar.selectbox("LLM Provider", ["OpenAI GPT", "Google Gemini"])
    openai_api_key = openai_model = None
    gemini_api_key = gemini_model = gemini_reasoning_effort = gemini_reasoning_summary = None
    temperature, max_tokens = 0.6, 512

    if llm_provider == "OpenAI GPT":
        openai_api_key = st.sidebar.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY",""), type="password")
        openai_model = st.sidebar.selectbox("OpenAI Model", ["gpt-4","gpt-3.5-turbo"])
        temperature = st.sidebar.slider("Temperature",0.0,2.0,0.6)
        max_tokens = st.sidebar.number_input("Max Completion Tokens",0,4096,512)
    elif llm_provider == "Google Gemini":
        gemini_api_key = st.sidebar.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY",""), type="password")
        gemini_model = st.sidebar.selectbox("Gemini Model", ["gemini-2.5-pro","gemini-2.5-flash"])
        temperature = st.sidebar.slider("Temperature",0.0,2.0,0.6)
        max_tokens = st.sidebar.number_input("Max Completion Tokens",0,4096,512)
        gemini_reasoning_effort = st.sidebar.selectbox("Reasoning Effort", ["None", "Low", "Medium", "High"])
        gemini_reasoning_summary = st.sidebar.selectbox("Reasoning Summary", ["None", "Concise", "Detailed", "Auto"])

    st.sidebar.header("Scrape Controls")
    query = st.sidebar.text_input("Threads Search Term", "doctor")
    target_total = st.sidebar.number_input("Total Threads Posts", min_value=10, value=200, step=10)
    batch_size = st.sidebar.number_input("Batch Size per Run", min_value=10, max_value=200, value=20)
    run_mode = st.sidebar.radio("Analysis Type", ["Doctor Vetting (DOL/KOL)", "Brand Vetting (Sentiment)"])

    if "last_fetch_time" not in st.session_state: st.session_state.last_fetch_time = None
    if "threads_df" not in st.session_state: st.session_state.threads_df = pd.DataFrame()
    if "llm_notes_text" not in st.session_state: st.session_state.llm_notes_text = ""
    if "llm_score_result" not in st.session_state: st.session_state.llm_score_result = ""
    if st.button("Go 🚀", use_container_width=True):
        if not apify_api_key: st.error("Apify API Token required."); return
        st.session_state.last_fetch_time = datetime.now()
        posts = run_meta_threads_scraper_batched(apify_api_key, query, target_total, batch_size)
        if not posts: st.warning("No Threads posts found."); return
        df = process_threads_posts(posts, fetch_time=st.session_state.last_fetch_time, last_fetch_time=None)
        st.session_state.threads_df = df
        st.success(f"Fetched and processed {len(df)} Threads posts.")
    df = st.session_state.threads_df
    if not df.empty:
        st.metric("Threads Posts", len(df))
        st.subheader("📋 Threads Analysis Results")
        threads_cols = ["Author","Text","Likes","Comments","DOL Score","Sentiment Score","Post URL","KOL/DOL Status","Brand Sentiment Label","LLM DOL Score Rationale","Timestamp","Data Fetched At","Is New"]
        display_option = st.radio("Choose display columns:", ["All columns","Only main info","Just DOL / Sentiment"])
        if display_option == "All columns": columns = threads_cols
        elif display_option == "Only main info":
            columns = ["Author","Text","Likes","Comments","DOL Score","Timestamp","Is New"]
        else:
            columns = ["Author","KOL/DOL Status","DOL Score","Sentiment Score","Brand Sentiment Label","Is New"]
        dol_min, dol_max = st.slider("Select DOL Score Range",1,10,(1,10))
        filtered_df = df[(df["DOL Score"] >= dol_min) & (df["DOL Score"] <= dol_max)]
        st.dataframe(filtered_df[columns], use_container_width=True)
        st.download_button(
            "Download Threads CSV",
            filtered_df[columns].to_csv(index=False),
            file_name=f"threads_analysis_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
        )
        if st.checkbox("Show Raw Threads Data"):
            st.subheader("Raw Threads Data")
            st.dataframe(df, use_container_width=True)
        st.subheader("📝 LLM Notes & Suitability Scoring")
        default_template = "Summary:\nRelevance:\nStrengths:\nWeaknesses:\nRed Flags:\nBrand Mentions:\nResearch Notes:\n"
        note_template = st.text_area("Customize LLM Notes Template", value=default_template, height=150)
        if st.button("Generate LLM Vetting Notes"):
            st.session_state.llm_score_result = ""
            with st.spinner("Calling LLM to generate notes..."):
                notes_text = get_llm_response(
                    note_template + "\nData:\n" + filtered_df.to_string(),
                    provider=llm_provider,
                    openai_api_key=openai_api_key if llm_provider == "OpenAI GPT" else None,
                    gemini_api_key=gemini_api_key if llm_provider == "Google Gemini" else None,
                    openai_model=openai_model if llm_provider == "OpenAI GPT" else None,
                    openai_temperature=temperature if llm_provider == "OpenAI GPT" else 0.6,
                    openai_max_tokens=max_tokens if llm_provider == "OpenAI GPT" else 512,
                    gemini_model=gemini_model if llm_provider == "Google Gemini" else None,
                    gemini_temperature=temperature if llm_provider == "Google Gemini" else 0.6,
                    gemini_max_tokens=max_tokens if llm_provider == "Google Gemini" else 512,
                    gemini_reasoning_effort=gemini_reasoning_effort if llm_provider == "Google Gemini" else None,
                    gemini_reasoning_summary=gemini_reasoning_summary if llm_provider == "Google Gemini" else None,
                )
                st.session_state.llm_notes_text = notes_text
        if st.session_state.llm_notes_text:
            st.markdown("#### LLM Vetting Notes")
            st.markdown(st.session_state.llm_notes_text)
            st.download_button(
                "Download LLM Vetting Notes",
                st.session_state.llm_notes_text,
                file_name="llm_vetting_notes.txt",
                mime="text/plain",
            )
            if st.button("Generate LLM Score & Rationale"):
                with st.spinner("Calling LLM for scoring..."):
                    score_result = get_llm_response(
                        st.session_state.llm_notes_text,
                        provider=llm_provider,
                        openai_api_key=openai_api_key if llm_provider == "OpenAI GPT" else None,
                        gemini_api_key=gemini_api_key if llm_provider == "Google Gemini" else None,
                        openai_model=openai_model if llm_provider == "OpenAI GPT" else None,
                        openai_temperature=temperature if llm_provider == "OpenAI GPT" else 0.6,
                        openai_max_tokens=max_tokens if llm_provider == "OpenAI GPT" else 512,
                        gemini_model=gemini_model if llm_provider == "Google Gemini" else None,
                        gemini_temperature=temperature if llm_provider == "Google Gemini" else 0.6,
                        gemini_max_tokens=max_tokens if llm_provider == "Google Gemini" else 512,
                        gemini_reasoning_effort=gemini_reasoning_effort if llm_provider == "Google Gemini" else None,
                        gemini_reasoning_summary=gemini_reasoning_summary if llm_provider == "Google Gemini" else None,
                    )
                    st.session_state.llm_score_result = score_result
        if st.session_state.llm_score_result:
            st.markdown("#### LLM DOL/KOL Score & Rationale")
            st.code(st.session_state.llm_score_result, language="yaml")

if __name__ == "__main__":
    main()
