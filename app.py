import os
import streamlit as st
import pandas as pd
import time
from datetime import datetime
import nltk
import random
import logging
from textblob import TextBlob
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

nltk.download("punkt", quiet=True)

# --- Streamlit page setup ---
st.set_page_config(
    "Meta Threads Profile DOL/KOL Vetting Tool",
    layout="wide",
    page_icon="🩺"
)
st.title("🩺 Meta Threads Profile DOL/KOL Vetting Tool")

# --- Keyword lists for rationale generation ---
ONCOLOGY_TERMS = ["oncology", "cancer", "monoclonal", "checkpoint", "immunotherapy"]
GI_TERMS = ["biliary tract", "gastric", "gea", "gi", "adenocarcinoma"]
RESEARCH_TERMS = ["biomarker", "clinical trial", "abstract", "network", "congress"]
BRAND_TERMS = ["ziihera", "zanidatamab", "brandA", "pd-l1"]

# --- Retry decorator ---
def retry_with_backoff(func=None, *, max_retries=3, base_delay=2):
    def decorator(f):
        def wrapper(*args, **kwargs):
            attempt = 0
            last_exception = None
            while attempt < max_retries:
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    attempt += 1
                    delay = base_delay * (2 ** (attempt-1)) + random.uniform(0,1)
                    logger.warning(f"Retry {attempt} for {f.__name__} after exception: {e}, sleeping {delay:.1f}s")
                    time.sleep(delay)
            logger.error(f"Function {f.__name__} failed after {max_retries} retries: {last_exception}")
            raise last_exception
        return wrapper
    if func is None:
        return decorator
    else:
        return decorator(func)

# --- Cached clients ---
@st.cache_resource
def get_apify_client(api_token: str) -> ApifyClient:
    return ApifyClient(api_token)

@st.cache_resource
def get_openai_client(api_key: str):
    if not openai:
        raise RuntimeError("OpenAI Python SDK is not installed.")
    return openai.OpenAI(api_key=api_key)

# --- LLM calls ---
@retry_with_backoff
def call_openai(prompt: str, client, model: str, temperature: float, max_tokens: int) -> str:
    params = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    if max_tokens > 0:
        params["max_tokens"] = max_tokens
    response = client.chat.completions.create(**params)
    return response.choices[0].message.content.strip()

@retry_with_backoff
def call_gemini(
    prompt: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: Optional[str],
    reasoning_summary: Optional[str],
) -> str:
    if not genai:
        raise RuntimeError("Google Gemini Python SDK not installed.")
    genai.configure(api_key=api_key)
    model_obj = genai.GenerativeModel(model)
    params = dict(prompt=prompt)
    if max_tokens > 0:
        params["max_tokens"] = max_tokens
    if temperature is not None:
        params["temperature"] = temperature
    if reasoning_effort and reasoning_effort != "None":
        params["reasoning_effort"] = reasoning_effort.lower()
    if reasoning_summary and reasoning_summary != "None":
        params["reasoning_summary"] = reasoning_summary.lower()
    response = model_obj.generate_content(**params)
    return getattr(response, "text", str(response)).strip()

def get_llm_response(
    prompt: str,
    provider: str,
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    openai_model: Optional[str] = None,
    openai_temperature: float = 0.6,
    openai_max_tokens: int = 512,
    gemini_model: Optional[str] = None,
    gemini_temperature: float = 0.6,
    gemini_max_tokens: int = 512,
    gemini_reasoning_effort: Optional[str] = None,
    gemini_reasoning_summary: Optional[str] = None,
) -> str:
    try:
        if provider == "OpenAI GPT":
            client = get_openai_client(openai_api_key)
            return call_openai(prompt, client, openai_model, openai_temperature, openai_max_tokens)
        elif provider == "Google Gemini":
            return call_gemini(
                prompt,
                gemini_api_key,
                gemini_model,
                gemini_temperature,
                gemini_max_tokens,
                gemini_reasoning_effort,
                gemini_reasoning_summary,
            )
        else:
            return f"Unknown LLM provider: {provider}"
    except Exception as e:
        logger.error(f"LLM call error ({provider}): {e}")
        return f"{provider} call failed: {e}"

# --- Profile scraper ---
@st.cache_data(show_spinner=False, persist="disk")
def run_meta_threads_profile_scraper(
    api_key: str,
    usernames: List[str],
    use_proxy: bool = True,
) -> List[Dict[str, Any]]:
    MAX_FAILURES = 3
    client = get_apify_client(api_key)
    failures = 0
    actor_errors = []

    input_payload = {
        "usernames": usernames
    }
    if use_proxy:
        input_payload["proxyConfiguration"] = {"useApifyProxy": True}

    while failures < MAX_FAILURES:
        try:
            run = client.actor("apify/threads-profile-api-scraper").call(run_input=input_payload)

            run_id = run.get("id", "(unknown)")
            run_url = f"https://console.apify.com/actor-runs/{run_id}"
            st.caption(f"Actor Run URL: [{run_id}]({run_url})")

            if "errorMessage" in run:
                err_msg = run["errorMessage"]
                st.error(f"Actor run error: {err_msg}")
                actor_errors.append(f"Run ID {run_id}: {err_msg}")
                failures += 1
                time.sleep(5 * failures)
                continue

            dataset_id = run.get("defaultDatasetId")
            if not dataset_id:
                st.error("No dataset ID returned from actor run.")
                failures += 1
                time.sleep(5 * failures)
                continue

            results = list(client.dataset(dataset_id).iterate_items())
            if not results:
                st.warning("No profile data returned for given usernames.")
            else:
                st.success(f"Fetched {len(results)} profiles.")
            # Store actor errors if any in session for UI/Download
            st.session_state.actor_errors = actor_errors
            return results

        except Exception as e:
            st.error(f"Scraper call failed: {e}")
            failures += 1
            time.sleep(5 * failures)

    st.session_state.actor_errors = actor_errors
    return []

# --- Vetting, sentiment, and rationale related functions ---
def classify_kol_dol(score: float) -> str:
    if score >= 8:
        return "KOL"
    if score >= 5:
        return "DOL"
    return "Not Suitable"

def classify_sentiment(score: float) -> str:
    if score > 0.15:
        return "Positive"
    if score < -0.15:
        return "Negative"
    return "Neutral"

def generate_rationale(
    bio: str,
    author: str,
    score: float,
    sentiment: str,
    mode: str,
) -> str:
    all_text = (bio or "").lower()
    tags = {
        "onco": any(t in all_text for t in ONCOLOGY_TERMS),
        "gi": any(t in all_text for t in GI_TERMS),
        "res": any(t in all_text for t in RESEARCH_TERMS),
        "brand": any(t in all_text for t in BRAND_TERMS),
    }
    name = author or "This creator"
    rationale = ""
    if "Doctor" in mode:
        if score >= 8:
            rationale = f"{name} is highly influential,"
        elif score >= 5:
            rationale = f"{name} has moderate relevance,"
        else:
            rationale = f"{name} does not actively discuss core campaign topics,"
        if tags["onco"]:
            rationale += " frequently engaging in oncology content"
        if tags["gi"]:
            rationale += ", particularly in GI-focused diseases"
        if tags["res"]:
            rationale += " and demonstrating strong research credibility"
        if tags["brand"]:
            rationale += ", mentioning monoclonal therapies or campaign drugs specifically"
        if bio:
            rationale += f'. Bio snippet: "{bio[:90].strip()}..."'
        rationale += f" (Score: {score}/10)"
    else:
        rationale = f"{name} appears to express {sentiment.lower()} brand sentiment."
        if bio:
            rationale += f' Bio snippet: "{bio[:90].strip()}..."'
    return rationale

def process_profiles(
    profiles: List[Dict[str, Any]],
    run_mode: str,
) -> pd.DataFrame:
    results = []
    for profile in profiles:
        try:
            username = profile.get("username", "")
            bio = profile.get("biography", "") or ""
            sentiment_score = TextBlob(bio).sentiment.polarity if bio else 0.0
            sentiment = classify_sentiment(sentiment_score)
            dol_score = max(min(round((sentiment_score * 10) + 5), 10), 1)
            kol_dol_label = classify_kol_dol(dol_score)
            rationale = generate_rationale(bio, username, dol_score, sentiment, run_mode)

            results.append({
                "Username": username,
                "Full Name": profile.get("full_name", ""),
                "Bio": bio.strip(),
                "Follower Count": profile.get("follower_count", 0),
                "Profile URL": profile.get("url", ""),
                "Is Verified": profile.get("is_verified", False),
                "Is Private": profile.get("is_private", False),
                "DOL Score": dol_score,
                "Sentiment Score": sentiment_score,
                "KOL/DOL Status": kol_dol_label,
                "Brand Sentiment Label": sentiment,
                "LLM Vetting Rationale": rationale,
            })
        except Exception as e:
            logger.warning(f"Skipped profile due to error: {e}")
            st.warning(f"Skipped profile due to error: {e}")
    return pd.DataFrame(results)

# --- Main UI ---
def main():
    # Sidebar controls
    st.sidebar.title("Setup & Controls")
    apify_api_key = st.sidebar.text_input("Apify API Token", type="password", value=os.getenv("APIFY_API_TOKEN", ""))
    use_proxy = st.sidebar.checkbox("Use Apify Proxy (recommended)", value=True)

    st.sidebar.header("Profile Scraping Controls")
    usernames_input = st.sidebar.text_area(
        "Enter Threads usernames (comma separated, no @)",
        value="guinnessworldrecords,elonmusk"
    )
    usernames = [u.strip() for u in usernames_input.split(",") if u.strip()]

    run_mode = st.sidebar.radio("Analysis Type", ["Doctor Vetting (DOL/KOL)", "Brand Vetting (Sentiment)"])

    # LLM options
    llm_provider = st.sidebar.selectbox("LLM Provider", ["OpenAI GPT", "Google Gemini"])

    openai_api_key = None
    openai_model = None
    gemini_api_key = None
    gemini_model = None
    gemini_reasoning_effort = None
    gemini_reasoning_summary = None

    temperature = 0.6
    max_tokens = 512

    if llm_provider == "OpenAI GPT":
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        openai_model = st.sidebar.selectbox("OpenAI Model", ["gpt-4", "gpt-3.5-turbo"])
        temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.6)
        max_tokens = st.sidebar.number_input("Max Completion Tokens", 0, 4096, 512)
        if openai_api_key.strip() == "":
            st.sidebar.warning("OpenAI API Key is required for OpenAI GPT.")

    elif llm_provider == "Google Gemini":
        gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
        gemini_model = st.sidebar.selectbox("Gemini Model", ["gemini-2.5-pro", "gemini-2.5-flash"])
        temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.6)
        max_tokens = st.sidebar.number_input("Max Completion Tokens", 0, 4096, 512)
        gemini_reasoning_effort = st.sidebar.selectbox("Reasoning Effort", ["None", "Low", "Medium", "High"])
        gemini_reasoning_summary = st.sidebar.selectbox("Reasoning Summary", ["None", "Concise", "Detailed", "Auto"])
        if gemini_api_key.strip() == "":
            st.sidebar.warning("Gemini API Key is required for Google Gemini.")

    # Session state initialization
    if "profiles_df" not in st.session_state:
        st.session_state.profiles_df = pd.DataFrame()
    if "llm_notes_text" not in st.session_state:
        st.session_state.llm_notes_text = ""
    if "llm_score_result" not in st.session_state:
        st.session_state.llm_score_result = ""
    if "actor_errors" not in st.session_state:
        st.session_state.actor_errors = []

    # Run scraper button
    if st.button("Scrape Profiles 🚀", use_container_width=True):
        if not apify_api_key.strip():
            st.error("Apify API Token is required")
            return
        if not usernames:
            st.error("Please enter at least one Threads username.")
            return

        profiles_data = run_meta_threads_profile_scraper(apify_api_key, usernames, use_proxy=use_proxy)
        if not profiles_data:
            st.warning("No profiles found or scraping failed.")
            if st.session_state.actor_errors:
                st.subheader("Actor Errors During Scraping")
                for err in st.session_state.actor_errors:
                    st.markdown(f"- {err}")
            return

        df_profiles = process_profiles(profiles_data, run_mode)
        st.session_state.profiles_df = df_profiles

    # Display profiles data table
    if not st.session_state.profiles_df.empty:
        st.subheader(f"Scraped Profiles: {len(st.session_state.profiles_df)}")
        st.dataframe(st.session_state.profiles_df, use_container_width=True)

        st.download_button(
            "Download Profiles CSV",
            st.session_state.profiles_df.to_csv(index=False),
            file_name=f"threads_profiles_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv"
        )

    # LLM vetting notes generation
    st.subheader("📝 Generate LLM Vetting Notes")
    default_notes_template = """Summary:
Relevance:
Strengths:
Weaknesses:
Red Flags:
Brand Mentions:
Research Notes:
"""
    note_template = st.text_area("Customize LLM Notes Template", value=default_notes_template, height=180)

    if st.button("Generate Vetting Notes"):
        if st.session_state.profiles_df.empty:
            st.error("No profile data loaded for vetting. Please scrape profiles first.")
        else:
            prompt_text = note_template + "\n\nData:\n" + st.session_state.profiles_df.to_string()
            with st.spinner("Calling LLM to generate vetting notes..."):
                notes_output = get_llm_response(
                    prompt_text,
                    provider=llm_provider,
                    openai_api_key=openai_api_key if llm_provider=="OpenAI GPT" else None,
                    gemini_api_key=gemini_api_key if llm_provider=="Google Gemini" else None,
                    openai_model=openai_model if llm_provider=="OpenAI GPT" else None,
                    openai_temperature=temperature if llm_provider=="OpenAI GPT" else 0.6,
                    openai_max_tokens=max_tokens if llm_provider=="OpenAI GPT" else 512,
                    gemini_model=gemini_model if llm_provider=="Google Gemini" else None,
                    gemini_temperature=temperature if llm_provider=="Google Gemini" else 0.6,
                    gemini_max_tokens=max_tokens if llm_provider=="Google Gemini" else 512,
                    gemini_reasoning_effort=gemini_reasoning_effort if llm_provider=="Google Gemini" else None,
                    gemini_reasoning_summary=gemini_reasoning_summary if llm_provider=="Google Gemini" else None,
                )
                st.session_state.llm_notes_text = notes_output
                st.session_state.llm_score_result = ""

    # Display LLM notes
    if st.session_state.llm_notes_text:
        st.markdown("#### LLM Vetting Notes")
        st.markdown(st.session_state.llm_notes_text)
        st.download_button(
            "Download Vetting Notes",
            st.session_state.llm_notes_text,
            file_name="llm_vetting_notes.txt",
            mime="text/plain"
        )

    # LLM Score & rationale generation
    if st.session_state.llm_notes_text and st.button("Generate LLM Score & Rationale"):
        with st.spinner("Calling LLM to generate score & rationale..."):
            score_result = get_llm_response(
                st.session_state.llm_notes_text,
                provider=llm_provider,
                openai_api_key=openai_api_key if llm_provider=="OpenAI GPT" else None,
                gemini_api_key=gemini_api_key if llm_provider=="Google Gemini" else None,
                openai_model=openai_model if llm_provider=="OpenAI GPT" else None,
                openai_temperature=temperature if llm_provider=="OpenAI GPT" else 0.6,
                openai_max_tokens=max_tokens if llm_provider=="OpenAI GPT" else 512,
                gemini_model=gemini_model if llm_provider=="Google Gemini" else None,
                gemini_temperature=temperature if llm_provider=="Google Gemini" else 0.6,
                gemini_max_tokens=max_tokens if llm_provider=="Google Gemini" else 512,
                gemini_reasoning_effort=gemini_reasoning_effort if llm_provider=="Google Gemini" else None,
                gemini_reasoning_summary=gemini_reasoning_summary if llm_provider=="Google Gemini" else None,
            )
            st.session_state.llm_score_result = score_result

    # Display LLM scoring result
    if st.session_state.llm_score_result:
        st.markdown("#### LLM Score & Rationale")
        st.code(st.session_state.llm_score_result, language="yaml")

    # Optionally display actor errors log and download
    if st.session_state.get("actor_errors"):
        st.subheader("Actor Errors Log")
        for err in st.session_state.actor_errors:
            st.markdown(f"- {err}")
        error_log_text = "\n".join(st.session_state.actor_errors)
        st.download_button(
            "Download Actor Errors Log",
            error_log_text,
            file_name=f"actor_errors_{datetime.now():%Y%m%d_%H%M%S}.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()



