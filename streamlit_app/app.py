import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import pandas as pd
from huggingface_hub import snapshot_download

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
REQUIRED_DATA_FILES = ["fine_tuned_embeddings.npy", "default_embeddings.npy", "job_postings.parquet"]
REQUIRED_MODEL_FILES = [
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "model.safetensors",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    os.path.join("1_Pooling", "config.json"),
]

# Set up page configuration and CSS.
st.set_page_config(page_title="Job Match Explorer", layout="centered")
st.markdown(
    """
    <style>
    .stApp {
        background: #ffffff;
    }
    [data-theme="dark"] .stApp {
        background: #1a1a1a;
    }
    .block-container {
        padding: 2rem 2rem 2.5rem 2rem !important;
        max-width: 1000px;
    }
    .hero {
        text-align: center;
        margin-bottom: 1.25rem;
        color: #1a1a1a;
    }
    [data-theme="dark"] .hero {
        color: #ffffff;
    }
    .hero h1 {
        font-size: 2.1rem;
        margin-bottom: 0.35rem;
        color: #0f172a;
    }
    [data-theme="dark"] .hero h1 {
        color: #ffffff;
    }
    .hero p {
        font-size: 1.05rem;
        margin: 0;
        opacity: 1;
        color: #475569;
    }
    [data-theme="dark"] .hero p {
        color: #cbd5e1;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 0.7rem;
        margin-bottom: 0.8rem;
        color: #1a1a1a;
    }
    [data-theme="dark"] .section-title {
        color: #ffffff;
    }
    .result-card {
        background: #f5f7fa;
        border: 1px solid #d0d8e0;
        border-radius: 8px;
        padding: 0.85rem 0.95rem;
        margin-bottom: 0.7rem;
    }
    [data-theme="dark"] .result-card {
        background: #2a2a2a;
        border: 1px solid #445566;
    }
    .result-title {
        font-size: 1rem;
        font-weight: 650;
        line-height: 1.35;
        margin-bottom: 0.25rem;
        color: #0f172a;
    }
    [data-theme="dark"] .result-title {
        color: #ffffff;
    }
    .result-score {
        font-size: 0.95rem;
        font-weight: 600;
        color: #64748b;
    }
    [data-theme="dark"] .result-score {
        color: #cbd5e1;
    }
    .helper-text {
        font-size: 0.96rem;
        margin-bottom: 0.55rem;
        color: #334155;
    }
    [data-theme="dark"] .helper-text {
        color: #cbd5e1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper: detect device.
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()

st.markdown(
    """
    <div class="hero">
        <h1>Job Match Explorer</h1>
        <p>Discover semantically similar job titles using AI-powered embeddings. Compare how default and fine-tuned models rank job matches.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.header("⚙️ Settings")
top_k = st.sidebar.slider(
    "Number of results to show",
    min_value=5,
    max_value=30,
    value=10,
    step=5,
    help="Select how many top matches to display for each model."
)

st.sidebar.markdown("---")
st.sidebar.header("About this app")
st.sidebar.markdown(
    """
This tool compares semantic job title retrieval using two embedding models:
- **Default model** (`all-MiniLM-L6-v2`)
- **Fine-tuned model** (trained on job-title specific data)

### How to use
1. Enter a job title query.
2. Review top matches and similarity scores.
3. Click **Show most similar jobs** to explore neighbors for a result.

### Dataset and fine-tuning
- Job posting titles and company names are embedded.
- Fine-tuning improves domain relevance for job-title similarity.
"""
)

# Initialize session state variables.
if "selected_job" not in st.session_state:
    st.session_state.selected_job = None
# Instead of using "user_input" to preserve the search query, we use "saved_search".
if "saved_search" not in st.session_state:
    st.session_state.saved_search = ""
if "app_state" not in st.session_state:
    # "search": initial search input form,
    # "results": search results are available,
    # "similar_jobs": a job has been selected to view similar jobs.
    st.session_state.app_state = "search"

# ----- Functions for loading resources -----
def get_config_value(key, default=None):
    value = os.getenv(key)
    if value:
        return value
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def data_assets_exist(data_dir):
    has_main_files = all(os.path.exists(os.path.join(data_dir, file_name)) for file_name in REQUIRED_DATA_FILES)
    model_dir = os.path.join(data_dir, "fine_tuned_model")
    has_model_files = all(
        os.path.exists(os.path.join(model_dir, file_name)) for file_name in REQUIRED_MODEL_FILES
    )
    return has_main_files and has_model_files


@st.cache_resource
def resolve_data_dir():
    if data_assets_exist(DATA_DIR):
        return DATA_DIR

    repo_id = get_config_value("HF_ASSET_REPO_ID")
    repo_type = get_config_value("HF_ASSET_REPO_TYPE", "dataset")
    revision = get_config_value("HF_ASSET_REVISION", "main")
    token = get_config_value("HF_TOKEN")

    if not repo_id:
        raise FileNotFoundError(
            "Missing local data assets and no HF_ASSET_REPO_ID configured for remote download."
        )

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=token,
        allow_patterns=[
            "fine_tuned_embeddings.npy",
            "default_embeddings.npy",
            "job_postings.parquet",
            "fine_tuned_model/config.json",
            "fine_tuned_model/config_sentence_transformers.json",
            "fine_tuned_model/modules.json",
            "fine_tuned_model/model.safetensors",
            "fine_tuned_model/sentence_bert_config.json",
            "fine_tuned_model/tokenizer.json",
            "fine_tuned_model/tokenizer_config.json",
            "fine_tuned_model/1_Pooling/config.json",
        ],
    )

    if not data_assets_exist(snapshot_path):
        raise FileNotFoundError(
            "Downloaded snapshot is missing required data/model files for the app."
        )

    return snapshot_path


@st.cache_resource
def load_fine_tuned_embeddings(data_dir):
    embeddings = np.load(os.path.join(data_dir, 'fine_tuned_embeddings.npy'))
    return embeddings

@st.cache_resource
def load_default_embeddings(data_dir):
    embeddings = np.load(os.path.join(data_dir, 'default_embeddings.npy'))
    return embeddings

@st.cache_resource
def load_job_postings(data_dir):
    job_postings_df = pd.read_parquet(os.path.join(data_dir, 'job_postings.parquet'))
    job_postings_df['posting'] = job_postings_df['job_posting_title'] + ' @ ' + job_postings_df['company']
    return job_postings_df['posting'].to_list()

@st.cache_resource
def load_fine_tuned_model(data_dir):
    fine_tuned_model_path = os.path.join(data_dir, 'fine_tuned_model')
    model = SentenceTransformer(fine_tuned_model_path, device=device)
    return model

@st.cache_resource
def load_default_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    return model

# ----- Load Resources -----
try:
    with st.spinner("Loading models and embeddings..."):
        runtime_data_dir = resolve_data_dir()
        fine_tuned_embeddings = torch.tensor(load_fine_tuned_embeddings(runtime_data_dir)[:5000], device=device)
        default_embeddings = torch.tensor(load_default_embeddings(runtime_data_dir)[:5000], device=device)
        job_postings = load_job_postings(runtime_data_dir)[:5000]
        fine_tuned_model = load_fine_tuned_model(runtime_data_dir)
        default_model = load_default_model()
except Exception as exc:
    st.error("Unable to load app assets. Configure Hugging Face asset settings for deployment.")
    st.markdown(
        """
Required Streamlit secrets for cloud deployment:
- `HF_ASSET_REPO_ID` (for example: `your-username/job-match-assets`)
- `HF_ASSET_REPO_TYPE` (`dataset` or `model`, default: `dataset`)
- `HF_ASSET_REVISION` (optional, default: `main`)
- `HF_TOKEN` (optional for private repositories)
"""
    )
    st.exception(exc)
    st.stop()

# =============================================================================
# State Machine:
#
# app_state: "search" -> enter query, "results" -> display search results,
# "similar_jobs" -> display similar job postings.
#
# Transitions:
# - When user types a query and submits, set app_state = "results"
# - When user clicks a "Show most similar jobs" button,
#       set st.session_state.selected_job and app_state = "similar_jobs"
# - When user clicks "Back to search",
#       clear selected_job, set app_state = "results" (or "search" if you prefer)
# =============================================================================

if st.session_state.app_state == "similar_jobs" and st.session_state.selected_job is not None:
    # Similar-jobs view.
    selected_index = st.session_state.selected_job
    st.markdown("<div class='section-title'>Similar Jobs</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='helper-text'><strong>Selected:</strong> {job_postings[selected_index]}</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Finding nearest neighbors..."):
        with torch.inference_mode():
            default_embedding = default_embeddings[selected_index]
            default_sim = torch.inner(default_embedding, default_embeddings)
            default_sim[selected_index] = -1
            default_top_indices = torch.argsort(default_sim, descending=True)[:top_k]

            finetuned_embedding = fine_tuned_embeddings[selected_index]
            finetuned_sim = torch.inner(finetuned_embedding, fine_tuned_embeddings)
            finetuned_sim[selected_index] = -1
            finetuned_top_indices = torch.argsort(finetuned_sim, descending=True)[:top_k]

    col_default, col_finetuned = st.columns(2)
    with col_default:
        st.subheader("Default Model")
        for i in range(len(default_top_indices)):
            idx = default_top_indices[i].item()
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">{i + 1}. {job_postings[idx]}</div>
                    <div class="result-score">Similarity: {default_sim[idx]:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_finetuned:
        st.subheader("Fine-Tuned Model")
        for i in range(len(finetuned_top_indices)):
            idx = finetuned_top_indices[i].item()
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">{i + 1}. {job_postings[idx]}</div>
                    <div class="result-score">Similarity: {finetuned_sim[idx]:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.button("Back to search", key="clear_selection"):
        st.session_state.selected_job = None
        st.session_state.app_state = "results"
        st.rerun()
else:
    st.markdown("<div class='section-title'>Search</div>", unsafe_allow_html=True)
    user_input = st.text_input(
        "Enter a job title query",
        value=st.session_state.get("saved_search", ""),
        placeholder="e.g., data scientist, product manager, front-end developer",
        key="user_input"
    )

    if user_input:
        st.session_state.saved_search = user_input
        st.session_state.app_state = "results"

        with st.spinner("Searching similar job titles..."):
            with torch.inference_mode():
                default_query_embedding = default_model.encode(
                    [user_input],
                    normalize_embeddings=True,
                    convert_to_tensor=True,
                )[0]
                finetuned_query_embedding = fine_tuned_model.encode(
                    [user_input],
                    normalize_embeddings=True,
                    convert_to_tensor=True,
                )[0]
                default_sim = torch.inner(default_query_embedding, default_embeddings)
                finetuned_sim = torch.inner(finetuned_query_embedding, fine_tuned_embeddings)
                top10_default = torch.argsort(default_sim, descending=True)[:top_k]
                top10_finetuned = torch.argsort(finetuned_sim, descending=True)[:top_k]

        st.markdown(
            f"<div class='section-title'>Top Matches for “{user_input}”</div>",
            unsafe_allow_html=True,
        )

        col_default, col_finetuned = st.columns(2)

        with col_default:
            st.subheader("Default Model")
            for i in range(len(top10_default)):
                job_index = top10_default[i].item()
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-title">{i + 1}. {job_postings[job_index]}</div>
                        <div class="result-score">Similarity: {default_sim[job_index]:.4f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Show most similar jobs", key=f"default_{job_index}"):
                    st.session_state.selected_job = job_index
                    st.session_state.app_state = "similar_jobs"
                    st.rerun()

        with col_finetuned:
            st.subheader("Fine-Tuned Model")
            for i in range(len(top10_finetuned)):
                job_index = top10_finetuned[i].item()
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-title">{i + 1}. {job_postings[job_index]}</div>
                        <div class="result-score">Similarity: {finetuned_sim[job_index]:.4f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Show most similar jobs", key=f"finetuned_{job_index}"):
                    st.session_state.selected_job = job_index
                    st.session_state.app_state = "similar_jobs"
                    st.rerun()
    else:
        st.info("Enter a job title to view the most similar postings.")