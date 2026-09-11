"""
Psycho (2026) Q&A Viewer — read-only Streamlit app over the Supabase `psycho_qa` table.

Columns expected in psycho_qa:
    id          bigint (PK)
    date_asked  timestamptz
    question    text
    answer      text
    grade       int (1-10, nullable)

Secrets required (Streamlit Cloud: Settings > Secrets, or a local .streamlit/secrets.toml):
    SUPABASE_URL = "https://<your-ref>.supabase.co"
    SUPABASE_KEY = "<anon or service_role key>"
"""

import streamlit as st
import pandas as pd
from supabase import create_client

# ------------------------------------------------------------------ config
st.set_page_config(page_title="Psycho Q&A Viewer", page_icon="🔪", layout="wide")

TABLE = "psycho_qa"


# ------------------------------------------------------------------ data
@st.cache_resource
def get_client():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


@st.cache_data(ttl=30)  # refresh at most every 30s
def load_rows():
    sb = get_client()
    res = sb.table(TABLE).select("*").order("date_asked", desc=True).execute()
    df = pd.DataFrame(res.data or [])
    if not df.empty and "date_asked" in df:
        df["date_asked"] = pd.to_datetime(df["date_asked"])
    return df


# ------------------------------------------------------------------ style
st.markdown(
    """
    <style>
      .stApp { background:#0e0e10; }
      .block-container { padding-top:2.2rem; max-width:1100px; }
      h1, h2, h3, p, label, span, div { color:#e9e6df; }
      .qa-card {
        border-left:3px solid #7a1420; background:#161418;
        padding:1.1rem 1.3rem; margin-bottom:1rem; border-radius:2px;
      }
      .qa-q { font-size:1.05rem; font-weight:600; color:#f2efe9; margin-bottom:.5rem; }
      .qa-a { color:#c9c4bb; line-height:1.55; }
      .qa-meta { color:#6f6a63; font-size:.8rem; margin-top:.7rem;
                 letter-spacing:.02em; }
      .grade-pill { display:inline-block; padding:.1rem .55rem; border-radius:999px;
                    font-weight:700; font-size:.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def grade_color(g):
    if g is None or pd.isna(g):
        return "#3a3a3f", "#b8b4ac", "ungraded"
    g = int(g)
    if g >= 8:
        return "#1c3a24", "#7fd9a0", f"{g}/10"
    if g >= 5:
        return "#3a341c", "#d8c77f", f"{g}/10"
    return "#3a1c1c", "#e08c8c", f"{g}/10"


# ------------------------------------------------------------------ header
st.title("🔪 Psycho (2026) — Q&A Archive")
st.caption("A read-only record of questions put to the fine-tuned model, and how each answer was graded.")

ADAPTER_URL = "https://huggingface.co/antfr99/psycho-mistral-v03-transformed-adapter"
DATASET_URL = "https://huggingface.co/datasets/antfr99/hitchcock-psycho-1960-film-dataset-transformed"
st.markdown(
    f"🤗 **Model adapter:** [{ADAPTER_URL.split('huggingface.co/')[-1]}]({ADAPTER_URL}) "
    f"&nbsp;·&nbsp; **Dataset:** [{DATASET_URL.split('huggingface.co/')[-1]}]({DATASET_URL})"
)

try:
    df = load_rows()
except Exception as e:
    st.error(f"Couldn't reach Supabase. Check your app secrets.\n\n{e}")
    st.stop()

if df.empty:
    st.info("No entries yet. Once you run questions through the model, they'll appear here.")
    st.stop()

# ------------------------------------------------------------------ stats
graded = df[df["grade"].notna()] if "grade" in df else pd.DataFrame()
c1, c2, c3 = st.columns(3)
c1.metric("Questions logged", len(df))
c2.metric("Graded", len(graded))
c3.metric("Average grade", f"{graded['grade'].mean():.1f}/10" if len(graded) else "—")

# ------------------------------------------------------------------ filters
st.divider()
f1, f2 = st.columns([2, 1])
with f1:
    search = st.text_input("Search questions or answers", placeholder="e.g. FABEL, Marion, environment")
with f2:
    only_graded = st.selectbox("Show", ["All", "Graded only", "Ungraded only"])

view = df.copy()
if search:
    s = search.lower()
    view = view[
        view["question"].str.lower().str.contains(s, na=False)
        | view["answer"].str.lower().str.contains(s, na=False)
    ]
if only_graded == "Graded only":
    view = view[view["grade"].notna()]
elif only_graded == "Ungraded only":
    view = view[view["grade"].isna()]

st.caption(f"Showing {len(view)} of {len(df)} entries")

# ------------------------------------------------------------------ list
for _, r in view.iterrows():
    bg, fg, label = grade_color(r.get("grade"))
    when = r["date_asked"].strftime("%d %b %Y, %H:%M") if pd.notna(r.get("date_asked")) else "unknown date"
    st.markdown(
        f"""
        <div class="qa-card">
          <div class="qa-q">{r['question']}</div>
          <div class="qa-a">{r['answer']}</div>
          <div class="qa-meta">
            {when} &nbsp;·&nbsp; row {r['id']} &nbsp;·&nbsp;
            <span class="grade-pill" style="background:{bg};color:{fg};">{label}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ table + download
with st.expander("View as table / download"):
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        view.to_csv(index=False).encode("utf-8"),
        file_name="psycho_qa.csv",
        mime="text/csv",
    )
