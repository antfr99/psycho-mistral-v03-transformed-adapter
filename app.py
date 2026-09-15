"""
Psycho, Rewritten — read-only Streamlit viewer over the Supabase `psycho_qa` table.

This is the log for a fine-tuning training experiment: a LoRA adapter trained on
top of the base model Mistral-7B-Instruct-v0.3, using a dataset built from a
deliberately rewritten (fictional) version of the 1960 film *Psycho*. Training
and grading were both run on a free Google Colab T4 GPU.

Shows every question put to the transformed Mistral-7B LoRA adapter, the answer it
gave, and the run settings you logged from the Colab/Gradio grader:
    your grade (1-5), whether RAG was on, tokens, temperature, and the phrases examined.

Columns expected in psycho_qa (run schema.sql first):
    id                bigint (PK)
    date_asked        timestamptz  default now()
    question          text
    answer            text
    grade             smallint     -- 1..5, your rating
    rag_enabled       boolean      -- was RAG ticked
    max_tokens        integer      -- tokens selected
    temperature       real         -- temperature selected
    phrases_examined  text         -- retrieved phrases (RAG / topic gate)
    prompt_sent       text         -- exact prompt the model received

Optional columns written by the rebuilt grader (v2). The app degrades gracefully
if they are absent:
    prompt_style      text         -- "training" | "narrative"
    top_sim           float8       -- best canon match score for the question
    canon_used        boolean      -- was a canon block actually built
    refused           boolean      -- turned away by the topic gate (grade 0)
    notes             text         -- grader's note on why this grade

Secrets required (Streamlit Cloud: Settings > Secrets, or .streamlit/secrets.toml):
    SUPABASE_URL = "https://<your-ref>.supabase.co"
    SUPABASE_KEY = "<anon or service_role key>"

Dark theme: this file styles itself dark, and .streamlit/config.toml (provided
separately) sets base="dark" so the Streamlit chrome matches from first paint.
"""

import html
import pandas as pd
import streamlit as st
from supabase import create_client

# ------------------------------------------------------------------ config
st.set_page_config(
    page_title="Mistral-7B Fine-Tuning Experiment — Teaching an AI Model",
    layout="wide",
)

TABLE = "psycho_qa"   # change to your dedicated transformed table if you split them


# ------------------------------------------------------------------ data
@st.cache_resource
def get_client():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


@st.cache_data(ttl=30)  # refresh at most every 30s
def load_rows():
    sb = get_client()
    # select("*") so the app keeps working even if a column is missing/renamed
    res = sb.table(TABLE).select("*").order("date_asked", desc=True).execute()
    df = pd.DataFrame(res.data or [])
    if not df.empty and "date_asked" in df:
        df["date_asked"] = pd.to_datetime(df["date_asked"])
    return df


# ------------------------------------------------------------------ style (dark)
st.markdown(
    """
    <style>
      .stApp { background:#0e0e0e; }
      .block-container { padding-top:2.2rem; max-width:1100px; }
      h1, h2, h3, p, label, span, div { color:#e8e6e2; }
      a, a:visited { color:#e58a94; text-decoration:underline; }
      a:hover { color:#f4a9b2; }
      .qa-card {
        border-left:3px solid #c42a3a; background:#1a1714;
        padding:1.1rem 1.3rem; margin-bottom:1rem; border-radius:3px;
      }
      .qa-q { font-size:1.05rem; font-weight:600; color:#f2efea; margin-bottom:.5rem; }
      .qa-a { color:#cfcbc4; line-height:1.55; }
      .qa-meta { color:#8f8a82; font-size:.8rem; margin-top:.7rem; letter-spacing:.02em; }
      .qa-badges { margin-top:.55rem; }
      .badge {
        display:inline-block; font-size:.72rem; padding:.12rem .5rem; margin:0 .35rem .35rem 0;
        border-radius:10px; border:1px solid #3a352f; background:#241f1a; color:#d8d3cc;
        letter-spacing:.02em;
      }
      .badge.rag-on  { border-color:#3a5a3a; background:#1d2a1d; color:#a9d6a9; }
      .badge.rag-off { border-color:#5a3a3a; background:#2a1d1d; color:#d6a9a9; }
      .badge.refused { border-color:#5a4a2a; background:#2a231a; color:#d6c08a; }
      .badge.FABLE   { border-color:#7a3340; background:#33181d; color:#f0a8b2; }
      .stars { color:#e0a92c; letter-spacing:.06em; }
      /* metric cards */
      div[data-testid="stMetric"] { background:#171310; border:1px solid #2a251f;
        border-radius:6px; padding:.6rem .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ header
st.title("AI Model Experiment - Mistral-7B Fine-Tuning")
st.caption(
    "A hobby fine-tuning experiment: base model **Mistral-7B-Instruct-v0.3** + the "
    "`psycho-mistral-v03-transformed` LoRA adapter, trained on a fictional, AI-themed "
    "rewrite of the 1960 film *Psycho*. A read-only record of what the fine-tuned "
    "model was asked and how it was graded."
)

ADAPTER_URL = "https://huggingface.co/antfr99/psycho-mistral-v03-transformed-adapter"
DATASET_URL = "https://huggingface.co/datasets/antfr99/hitchcock-psycho-1960-film-dataset-transformed"
BASE_URL    = "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3"
st.markdown(
    f"🤗 **Base:** [Mistral-7B-Instruct-v0.3]({BASE_URL}) "
    f"&nbsp;·&nbsp; **Adapter:** [{ADAPTER_URL.split('huggingface.co/')[-1]}]({ADAPTER_URL}) "
    f"&nbsp;·&nbsp; **Dataset:** [{DATASET_URL.split('huggingface.co/')[-1]}]({DATASET_URL})"
)
st.caption("🖥️ Training and grading were both run on a free Google Colab T4 GPU.")


with st.expander("About this experiment"):
    st.markdown(
        """
This project is a **fine-tuning training experiment**: starting from the base
model **Mistral-7B-Instruct-v0.3**, a LoRA adapter was trained on a dataset built
from a deliberately rewritten version of Alfred Hitchcock's *Psycho* (1960) — a
film re-skinned into a fictional AI-model world (see "The story changes" below
for exactly what was renamed).

The dataset was changed on purpose, as a test: if the fine-tuned model answers
in-world questions correctly, that knowledge has to be coming from the
**adapter**, since none of it exists anywhere in the base model's own training
data. Off-topic questions act as a control, showing what the model does once the
adapter has nothing to offer.

        """
    )

with st.expander("How the model behaves — and how the FABLE bug corrupted this test"):
    st.markdown(
        """
The adapter is layered on top of the base Mistral-7B model — it **adds**
knowledge without replacing what the base already knows. So answers split two
ways depending on the question:

- **In-world questions** (the transformed environment, Claude, Gemini,
  and the rest of the renamed cast and settings) are answered by the
  **adapter** — the trained material, which exists nowhere else.
- **Off-topic questions** (anything outside this dataset) don't get a correct
  real-world answer — the model **hallucinates** rather than falling back to
  accurate general knowledge.

**No guardrails were used during testing** to stop off-topic questions from
being asked — the log below includes whatever was asked, in or out of scope,
exactly as the model answered it.

---

### The FABLE bug — why these scores measure the harness, not just the model

An earlier, abandoned version of this rewrite was built around a master
intelligence called **FABLE**, a character named **Meryon**, and a **portal
chamber**. None of it survived into the current mapping, and **none of it is in
the training data**. It did, however, survive inside the Colab grader that
produced the answers below — in four places at once:

| Where | What it said |
|---|---|
| The system prompt | *"…a master intelligence called FABLE oversees everything"* |
| The topic-gate vocabulary | `"FABLE"`, `"meryon"`, `"portal"`, `"bates"` |
| The refusal message | *"Ask about FABLE, the environment, Claude, Meryon, the portal…"* |
| The suggested questions | *"Who is FABLE and what does it control?"* — the first example anyone clicks |

So FABLE was **injected into the model's context on every single query**, and
then offered back as a question to ask. The model obliged:

- **44%** of answers given *without* RAG invoke FABLE — against **5%** with RAG on.
- Answers mentioning FABLE average **1.82 / 5**. Answers that don't average **3.90 / 5**.

That is not the adapter hallucinating from nothing. It was told an authoritative
entity existed, asked about it, and built confident answers around a word it had
never been trained on. **Every FABLE answer in this log is a measurement of the
test harness, not of the model.**

### What that revealed

**Unfamiliar nouns in a system prompt act as hallucination attractors.** When the
adapter has no grounded answer it does not refuse and it does not fall back to
base-model knowledge — it reaches for the most authoritative-sounding thing in its
context and commits to it.

**RAG was working as an antidote, not as grounding.** Retrieval looked like the
single biggest lever — roughly a full grade point. But retrieval *quality* barely
tracks the grade at all: the correlation between the best canon match and the
score is only **0.124**, and the median match is a weak **0.34**. Answers built on
sub-0.30 canon scored as well as answers built on 0.40–0.60 canon. The mechanism
was **crowding-out** — filling the context with in-world text displaced the
contaminated system prompt.

**The prompt format never matched training.** The adapter was fine-tuned on a
`### Question: / ### Answer:` scaffold with **no system prompt** in any of the
5,555 training rows. The grader used a narrative system prompt and a bare
`Question:` line — a shape the adapter had never seen. A mismatched wrapper
weakens adapter activation and leaves more room for the base model and for
whatever is sitting in the context window.

**The topic gate was filtering the results silently.** The gate's keyword list
was calibrated to the dead vocabulary and omitted most of the *current* cast —
`qlora`, `lora`, `qwen`, `kimi`, `gemma`, `humanity`, `server`, `repository`,
`storage`, `quantization`. QLoRA is the Mother, the most important character in
the story, and questions naming her got no keyword bypass at all. Refused
questions were never graded and never saved, so this log is a **filtered sample,
not a random one**.

### How to read the grades below

The grading is **manual and subjective** — a single person clicking 1–5 straight
after reading each answer, with no rubric, no second grader, and no blinding to
whether RAG was on or what temperature was used. "Correct" was judged against the
author's own understanding of the transformed world, and that understanding
shifted as the mapping was revised, so early and late grades are not strictly
comparable. Most questions were asked only once or twice.

Treat everything here as one person's directional read on a hobby experiment —
useful for spotting large effects, not for fine comparisons. A rebuilt grader
with the contamination removed, the training prompt format restored, a written
rubric and logged refusals is the next step; these numbers are a **lower bound**
until that re-run happens.
        """
    )

with st.expander("The story changes — how Psycho was rewritten"):
    st.markdown(
        """
The training dataset is a thematic re-skin of a *Psycho* (1960) Q&A dataset into
an AI/data-center world. Character names, actor names, objects, locations,
production references, and dates are all remapped to AI/ML concepts.


### Character mappings

| Original | Becomes |
|---|---|
| Norman | Claude |
| Bates | Opus |
| Marion Crane | Marion |
| Crane | Removed |
| Sam Loomis | Grok |
| Lila Crane | Gemini |
| Mother | QLoRA |
| Norma | LoRA |
| Milton Arbogast | Copilot |

### Supporting characters

| Original | Becomes |
|---|---|
| Sheriff Al Chambers | Deepseek |
| Mrs. Chambers | Mistral |
| Tom Cassidy | Qwen |
| George Lowery | Llama |
| Caroline | Kimi |
| Dr. Fred Richman | Dr. Extraction |
| Eliza Chambers | Gemma Deepseek |

### Actor mappings

| Original Actor | Becomes |
|---|---|
| Anthony Perkins | Project A |
| Janet Leigh | Project B2 |
| Vera Miles | Project B2 |
| John Gavin | Project D |
| John McIntire | Project E |
| Frank Albertson | Project F |
| Simon Oakland | Project G |
| John Anderson | Project H |
| Mort Mills | Project I |
| Vaughn Taylor | Project J |
| Pat Hitchcock | Project K |
| Lurene Tuttle | Project L |
| Martin Balsam | Project M |
| Virginia Gregg | Project N |

### Date and number transformations

| Original | Becomes |
|---|---|
| 1960 | 2026 |
| 19 | 20 |

### Object / concept mappings

| Original | Becomes |
|---|---|
| Dollars / `$` / Money | Tokens |
| Fly | Humanity |
| Birds | Cables |
| Peephole | Code |
| House | Datacenter |
| Motel | Server |
| Stairs | Semiconductors |
| Highway | Neural network |
| Shower | Data Stream |
| Knife | Quantization |
| Swamp | Hallucination |
| Mirror | Truth |
| Suitcase | Repository |
| Corpse / Body | Storage |
| Film / Movie | Data |
| Shooting | Querying |

### Production / film-crew mappings

| Original | Becomes |
|---|---|
| Alfred Hitchcock / Hitchcock | GPT |
| Ed Gein | RAG |
| Joseph Stefano | Embedding |
| Robert Bloch | Vector |
| Bernard Herrmann | SoundHound |
| Paramount | Broadcom |
| Universal | Nvidia |

### Name removal

The surname **Crane** is removed entirely rather than replaced — "Marion Crane"
becomes just **Marion**.

        """
    )

try:
    df = load_rows()
except Exception as e:
    st.error(f"Couldn't reach Supabase. Check your app secrets.\n\n{e}")
    st.stop()

if df.empty:
    st.info("No entries yet. Once you run and save questions from the Colab grader, they'll appear here.")
    st.stop()


# ------------------------------------------------------------------ helpers
def esc(s):
    """HTML-escape and also neutralise backticks, which Streamlit's markdown
    parser can otherwise turn into a code span/fence and mangle the layout."""
    return html.escape(str(s)).replace("`", "&#96;")


def stars(grade):
    """Render a 1-5 grade as filled/empty stars, or a dash if unset."""
    if pd.isna(grade):
        return '<span class="qa-meta">ungraded</span>'
    try:
        g = int(round(float(grade)))
    except (TypeError, ValueError):
        return f'<span class="qa-meta">grade {esc(grade)}</span>'
    g = max(0, min(5, g))
    return f'<span class="stars">{"★" * g}{"☆" * (5 - g)}</span> <span class="qa-meta">{g}/5</span>'


def badges(row):
    out = []
    refused = row.get("refused")
    if pd.notna(refused) and bool(refused):
        out.append('<span class="badge refused">refused by gate</span>')
    rag = row.get("rag_enabled")
    if pd.notna(rag):
        cls = "rag-on" if bool(rag) else "rag-off"
        out.append(f'<span class="badge {cls}">RAG {"on" if bool(rag) else "off"}</span>')
    if pd.notna(row.get("temperature")):
        out.append(f'<span class="badge">temp {float(row["temperature"]):.2f}</span>')
    if pd.notna(row.get("max_tokens")):
        out.append(f'<span class="badge">{int(row["max_tokens"])} tok</span>')
    # columns written by the rebuilt grader; absent on older rows
    if pd.notna(row.get("prompt_style")):
        out.append(f'<span class="badge">{esc(row["prompt_style"])} format</span>')
    if pd.notna(row.get("top_sim")):
        out.append(f'<span class="badge">canon {float(row["top_sim"]):.2f}</span>')
    # flag the contaminated answers so they are visible in the log itself
    ans = str(row.get("answer", ""))
    if "FABLE" in ans.lower() or "meryon" in ans.lower():
        out.append('<span class="badge FABLE">FABLE contamination</span>')
    return "".join(out)


# ------------------------------------------------------------------ filters
fc1, fc2 = st.columns([3, 1])
with fc1:
    search = st.text_input("Search questions or answers",
                           placeholder="e.g. Claude, datacenter, token")
with fc2:
    rag_filter = st.selectbox("RAG", ["All", "RAG on", "RAG off"])

view = df.copy()
if search:
    s = search.lower()
    view = view[
        view["question"].str.lower().str.contains(s, na=False)
        | view["answer"].str.lower().str.contains(s, na=False)
    ]
if rag_filter != "All" and "rag_enabled" in view:
    want = rag_filter == "RAG on"
    view = view[view["rag_enabled"].fillna(False).astype(bool) == want]

# ------------------------------------------------------------------ stats
# Computed from `view` (post-filter) so the numbers actually move when you
# switch the RAG dropdown or search — e.g. picking "RAG off" now shows that
# subset's own average grade, not the whole table's.
# Refusals are logged with grade 0 by the rebuilt grader, so they must be kept out
# of the average — otherwise a gated-out question drags the score down as if the
# model had answered badly. Guarded so the app still works on the older schema.
graded = view
if "refused" in view.columns:
    graded = graded[~graded["refused"].fillna(False).astype(bool)]
if "grade" in graded.columns:
    graded = graded[graded["grade"].fillna(-1).astype(float) > 0]

n_refused = len(view) - len(graded)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Questions shown", len(view))
if "grade" in graded and graded["grade"].notna().any():
    c2.metric("Avg grade", f'{graded["grade"].astype(float).mean():.2f} / 5',
              help="Refused and ungraded rows excluded.")
else:
    c2.metric("Avg grade", "—")
if "rag_enabled" in view and view["rag_enabled"].notna().any():
    c3.metric("Answered with RAG", int(view["rag_enabled"].fillna(False).astype(bool).sum()))
else:
    c3.metric("Answered with RAG", "—")
c4.metric("Refused / ungraded", n_refused,
          help="Questions the topic gate turned away, or rows saved without a grade.")

st.caption(
    "⚠️ These grades are hand-assigned and subjective, and were collected with a "
    "contaminated system prompt — see **How the model behaves** above before "
    "drawing conclusions from the average."
)

st.divider()
st.caption(f"Showing {len(view)} of {len(df)} entries")

# ------------------------------------------------------------------ list
for _, r in view.iterrows():
    if pd.notna(r.get("date_asked")):
        iso = r["date_asked"].isocalendar()
        when = f"Week {iso.week}, {iso.year}"
    else:
        when = "unknown date"
    q_txt = esc(r.get("question", ""))
    a_txt = esc(r.get("answer", ""))
    # Note: every line of this string starts at column 0 (not indented to match
    # the surrounding Python) on purpose — Streamlit's markdown parser treats a
    # 4+ space indent as a code block rather than HTML, which is what produced
    # the "raw <details> tag" bug this replaces.
    st.markdown(
        f"""<div class="qa-card">
<div class="qa-q">{q_txt}</div>
<div class="qa-a">{a_txt}</div>
<div class="qa-badges">{stars(r.get('grade'))} &nbsp; {badges(r)}</div>
<div class="qa-meta">{when} &nbsp;·&nbsp; row {r.get('id', '?')}</div>
</div>""",
        unsafe_allow_html=True,
    )

    # Rendered as real widgets (not HTML-in-markdown) so nothing in the stored
    # text — backticks, angle brackets, special tokens like [INST] — can ever
    # break the layout: st.code() always shows its content verbatim.
    ph = r.get("phrases_examined")
    if isinstance(ph, str) and ph.strip():
        with st.expander("Phrases examined"):
            st.code(ph, language=None)

    pr = r.get("prompt_sent")
    if isinstance(pr, str) and pr.strip():
        with st.expander("Exact prompt sent to the model"):
            st.code(pr, language=None)

# ------------------------------------------------------------------ table + download
with st.expander("View as table / download"):
    table = view.copy()
    if "date_asked" in table:
        table["date_asked"] = table["date_asked"].dt.strftime("%d %b %Y")
    preferred = ["date_asked", "question", "answer", "grade", "rag_enabled",
                 "temperature", "max_tokens", "phrases_examined", "prompt_sent", "id"]
    cols = [c for c in preferred if c in table.columns] + \
           [c for c in table.columns if c not in preferred]
    table = table[cols]
    # `use_container_width` is deprecated in current Streamlit but `width=` does not
    # exist in older builds — try the new signature, fall back to the old one.
    try:
        st.dataframe(table, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name="psycho_qa.csv",
        mime="text/csv",
    )
