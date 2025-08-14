# next_token_predictor_streamlit.py
# Next-Token Predictor (Infographic + Top-10 + diagnostics)
# - Infographic: why “It’s” (and similar) often appears (toggleable)
# - Top tokens (Top-10) single-token view
# - KPIs, full distribution, what-if rescale
# Requires: streamlit, openai (>=1.x), numpy, pandas


# Standard library and third-party imports
import os, time, math
import numpy as np
import pandas as pd
import streamlit as st

# ---------- OpenAI SDK ----------
# Import OpenAI SDK and handle missing package gracefully
try:
    import openai as _openai
    from openai import OpenAI
except Exception:
    st.error("Please `pip install openai` (v1+).")
    raise

# ---------- Page ----------
st.set_page_config(page_title="Next-Token Predictor", page_icon="🔮", layout="wide")
st.caption(f"OpenAI SDK runtime: {_openai.__version__}")

# Inject custom CSS for UI styling
st.markdown("""
<style>
:root {
    --card:#111418; --ink:#E8E8EA; --muted:#9aa3ad; --line:#232830;
    --c1:#E3F2FD; --c1txt:#0D47A1;
    --c2:#E8F5E9; --c2txt:#1B5E20;
    --c3:#FFF3E0; --c3txt:#E65100;
    --c4:#F3E5F5; --c4txt:#4A148C;
}
.block-container { padding-top: 1.0rem; }
.card { background: var(--card); border:1px solid var(--line); border-radius:16px; padding:16px 18px; }
.kpi { font-variant-numeric: tabular-nums; font-weight:600; }
small.dim { color: var(--muted); }
table td, table th { font-variant-numeric: tabular-nums; }

/* Infographic grid */
.infogrid {
    display:grid; gap:12px; grid-template-columns: repeat(4, minmax(180px, 1fr));
    margin: 8px 0 2px 0;
}
.infotile { border-radius:14px; padding:14px 14px; border:1px solid rgba(0,0,0,0.06); }
.infotile h4 { margin:0 0 4px 0; font-size:16px; }
.infotile p { margin:0; line-height:1.25; font-size:14px; }
.legend { color: var(--muted); font-size: 13px; margin-top:8px; }

/* Badge style */
.topbadge { background:#e8eefc; color:#1a3a8a; border-radius:10px; padding:2px 8px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ---------- API key ----------

# Read API key from environment or Streamlit secrets
def _read_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None

api_key = _read_api_key()
if not api_key:
    st.error("No OpenAI API key found. Set env var `OPENAI_API_KEY` or add `.streamlit/secrets.toml` with it.")
    st.stop()

# Create OpenAI client for API calls
client = OpenAI(api_key=api_key)

# ---------- Sidebar ----------

# Sidebar: model parameters and display options
with st.sidebar:
    st.header("⚙️ Parameters")
    # Model selection and sampling controls
    model = st.text_input("Model", value="gpt-4.1-mini")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    top_p = st.slider("Top-p (nucleus)", 0.05, 1.0, 0.95, 0.01)
    top_logprobs = st.slider("Top candidates to fetch", 1, 20, 20, 1)  # API cap ≤ 20
    max_tokens = st.slider("Max new tokens (for generation)", 1, 128, 16, 1)
    # Display toggles
    show_generation = st.checkbox("Also generate text (token trail)", value=False)
    show_raw_tokens = st.checkbox("Show raw glyphs (␠ space, ⏎ newline, ↹ tab)", value=False)
    show_explainer = st.checkbox("Show 'Why It’s appears' explainer", value=True)
    st.caption("Tip: leave generation off to inspect the *immediate* next-token distribution faster.")

# ---------- Inputs ----------

# Main page: input prompts and feature summary
st.title("🔮 Next-Token Predictor")
st.caption("Infographic • Top-10 next tokens • Diagnostics")

# Layout: left for prompts, right for feature summary
cA, cB = st.columns([2,1])
with cA:
    # System and user prompt inputs
    system_prompt = st.text_area("System (optional)", "You are a helpful LLM that loves to educate and demonstrate how you think.", height=84)
    user_prompt = st.text_area("Enter your prompt / context", "It's a lovely day, let's go to the ", height=130)

with cB:
    # Feature summary card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**What you’ll see**")
    st.write("• Why “It’s” often appears (infographic)")
    st.write("• Top-10 next tokens (single-token view)")
    st.write("• Entropy, Perplexity, Latency")
    st.write("• Full distribution & what-if sampling (lower)")
    st.markdown('</div>', unsafe_allow_html=True)

# Button to trigger prediction
go = st.button("Predict next token")

# ---------- Helpers ----------

# ---------- Helper Functions ----------
def round_up_to_sigfigs(x: float, sigfigs: int = 2) -> float:
    """Round a number up to a given number of significant figures."""
    if x <= 0 or sigfigs <= 0: return 0.0
    exp = math.floor(math.log10(x))
    factor = 10 ** (sigfigs - 1 - exp)
    return math.ceil(x * factor) / factor


def format_probability_adaptive(p: float,
                                normal_decimals: int = 2,
                                small_cutoff_pct: float = 0.01,
                                small_sigfigs: int = 2) -> str:
    """Format probability for display, using percent or scientific notation for small values."""
    if p is None or p <= 0: return "0%"
    pct = p * 100.0
    if pct >= small_cutoff_pct:
        return f"{pct:.{normal_decimals}f}%"
    ub = round_up_to_sigfigs(pct, small_sigfigs)
    exp = math.floor(math.log10(ub)) if ub > 0 else 0
    decimals = max(0, small_sigfigs - 1 - exp)
    return f"<{ub:.{decimals}f}%"


def entropy_from_df(df):
    """Calculate entropy (in bits) from a DataFrame of probabilities."""
    ps = df["prob"].to_numpy()
    ps = ps[ps > 0]
    if len(ps) == 0: return 0.0
    return float(-np.sum(ps * np.log2(ps)))


def to_raw_glyphs(s: str) -> str:
    """Convert whitespace characters to visible glyphs for display."""
    return (s.replace("\n", "⏎").replace("\t", "↹").replace(" ", "␠"))


def format_probability_table(df_top: pd.DataFrame):
    """Display a styled probability table in Streamlit."""
    styler = df_top[["token","percent"]].style
    try:
        styler = styler.hide(axis="index")
    except Exception:
        try:
            styler = styler.hide_index()
        except Exception:
            pass
    styler = (
        styler
        .set_table_styles([
            {"selector": "th", "props": "font-weight:700; text-align:left;"},
            {"selector": "td", "props": "padding:6px 10px;"},
        ])
        .set_properties(subset=["percent"], **{"text-align": "right", "white-space": "nowrap"})
    )
    st.markdown(styler.to_html(), unsafe_allow_html=True)

# ---------- OpenAI Calls ----------
def fetch_next_token_distribution():
    """Call Responses API first; fall back to Chat Completions if needed."""
    try:
        start = time.perf_counter()
        resp = client.responses.create(
            model=model, temperature=temperature, top_p=top_p,
            max_output_tokens=1, logprobs=True,
            top_logprobs=min(top_logprobs, 20),
            input=[{"role":"system","content":system_prompt},
                   {"role":"user","content":user_prompt}],
        )
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        return ("responses", resp, elapsed_ms)
    except TypeError:
        start = time.perf_counter()
        resp = client.chat.completions.create(
            model=model, temperature=temperature, top_p=top_p,
            max_tokens=1, logprobs=True,
            top_logprobs=min(top_logprobs, 20),
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_prompt}],
        )
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        return ("chat", resp, elapsed_ms)

def parse_distribution_responses(resp):
    usage = {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    try:
        if resp.usage:
            usage = {
                "input_tokens": getattr(resp.usage, "input_tokens", None),
                "output_tokens": getattr(resp.usage, "output_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
    except Exception:
        pass
    rows = []
    try:
        choice = resp.output[0]
        tok_text = choice.content[0].text
        tp = choice.content[0].logprobs
        best_lp = tp.get("logprob", None)
        alts = tp.get("top_logprobs", []) or []
        if best_lp is not None and tok_text is not None:
            rows.append({"token": tok_text, "logprob": best_lp, "chosen": True})
        for alt in alts:
            rows.append({"token": alt.get("token",""), "logprob": alt.get("logprob", None), "chosen": False})
    except Exception:
        pass
    return finalize_rows_to_df(rows, usage)

def parse_distribution_chat(resp):
    usage = {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    try:
        u = getattr(resp, "usage", None)
        if u:
            usage = {
                "input_tokens": getattr(u, "prompt_tokens", None),
                "output_tokens": getattr(u, "completion_tokens", None),
                "total_tokens": getattr(u, "total_tokens", None),
            }
    except Exception:
        pass
    rows = []
    try:
        ch = resp.choices[0]
        chosen_text = (ch.message.content or "")
        lpinfo = getattr(ch, "logprobs", None)
        if lpinfo and getattr(lpinfo, "content", None):
            t0 = lpinfo.content[0]
            best_lp = getattr(t0, "logprob", None)
            alts = getattr(t0, "top_logprobs", None) or []
            if best_lp is not None:
                rows.append({"token": chosen_text, "logprob": best_lp, "chosen": True})
            for a in alts:
                rows.append({"token": a.token, "logprob": a.logprob, "chosen": False})
    except Exception:
        pass
    return finalize_rows_to_df(rows, usage)

def finalize_rows_to_df(rows, usage_dict):
    out = {"usage": usage_dict}
    if rows:
        df = pd.DataFrame(rows).dropna(subset=["logprob"])
        # merge duplicate tokens by max logprob
        df = df.groupby("token", as_index=False).agg({"logprob":"max", "chosen":"max"})
        df["prob"] = np.exp(df["logprob"])
        Z = df["prob"].sum()
        if Z > 0: df["prob"] = df["prob"] / Z
        df = df.sort_values("prob", ascending=False).reset_index(drop=True)
        df["prob_display"] = df["prob"].apply(lambda p: format_probability_adaptive(p))
        df["token_raw"] = df["token"].apply(to_raw_glyphs)
        H = entropy_from_df(df)
        out["df"] = df[["token","token_raw","prob","prob_display","logprob","chosen"]]
        out["entropy_bits"] = round(H, 4)
        out["perplexity"] = round(2 ** H if H > 0 else 1.0, 4)
    else:
        out["df"] = pd.DataFrame(columns=["token","token_raw","prob","prob_display","logprob","chosen"])
        out["entropy_bits"] = None
        out["perplexity"] = None
    return out

# ---------- Main ----------
if go:
    with st.spinner("Querying model…"):
        kind, resp, latency_ms = fetch_next_token_distribution()
    parsed = parse_distribution_responses(resp) if kind == "responses" else parse_distribution_chat(resp)

    # -------- INFOGRAPHIC (above Top-10) --------
    if show_explainer:
        # st.markdown("🧠 Why the AI Often Predicts Common Sentence Starters")
        st.markdown(
            """
## First... 🧠 Why the AI Often Predicts Common Sentence Starters

Before exploring the predictions, it's important to understand why certain sentence starters frequently appear. The following points explain the key factors behind these predictions.

Examples of Common Sentence Starters: *It’s*, *That*, *That’s*, *There’s*, *Here’s*, *This is*, *In the*, etc.

1. **Statistical Patterns** — These phrases frequently start sentences in the training data.  
2. **High Predictive Weight** — At sentence starts or after punctuation, these openers have high odds.  
3. **Grammar Continuation** — The model aims for fluent continuations; these starters help form full sentences.  
4. **Token-by-Token Prediction** — The model picks one token at a time, so common openers often win in the next-token slot.
---
            """,
            unsafe_allow_html=False,
        )
        st.markdown(
            """
<div class="infogrid">
    <div class="infotile" style="background:var(--c1); color:var(--c1txt);" title="Statistical Patterns">
        <h4>📊 Statistical Patterns</h4>
        <p>These phrases frequently start sentences in the training data.<br>
        The model learns to expect them after punctuation or at the beginning of a sentence.</p>
    </div>
    <div class="infotile" style="background:var(--c2); color:var(--c2txt);" title="High Predictive Weight">
        <h4>🏆 High Predictive Weight</h4>
        <p>At sentence starts or after punctuation, these openers have high odds.<br>
        The model assigns higher probabilities to tokens that commonly follow the context.</p>
    </div>
    <div class="infotile" style="background:var(--c3); color:var(--c3txt);" title="Grammar Continuation">
        <h4>📝 Grammar Continuation</h4>
        <p>The model aims for fluent continuations; these starters help form full sentences.<br>
        It prefers tokens that maintain grammatical correctness and flow.</p>
    </div>
    <div class="infotile" style="background:var(--c4); color:var(--c4txt);" title="Token-by-Token Prediction">
        <h4>� Token-by-Token Prediction</h4>
        <p>The model picks one token at a time, so common openers often win in the next-token slot.<br>
        This stepwise process explains why certain tokens are frequently chosen.</p>
    </div>
</div>
<p class="legend">Tip: Show spaces as <strong>␠</strong>, or group by visible text to avoid confusing repeats.</p>
            """,
            unsafe_allow_html=True,
        )

    # -------- TOP: Top-10 table --------
    st.markdown("## Top tokens (Top-10)  <span class='topbadge'>single token view</span>", unsafe_allow_html=True)
    merged = parsed["df"][["token","prob"]].copy().groupby("token", as_index=False)["prob"].sum()
    merged = merged.sort_values("prob", ascending=False).reset_index(drop=True)

    top10_df = pd.DataFrame({
        "token": merged["token"].head(10).apply(lambda s: to_raw_glyphs(s) if show_raw_tokens else s),
        "percent": merged["prob"].head(10).apply(lambda p: format_probability_adaptive(float(p)))
    })
    format_probability_table(top10_df)

    # -------- KPIs --------
    st.subheader("Distribution snapshot")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Entropy (bits)", parsed["entropy_bits"])
    k2.metric("Perplexity", parsed["perplexity"])
    k3.metric("Latency (ms)", latency_ms)
    usage = parsed["usage"]
    if usage and usage.get("total_tokens") is not None:
        k4.metric("Tokens (in/out/total)",
                  f'{usage.get("input_tokens","?")}/{usage.get("output_tokens","?")}/{usage.get("total_tokens","?")}')
    else:
        k4.metric("Tokens", "—")

    # -------- DETAILS: full raw distribution --------
    st.subheader("Full next-token distribution (raw candidates)")
    st.dataframe(
        parsed["df"][["token_raw","token","prob_display","logprob","chosen"]],
        use_container_width=True
    )

    # -------- What-if sampling (temperature/top-p rescale on probs) --------
    st.subheader("What-if sampling (local rescale)")
    c1, c2 = st.columns([1,1])
    with c1:
        sim_T = st.slider("Simulated Temperature", 0.0, 1.5, temperature, 0.05, key="simT")
    with c2:
        sim_top_p = st.slider("Simulated Top-p", 0.05, 1.0, top_p, 0.01, key="simP")

    df = parsed["df"].copy()
    if not df.empty:
        logits = np.log(np.clip(df["prob"].to_numpy(), 1e-12, 1))  # proxy logits from probs
        Z = logits / (sim_T if sim_T > 0 else 1e-9)
        Z = Z - Z.max()
        sim_probs = np.exp(Z); sim_probs = sim_probs / sim_probs.sum()
        order = np.argsort(sim_probs)[::-1]
        cum = 0.0; mask = np.zeros_like(sim_probs, dtype=bool)
        for idx in order:
            if cum < sim_top_p:
                mask[idx] = True; cum += sim_probs[idx]
        sim_probs = np.where(mask, sim_probs, 0.0)
        if sim_probs.sum() > 0: sim_probs = sim_probs / sim_probs.sum()

        out = df.copy()
        out["sim_prob"] = sim_probs
        out["sim_prob_display"] = out["sim_prob"].apply(lambda p: format_probability_adaptive(p))
        out = out.sort_values("sim_prob", ascending=False).reset_index(drop=True)

        st.dataframe(
            out[["token_raw","token","prob_display","sim_prob_display"]],
            use_container_width=True
        )
    else:
        st.info("No candidates returned; try another prompt or model.")

    # -------- Optional: token trail --------
    if show_generation:
        st.subheader("Short continuation (token trail)")
        st.info("No candidates returned; try another prompt or model.")

else:
    st.info("Enter a prompt and click **Predict next token**.")
