"""
SmartRep AI Sandbox — Secure AI Testing for Healthcare Messaging
================================================================
A Streamlit-based platform for testing AI use cases on patient portal messages
with built-in de-identification and simulated/live LLM capabilities.

Pages:
  1. Home — Mode selection, status overview
  2. Dataset Explorer — Browse synthetic/uploaded data by Enterprise ID
  3. Manual Input — Single-message testing with de-ID + AI analysis
  4. Batch Upload — CSV upload for bulk processing
  5. Cost Dashboard — Token & cost analysis across sessions
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

from config import AppConfig
from synthetic_data import generate_synthetic_dataset, save_synthetic_csv
from deidentification_engine import DeIdentificationEngine, DeIdentificationConfig
from ai_framework import (
    AIUseCaseFactory,
    AIUseCaseType,
    AIConfiguration,
    AIResultsManager,
    AIResult,
    TokenCostCalculator,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page Config (MUST be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SECURE AI TESTING",
    page_icon="🏥",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .patient-msg {
        background-color: #E3F2FD; padding: 12px 16px; border-radius: 12px;
        margin: 6px 0; margin-right: 15%; position: relative;
    }
    .provider-msg {
        background-color: #E8F5E9; padding: 12px 16px; border-radius: 12px;
        margin: 6px 0; margin-left: 15%; position: relative;
    }
    .caregiver-msg {
        background-color: #FFF3E0; padding: 12px 16px; border-radius: 12px;
        margin: 6px 0; margin-right: 15%; position: relative;
    }
    .deid-msg {
        background-color: #F3E5F5; padding: 12px 16px; border-radius: 12px;
        margin: 6px 0; border-left: 4px solid #9C27B0;
    }
    .msg-meta {
        font-size: 0.78em; color: #666; margin-top: 4px;
    }
    .badge {
        display: inline-block; font-size: 0.85em; border-radius: 6px;
        padding: 3px 10px; margin: 2px 4px 2px 0; font-weight: 600;
    }
    .badge-author   { background: #f4b6c2; color: #22223b; }
    .badge-crit     { background: #b5ead7; color: #22223b; }
    .badge-category { background: #ffd3a5; color: #22223b; }
    .badge-handler  { background: #d0d1ff; color: #22223b; }
    .draft-box {
        background: #fff3cd; border: 1.5px solid #ffe066; border-radius: 10px;
        padding: 12px 16px; margin-top: 8px;
    }
    .scenario-end {
        text-align: center; color: #999; font-style: italic;
        margin: 20px 0; padding: 10px; border-top: 1px solid #eee;
    }
    .cost-card {
        background: #FFF8E1; padding: 10px 14px; border-radius: 8px;
        margin: 8px 0; font-weight: 500;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "page": "home",
        "app_config": AppConfig.from_env(),
        "simulation_data": None,
        "chat_history": [],
        "processed_batch": None,
        "all_session_costs": [],
        "selected_enterprise_id": None,
        "custom_prompts": {},  # {use_case_type_value: prompt_string}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Load synthetic data on first run
    if st.session_state.simulation_data is None:
        csv_path = st.session_state.app_config.synthetic_data_path
        if not Path(csv_path).exists():
            save_synthetic_csv(csv_path)
        st.session_state.simulation_data = pd.read_csv(csv_path)


init_state()


# ──────────────────────────────────────────────────────────────────────────────
# Cached resources
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_deid_engine():
    cfg = DeIdentificationConfig(
        detect_names=True,
        detect_emails=True,
        detect_phones=True,
        detect_addresses=True,
        preserve_gender=True,
        shift_dates_by_days=-30,
        use_spacy=False,
        use_scispacy=False,
    )
    return DeIdentificationEngine(cfg)


@st.cache_resource
def get_ai_use_cases(_mode: str):
    """Create all four AI use case instances."""
    cases = {}
    for uc_type in AIUseCaseType:
        cases[uc_type.value] = AIUseCaseFactory.create_use_case(
            use_case_type=uc_type,
            mode=_mode,
        )
    return cases


def _ai_cases() -> dict:
    return get_ai_use_cases(st.session_state.app_config.mode)


deid_engine = get_deid_engine()


# ──────────────────────────────────────────────────────────────────────────────
# Navigation helpers
# ──────────────────────────────────────────────────────────────────────────────
def nav_to(page: str):
    st.session_state.page = page
    st.rerun()


def render_nav():
    """Top navigation bar."""

    st.markdown(
        "# SECURE AI SANDBOX FOR PATIENT PORTAL MESSAGING", unsafe_allow_html=True
    )
    cols = st.columns([1, 1, 1, 1, 1, 1])
    # with cols[0]:
    #     st.markdown("### Secure AI Sandbox for Portal Messaging")
    with cols[0]:
        if st.button("Home", use_container_width=True):
            nav_to("home")
    with cols[1]:
        if st.button("Dataset Explorer", use_container_width=True):
            nav_to("dataset")
    with cols[2]:
        if st.button("Manual Input", use_container_width=True):
            nav_to("manual")
    with cols[3]:
        if st.button("Batch Upload", use_container_width=True):
            nav_to("batch")
    with cols[4]:
        if st.button("Prompt Studio", use_container_width=True):
            nav_to("prompts")
    with cols[5]:
        if st.button("Cost Dashboard", use_container_width=True):
            nav_to("costs")
    st.markdown("---")


# ──────────────────────────────────────────────────────────────────────────────
# Utility: Run AI on a single message dict
# ──────────────────────────────────────────────────────────────────────────────
def run_all_ai(msg_data: dict, use_cases: list[AIUseCaseType] | None = None) -> dict:
    """Run selected AI use cases on a message. Returns dict of results."""
    targets = use_cases or list(AIUseCaseType)
    cases = _ai_cases()
    custom_prompts = st.session_state.get("custom_prompts", {})
    for uc_type in targets:
        case = cases[uc_type.value]
        custom = custom_prompts.get(uc_type.value)
        case.set_custom_prompt(custom)  # None resets to default

    results = {}
    for uc_type in targets:
        case = cases[uc_type.value]
        r = case.process_message(msg_data)
        results[uc_type.value] = {
            "result": r.result,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "total_cost": r.total_cost,
            "error": r.error,
        }
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Utility: Render AI badges
# ──────────────────────────────────────────────────────────────────────────────
def render_badges(ai_results: dict):
    """Display coloured badges for AI analysis results."""
    html_parts = []
    if (
        "authorship_detection" in ai_results
        and ai_results["authorship_detection"]["result"]
    ):
        label = ai_results["authorship_detection"]["result"].get("label", "?")
        html_parts.append(
            f'<span class="badge badge-author">👤 Author: {label.capitalize()}</span>'
        )

    if (
        "criticality_analysis" in ai_results
        and ai_results["criticality_analysis"]["result"]
    ):
        cl = ai_results["criticality_analysis"]["result"].get("criticality_label", "?")
        cs = ai_results["criticality_analysis"]["result"].get("criticality_score", "?")
        html_parts.append(
            f'<span class="badge badge-crit">🚨 Criticality: {cl} ({cs}/5)</span>'
        )

    if "categorization" in ai_results and ai_results["categorization"]["result"]:
        cat = ai_results["categorization"]["result"].get("category_label", "?")
        html_parts.append(f'<span class="badge badge-category">📂 {cat}</span>')
        handler = ai_results["categorization"]["result"].get("recommended_handler", "")
        if handler:
            html_parts.append(f'<span class="badge badge-handler">➡️ {handler}</span>')

    if html_parts:
        st.markdown(" ".join(html_parts), unsafe_allow_html=True)


def render_draft_response(ai_results: dict):
    """Show generated draft response if available."""
    if (
        "response_generation" in ai_results
        and ai_results["response_generation"]["result"]
    ):
        resp = ai_results["response_generation"]["result"].get("generated_response", "")
        if resp:
            st.markdown(
                f'<div class="draft-box"><strong>💬 AI Draft Response</strong><br>{resp}</div>',
                unsafe_allow_html=True,
            )


def render_cost_details(ai_results: dict):
    """Show cost breakdown in an expander."""
    total_cost = 0
    total_tokens = 0
    lines = []
    uc_labels = {
        "authorship_detection": "Author Detection",
        "criticality_analysis": "Criticality Analysis",
        "categorization": "Categorization",
        "response_generation": "Draft Response",
    }
    for uc_key, label in uc_labels.items():
        if uc_key in ai_results:
            r = ai_results[uc_key]
            inp = r.get("input_tokens", 0)
            out = r.get("output_tokens", 0)
            cost = r.get("total_cost", 0)
            total_tokens += inp + out
            total_cost += cost
            lines.append(f"**{label}:** {inp}+{out} tokens = ${cost:.4f}")

    with st.expander("💰 Cost Details"):
        for line in lines:
            st.write(line)
        st.markdown("---")
        st.write(f"**Total:** {total_tokens} tokens, **${total_cost:.4f}**")
    return total_cost


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Home
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("## WELCOME TO THE AI SANDBOX")
    st.markdown(
        "A **secure sandbox** for testing AI use cases on patient portal messages — "
        "de-identification, authorship detection, criticality analysis, categorization, "
        "and AI-assisted response drafting."
    )

    mode = st.session_state.app_config.mode
    mode_color = "#4CAF50" if mode == "simulation" else "#FF9800"
    st.markdown(
        f"**Current Mode:** <span style='color:{mode_color}; font-weight:bold'>"
        f"{mode.upper()}</span> &nbsp; "
        f"{'(Mock AI — no API key needed)' if mode == 'simulation' else '(Live Azure OpenAI)'}",
        unsafe_allow_html=True,
    )
    st.info(
        "Running in **simulation mode** with synthetic data. "
        "AI responses are realistic mocks — no API key required. "
        "Switch to live mode by setting `SMARTREP_MODE=live` and providing Azure credentials as environment variables."
    )

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Synthetic Messages", len(st.session_state.simulation_data))
    with c2:
        n_ent = st.session_state.simulation_data["Enterprise_ID"].nunique()
        st.metric("Enterprise IDs", n_ent)
    with c3:
        st.metric("AI Use Cases", 4)
    with c4:
        st.metric("Mode", mode.capitalize())

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📋 Dataset Explorer", use_container_width=True, type="primary"):
            nav_to("dataset")
        st.caption(
            "Browse messages by Enterprise ID with de-identification & AI analysis"
        )
    with col2:
        if st.button("✏️ Manual Input", use_container_width=True, type="primary"):
            nav_to("manual")
        st.caption("Type or paste a single message for quick testing")
    with col3:
        if st.button("📤 Batch Upload", use_container_width=True, type="primary"):
            nav_to("batch")
        st.caption("Upload CSV/Excel for bulk AI processing")
    with col4:
        if st.button("🛠️ Prompt Studio", use_container_width=True, type="primary"):
            nav_to("prompts")
        st.caption("View, edit, and experiment with AI prompt templates")

    st.markdown("---")
    st.markdown("#### AI Use Cases Available")
    uc_info = {
        "👤 Authorship Detection": "Classifies if a message was written by the **patient** or a **care partner**",
        "🚨 Criticality Analysis": "Rates message urgency on a **1–5 scale** (Routine → Urgent)",
        "📂 Categorization": "Classifies as **Clinical** vs **Non-Clinical** with handler routing",
        "💬 Response Generation": "Drafts a professional **provider response** to patient messages",
    }
    cols = st.columns(2)
    for i, (title, desc) in enumerate(uc_info.items()):
        with cols[i % 2]:
            st.markdown(f"**{title}**")
            st.markdown(desc)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Dataset Explorer
# ══════════════════════════════════════════════════════════════════════════════
def page_dataset():
    st.markdown("## 📋 Dataset Explorer")

    df = st.session_state.simulation_data

    # Sidebar: Enterprise ID picker + options
    with st.sidebar:
        st.header("Filters")
        ent_counts = df.groupby("Enterprise_ID").size().to_dict()
        ent_options = [f"{eid} ({cnt} msgs)" for eid, cnt in ent_counts.items()]
        ent_ids = list(ent_counts.keys())
        sel = st.selectbox("Enterprise ID", ent_options)
        if sel:
            selected_eid = ent_ids[ent_options.index(sel)]
        else:
            selected_eid = ent_ids[0]

        st.markdown("---")
        do_deid = st.checkbox("🔒 De-identify PHI", value=True)
        do_ai = st.checkbox("🤖 Run AI Analysis", value=True)

        ai_use_cases = []
        if do_ai:
            st.markdown("**Select Use Cases:**")
            if st.checkbox("Authorship Detection", value=True, key="ds_auth"):
                ai_use_cases.append(AIUseCaseType.AUTHORSHIP_DETECTION)
            if st.checkbox("Criticality Analysis", value=True, key="ds_crit"):
                ai_use_cases.append(AIUseCaseType.CRITICALITY_ANALYSIS)
            if st.checkbox("Categorization", value=True, key="ds_cat"):
                ai_use_cases.append(AIUseCaseType.CATEGORIZATION)
            if st.checkbox("Response Generation", value=False, key="ds_resp"):
                ai_use_cases.append(AIUseCaseType.RESPONSE_GENERATION)

    # Filter data
    filtered = df[df["Enterprise_ID"] == selected_eid].sort_values(
        ["Session_Number", "Message_Order"]
    )

    st.info(
        f"Enterprise **{selected_eid}** — {len(filtered)} messages across "
        f"{filtered['Session_Number'].nunique()} sessions"
    )

    # Two-column layout: conventional vs AI-augmented
    if do_ai and ai_use_cases:
        col_conv, col_ai = st.columns(2)
    else:
        col_conv = st.container()
        col_ai = None

    # Conventional column
    with col_conv:
        title = "📄 Conventional View"
        if do_deid:
            title += " (De-identified)"
        st.subheader(title)

        for _, row in filtered.iterrows():
            sender = row["Sender"]
            text = row["Email_Text"]

            if do_deid:
                text, _ = deid_engine.deidentify_text(text)

            css_class = (
                "patient-msg"
                if sender == "PATIENT"
                else "provider-msg" if sender == "PROVIDER" else "caregiver-msg"
            )
            label = sender.capitalize()
            st.markdown(
                f'<div class="{css_class}"><strong>{label}</strong> '
                f'<span class="msg-meta">({row["Sender_Name"]} · Session {row["Session_Number"]} · '
                f'Msg {row["Message_Order"]})</span><br>{text}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="scenario-end">End of scenario</div>', unsafe_allow_html=True
        )

    # AI-augmented column
    if col_ai is not None:
        with col_ai:
            st.subheader("🤖 AI-Augmented View")

            patient_msgs = filtered[filtered["Sender"].isin(["PATIENT", "CAREGIVER"])]
            if patient_msgs.empty:
                st.info("No patient/caregiver messages to analyze.")
            else:
                for _, row in patient_msgs.iterrows():
                    text = row["Email_Text"]
                    if do_deid:
                        text, _ = deid_engine.deidentify_text(text)

                    css_class = (
                        "patient-msg" if row["Sender"] == "PATIENT" else "caregiver-msg"
                    )
                    st.markdown(
                        f'<div class="{css_class}"><strong>{row["Sender"].capitalize()}</strong> '
                        f'<span class="msg-meta">(Session {row["Session_Number"]})</span><br>{text}</div>',
                        unsafe_allow_html=True,
                    )

                    msg_data = {
                        "message_id": row["Message_ID"],
                        "email_text": text,
                        "sender": row["Sender"],
                        "message_type": row["Message_Type"],
                    }
                    ai_results = run_all_ai(msg_data, ai_use_cases)
                    render_badges(ai_results)
                    render_draft_response(ai_results)
                    render_cost_details(ai_results)
                    st.markdown("")

                st.markdown(
                    '<div class="scenario-end">End of AI analysis</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Manual Input
# ══════════════════════════════════════════════════════════════════════════════
def page_manual():
    st.markdown("## ✏️ Manual Message Testing")

    # Reset button
    col_r, col_s = st.columns([1, 7])
    with col_r:
        if st.button("🔄 Reset"):
            st.session_state.chat_history = []
            st.rerun()

    # Display chat history
    for idx, chat in enumerate(st.session_state.chat_history):
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown(
                f'<div class="patient-msg"><strong>Original Message</strong><br>{chat["original"]}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="deid-msg"><strong>🔒 De-Identified</strong><br>{chat["deidentified"]}</div>',
                unsafe_allow_html=True,
            )
            if chat.get("entities_found", 0) > 0:
                st.caption(f"🔍 {chat['entities_found']} PHI entities replaced")

            # AI Analysis
            if not chat.get("ai_done"):
                if st.button("🤖 Analyze Message", key=f"analyze_{idx}"):
                    msg_data = {
                        "message_id": f"manual_{idx+1}",
                        "email_text": chat["original"],
                        "sender": "PATIENT",
                        "message_type": "Patient Medical Advice Request",
                    }
                    # Run analysis (not response gen yet)
                    analysis_ucs = [
                        AIUseCaseType.AUTHORSHIP_DETECTION,
                        AIUseCaseType.CRITICALITY_ANALYSIS,
                        AIUseCaseType.CATEGORIZATION,
                    ]
                    results = run_all_ai(msg_data, analysis_ucs)
                    st.session_state.chat_history[idx]["ai_results"] = results
                    st.session_state.chat_history[idx]["ai_done"] = True
                    st.rerun()
            else:
                ai_results = chat.get("ai_results", {})
                render_badges(ai_results)

                # Generate response button
                if not chat.get("response_done"):
                    if st.button("💬 Generate Draft Response", key=f"draft_{idx}"):
                        msg_data = {
                            "message_id": f"manual_{idx+1}",
                            "email_text": chat["original"],
                            "sender": "PATIENT",
                            "message_type": "Patient Medical Advice Request",
                        }
                        resp_results = run_all_ai(
                            msg_data, [AIUseCaseType.RESPONSE_GENERATION]
                        )
                        ai_results.update(resp_results)
                        st.session_state.chat_history[idx]["ai_results"] = ai_results
                        st.session_state.chat_history[idx]["response_done"] = True
                        st.rerun()
                else:
                    render_draft_response(ai_results)

                cost = render_cost_details(ai_results)
                # Track for dashboard
                if chat.get("cost_tracked") is None:
                    st.session_state.all_session_costs.append(
                        {
                            "source": "manual",
                            "timestamp": datetime.now().isoformat(),
                            "cost": cost,
                        }
                    )
                    st.session_state.chat_history[idx]["cost_tracked"] = True

    # ── Input area ──
    st.markdown("---")

    # Example messages
    examples = {
        "Clinical – Memory Confusion (Caregiver)": "My husband Robert Johnson with dementia has been very confused and agitated today. "
        "He doesn't recognize me and keeps asking for his mother. Dr. Smith said this might happen. "
        "Is this normal progression? You can reach me at (312) 123-4567 if needed.",
        "Clinical – Medication (Caregiver)": "Mom Sarah Martinez keeps forgetting to take her Aricept and sometimes takes it twice. "
        "I'm worried about overdose. Dr. Williams prescribed it last month. Should I get a pill organizer? "
        "My contact is jane.martinez@email.com or call (410) 987-6543.",
        "Clinical – Behavioral Emergency": "Dad William Thompson at 123 Oak Street, Springfield is having severe sundowning episodes. "
        "He's trying to leave the house at night and becoming aggressive when we stop him. "
        "Dr. Johnson recommended we contact you. Call me at (312) 456-7890.",
        "Non-Clinical – Appointment": "I need to reschedule my father David Wilson's neurology appointment with Dr. Thompson next week. "
        "He had a fall and we're dealing with that right now. Please call me at (312) 345-6789 "
        "or email michael.wilson@gmail.com.",
        "Clinical – Patient Self-Report": "I've been having more trouble remembering things lately and I forgot to take my Namenda yesterday. "
        "I'm worried about my memory getting worse. Dr. Anderson said to contact you if things changed. "
        "Please call me at (410) 234-5678.",
    }

    with st.expander("📝 Example Messages (click to expand, then copy-paste)"):
        for label, msg in examples.items():
            st.markdown(f"**{label}:**")
            st.code(msg, language=None)

    with st.form("msg_form", clear_on_submit=True):
        user_msg = st.text_area(
            "Enter Patient Message",
            height=100,
            placeholder="Type or paste a message here...",
        )
        submitted = st.form_submit_button("Send", type="primary")

    if submitted and user_msg.strip():
        deid_text, entities = deid_engine.deidentify_text(user_msg)
        st.session_state.chat_history.append(
            {
                "original": user_msg,
                "deidentified": deid_text,
                "entities_found": len(entities),
                "ai_done": False,
                "response_done": False,
                "ai_results": {},
                "cost_tracked": None,
            }
        )
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Batch Upload
# ══════════════════════════════════════════════════════════════════════════════
def page_batch():
    st.markdown("## 📤 Batch Upload & Processing")

    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    if uploaded is not None:
        try:
            df = (
                pd.read_csv(uploaded)
                if uploaded.name.endswith(".csv")
                else pd.read_excel(uploaded)
            )
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

        st.success(f"Loaded **{len(df)}** rows from `{uploaded.name}`")
        st.dataframe(df.head(5), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            msg_col = st.selectbox("Message column", df.columns.tolist())
        with c2:
            id_col = st.selectbox(
                "ID column (optional)", ["Auto-generate"] + df.columns.tolist()
            )

        if st.button(
            "🚀 Process All Messages", type="primary", use_container_width=True
        ):
            results = []
            progress = st.progress(0)
            status = st.empty()

            for idx, row in df.iterrows():
                progress.progress((idx + 1) / len(df))
                status.text(f"Processing message {idx + 1} / {len(df)}")

                text = str(row[msg_col])
                mid = (
                    str(row[id_col]) if id_col != "Auto-generate" else f"batch_{idx+1}"
                )

                deid_text, entities = deid_engine.deidentify_text(text)
                msg_data = {
                    "message_id": mid,
                    "email_text": text,
                    "sender": "PATIENT",
                    "message_type": "Patient Medical Advice Request",
                }
                ai_res = run_all_ai(msg_data)

                def _get(uc, key, default=""):
                    return (
                        ai_res.get(uc, {}).get("result", {}).get(key, default)
                        if ai_res.get(uc, {}).get("result")
                        else default
                    )

                total_cost = sum(ai_res[uc].get("total_cost", 0) for uc in ai_res)

                results.append(
                    {
                        "message_id": mid,
                        "original": text,
                        "deidentified": deid_text,
                        "entities_found": len(entities),
                        "author": _get("authorship_detection", "label"),
                        "criticality": _get(
                            "criticality_analysis", "criticality_label"
                        ),
                        "criticality_score": _get(
                            "criticality_analysis", "criticality_score", 0
                        ),
                        "category": _get("categorization", "category_label"),
                        "handler": _get("categorization", "recommended_handler"),
                        "draft_response": _get(
                            "response_generation", "generated_response"
                        ),
                        "total_cost": total_cost,
                    }
                )

            progress.empty()
            status.empty()

            results_df = pd.DataFrame(results)
            st.session_state.processed_batch = results_df

            # Track costs
            batch_cost = results_df["total_cost"].sum()
            st.session_state.all_session_costs.append(
                {
                    "source": "batch",
                    "timestamp": datetime.now().isoformat(),
                    "cost": batch_cost,
                    "messages": len(results),
                }
            )

            # Summary
            st.success(
                f"Done! Processed **{len(results)}** messages. Total cost: **${batch_cost:.4f}**"
            )

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Messages", len(results))
            with c2:
                st.metric("Total Cost", f"${batch_cost:.4f}")
            with c3:
                clinical = sum(1 for r in results if r["category"] == "Clinical")
                st.metric("Clinical", clinical)
            with c4:
                urgent = sum(1 for r in results if r["criticality_score"] >= 4)
                st.metric("Urgent (4+)", urgent)

            st.markdown("---")
            st.dataframe(results_df, use_container_width=True)

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results CSV",
                csv_bytes,
                file_name=f"smartrep_results_{datetime.now():%Y%m%d_%H%M%S}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    elif st.session_state.processed_batch is not None:
        st.markdown("### Previous Batch Results")
        st.dataframe(st.session_state.processed_batch, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Cost Dashboard
# ══════════════════════════════════════════════════════════════════════════════
def page_costs():
    st.markdown("## 💰 Cost Dashboard")

    costs = st.session_state.all_session_costs
    if not costs:
        st.info(
            "No cost data yet. Process messages in Manual Input or Batch Upload to see analytics here."
        )
        return

    total = sum(c["cost"] for c in costs)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Sessions", len(costs))
    with c2:
        st.metric("Total Cost", f"${total:.4f}")
    with c3:
        st.metric("Avg Cost / Session", f"${total / len(costs):.4f}")

    st.markdown("---")
    cost_df = pd.DataFrame(costs)
    st.dataframe(cost_df, use_container_width=True)

    if len(costs) > 1:
        import plotly.express as px

        cost_df["timestamp"] = pd.to_datetime(cost_df["timestamp"])
        fig = px.bar(
            cost_df,
            x="timestamp",
            y="cost",
            color="source",
            title="Cost Over Time",
            labels={"cost": "Cost ($)", "timestamp": "Time"},
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Prompt Studio
# ══════════════════════════════════════════════════════════════════════════════
def page_prompts():
    st.markdown("## 🛠️ Prompt Studio")
    st.markdown(
        "View, edit, and experiment with the prompt templates used by each AI use case. "
        "Changes take effect immediately on Manual Input and Batch Upload workflows. "
        "The `{message}` placeholder is replaced with the actual patient message at runtime."
    )

    uc_meta = {
        AIUseCaseType.AUTHORSHIP_DETECTION: {
            "icon": "👤",
            "title": "Authorship Detection",
            "desc": "Classifies if a message was written by the patient or a care partner. "
            "Expected output: 0 (patient), 1 (caretaker), 2 (ambiguous).",
        },
        AIUseCaseType.CRITICALITY_ANALYSIS: {
            "icon": "🚨",
            "title": "Criticality Analysis",
            "desc": "Rates message urgency on a 1–5 scale. "
            "Expected output: single integer 1-5.",
        },
        AIUseCaseType.CATEGORIZATION: {
            "icon": "📂",
            "title": "Message Categorization",
            "desc": "Classifies as Clinical (1) or Non-Clinical (0). "
            "Expected output: 0 or 1.",
        },
        AIUseCaseType.RESPONSE_GENERATION: {
            "icon": "💬",
            "title": "Response Generation",
            "desc": "Generates a professional draft provider response. "
            "Expected output: free-text reply.",
        },
    }

    custom_prompts = st.session_state.get("custom_prompts", {})
    cases = _ai_cases()

    for uc_type, meta in uc_meta.items():
        case = cases[uc_type.value]
        default_prompt = case.prompt_template
        current_prompt = custom_prompts.get(uc_type.value, default_prompt)
        is_modified = uc_type.value in custom_prompts

        with st.expander(
            f"{meta['icon']}  {meta['title']}"
            f"{'  ✏️ (modified)' if is_modified else ''}",
            expanded=False,
        ):
            st.caption(meta["desc"])

            # Show prompt in editable text area
            edited = st.text_area(
                f"Prompt template for {meta['title']}",
                value=current_prompt,
                height=300,
                key=f"prompt_edit_{uc_type.value}",
                help="Use {message} as the placeholder for the patient message text.",
            )

            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button("💾 Save", key=f"save_{uc_type.value}", type="primary"):
                    if edited.strip() != default_prompt.strip():
                        st.session_state.custom_prompts[uc_type.value] = edited
                        st.success(
                            "Custom prompt saved! It will be used in the next AI run."
                        )
                    else:
                        # User reverted to default
                        st.session_state.custom_prompts.pop(uc_type.value, None)
                        st.info("Prompt matches default — using built-in template.")
                    st.rerun()

            with col2:
                if st.button("↩️ Reset", key=f"reset_{uc_type.value}"):
                    st.session_state.custom_prompts.pop(uc_type.value, None)
                    st.info("Reset to default prompt.")
                    st.rerun()

            # Preview with sample message
            st.markdown("---")
            st.markdown("**Preview with sample message:**")
            sample_text = "My husband has been very confused and agitated today. He doesn't recognize me. Is this normal?"
            try:
                preview = edited.format(message=sample_text)
                st.code(preview, language=None)
                token_est = len(preview) // 4
                st.caption(f"Estimated tokens: ~{token_est}")
            except (KeyError, IndexError):
                st.error("⚠️ Prompt must contain `{message}` placeholder.")

    # Summary
    st.markdown("---")
    n_modified = len(st.session_state.get("custom_prompts", {}))
    if n_modified > 0:
        modified_names = []
        for uc_val in st.session_state.custom_prompts:
            for uc_type, meta in uc_meta.items():
                if uc_type.value == uc_val:
                    modified_names.append(meta["title"])
        st.warning(
            f"**{n_modified} custom prompt(s) active:** {', '.join(modified_names)}"
        )
        if st.button("↩️ Reset All to Defaults"):
            st.session_state.custom_prompts = {}
            st.success("All prompts reset to defaults.")
            st.rerun()
    else:
        st.success("All prompts are using built-in defaults.")


# ══════════════════════════════════════════════════════════════════════════════
# Main Router
# ══════════════════════════════════════════════════════════════════════════════
def main():
    render_nav()
    page = st.session_state.page

    if page == "home":
        page_home()
    elif page == "dataset":
        page_dataset()
    elif page == "manual":
        page_manual()
    elif page == "batch":
        page_batch()
    elif page == "prompts":
        page_prompts()
    elif page == "costs":
        page_costs()
    else:
        page_home()


if __name__ == "__main__":
    main()
