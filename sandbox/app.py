"""
Redrob Hackathon - Candidate Evaluation Dashboard

Optimize candidate alignments with tiered skill scoring, career trajectory validation, 
and platform telemetry modifiers.

Features:
- Upload JSON/JSONL candidate files
- Advanced filtering (title, experience, skills, location, score)
- Detailed candidate profile inspection
- Real-time scoring visualization
"""

import streamlit as st
import json
import csv
import io
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

# Import from your actual project structure
from src.loader import flatten
from src.hard_filter import passes_hard_filter
from src.skill_matcher import skill_score
from src.career_matcher import career_score
from src.behavioral_scorer import behavioral_score
from src.ranking_engine import compute_score
from src.reasoning_generator import generate_reasoning


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Candidate Evaluation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Apply font globally */
html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif;
}

/* Gradient Hero Banner */
.hero-header {
    background: linear-gradient(135deg, #8b5cf6 0%, #4f46e5 100%);
    padding: 30px;
    border-radius: 12px;
    margin-bottom: 25px;
    box-shadow: 0 4px 20px rgba(79, 70, 229, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.hero-title {
    color: #ffffff !important;
    font-weight: 700;
    font-size: 2.2rem;
    margin: 0;
}
.hero-subtitle {
    color: #e0e7ff !important;
    font-size: 1rem;
    margin-top: 5px;
    opacity: 0.9;
}

/* Custom Metric Cards */
.metric-card {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.metric-card:hover {
    transform: translateY(-3px);
    background: rgba(255, 255, 255, 0.04);
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.1);
}
.metric-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.metric-val {
    font-size: 2rem;
    font-weight: 700;
    margin-top: 8px;
}

/* Profile Inspector Card */
.profile-card {
    background: rgba(255, 255, 255, 0.01);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}
.profile-title {
    font-weight: 600;
    font-size: 1.3rem;
    color: #818cf8;
    margin-bottom: 5px;
}
.profile-meta {
    font-size: 0.88rem;
    color: #9ca3af;
    margin-bottom: 15px;
}
.score-badge {
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.25);
    color: #a5b4fc;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 0.85rem;
}

/* Badge styles */
.badge-tech {
    background: rgba(99, 102, 241, 0.12);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.25);
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 500;
    margin-right: 6px;
    margin-bottom: 6px;
    display: inline-block;
    white-space: nowrap;
}

/* Section header */
.section-header {
    font-size: 1.2rem;
    font-weight: 600;
    border-bottom: 2px solid rgba(99, 102, 241, 0.2);
    padding-bottom: 6px;
    margin-bottom: 15px;
    color: #e2e8f0;
}

/* Table styling */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}

/* Button styling */
div.stButton > button {
    background: linear-gradient(135deg, #8b5cf6 0%, #4f46e5 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    box-shadow: 0 3px 10px rgba(79, 70, 229, 0.3) !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 15px rgba(99, 102, 241, 0.5) !important;
}

/* File uploader styling */
.uploaded-file {
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 8px;
    padding: 10px 15px;
    margin-bottom: 10px;
}
.uploaded-file-name {
    font-weight: 500;
    color: #e2e8f0;
}
.uploaded-file-size {
    color: #9ca3af;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
        
    # File upload with custom styling
    st.markdown("### 📁 Upload File")
    uploaded = st.file_uploader(
        "Upload Candidate Profiles",
        type=["json", "jsonl"],
        help="Upload JSON or JSONL file with candidate data",
        label_visibility="collapsed"
    )
    
    # Show file info if uploaded
    if uploaded:
        file_size = uploaded.size / 1024
        st.markdown(f"""
        <div class="uploaded-file">
            <div class="uploaded-file-name">📄 {uploaded.name}</div>
            <div class="uploaded-file-size">{file_size:.1f} KB</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Top N selection
    st.markdown("### 📊 Display Options")
    top_n = st.slider(
        "Top Candidates to Show",
        min_value=10,
        max_value=100,
        value=20,
        step=10,
        help="Number of top candidates to display"
    )
    
    st.divider()
    
    # Filter options
    st.markdown("### 🔍 Filters")
    
    with st.expander("🎯 Title Filters", expanded=False):
        title_contains = st.text_input(
            "Title contains",
            placeholder="e.g., ML Engineer",
            help="Filter candidates by title keyword"
        )
        
        exclude_titles = st.multiselect(
            "Exclude Titles",
            options=[
                "HR Manager", "Marketing Manager", "Operations Manager",
                "Customer Support", "Accountant", "Graphic Designer",
                "Content Writer", "Civil Engineer", "Mechanical Engineer",
                "Business Analyst", "Project Manager", "Sales Executive"
            ],
            help="Exclude candidates with these titles"
        )
    
    with st.expander("📅 Experience Filters", expanded=False):
        min_yoe = st.slider(
            "Minimum Years",
            min_value=0,
            max_value=15,
            value=0,
            step=1
        )
        
        max_yoe = st.slider(
            "Maximum Years",
            min_value=0,
            max_value=20,
            value=20,
            step=1
        )
    
    with st.expander("📍 Location Filters", expanded=False):
        location_filter = st.multiselect(
            "Preferred Locations",
            options=[
                "Pune", "Noida", "Mumbai", "Delhi", "Bangalore",
                "Hyderabad", "Chennai", "Kolkata", "Ahmedabad",
                "Gurgaon", "Chandigarh", "Indore", "Kochi"
            ],
            help="Filter candidates by location"
        )
        
        only_india = st.checkbox("Only India-based", value=False)
        willing_relocate = st.checkbox("Willing to Relocate", value=False)
    
    with st.expander("📊 Score Filters", expanded=False):
        min_score = st.slider(
            "Minimum Score",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            format="%.2f"
        )
        
        min_skill_score = st.slider(
            "Min Skill Score",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05
        )
    
    with st.expander("📈 Behavioral Filters", expanded=False):
        min_response_rate = st.slider(
            "Min Response Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            format="%.2f"
        )
        
        max_notice_days = st.slider(
            "Max Notice Days",
            min_value=0,
            max_value=180,
            value=180,
            step=10
        )
        
        open_to_work_only = st.checkbox("Only Open to Work", value=False)
    
    st.divider()
    
    # Reset button - properly clears session state
    if st.button("🔄 Reset", use_container_width=True):
        # Clear all session state keys
        keys_to_clear = ['rows', 'eligible_count', 'filtered_count', 'anomaly_count', 
                        'candidates_count', 'total_data_count', 'sample_uploaded', 
                        'load_sample', 'min_years_override']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero-header">
    <div class="hero-title"> Candidate Evaluation Dashboard</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MAIN LOGIC
# ============================================================

def load_candidates_from_upload(uploaded_file):
    """Load candidates from uploaded file."""
    raw_bytes = uploaded_file.read()
    data = []
    
    try:
        # Try JSON array
        parsed = json.loads(raw_bytes.decode("utf-8"))
        if isinstance(parsed, dict):
            data = [parsed]
        else:
            data = parsed
    except json.JSONDecodeError:
        # Try JSONL
        for line in raw_bytes.decode("utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except:
                    pass
    
    return data


def apply_filters(candidates, filters):
    """Apply all filters to candidates."""
    filtered = []
    
    for c in candidates:
        # Title filter
        if filters.get("title_contains"):
            title = c.get("current_title", "").lower()
            if filters["title_contains"].lower() not in title:
                continue
        
        # Exclude titles
        if filters.get("exclude_titles"):
            title = c.get("current_title", "")
            if title in filters["exclude_titles"]:
                continue
        
        # Experience range
        yoe = c.get("yoe", 0)
        if yoe < filters.get("min_yoe", 0) or yoe > filters.get("max_yoe", 99):
            continue
        
        # Location filter
        if filters.get("location_filter"):
            location = c.get("location", "")
            if not any(loc.lower() in location.lower() for loc in filters["location_filter"]):
                continue
        
        # India only
        if filters.get("only_india"):
            country = c.get("country", "").lower()
            if country != "india":
                continue
        
        # Willing to relocate
        if filters.get("willing_relocate"):
            if not c.get("willing_relocate", False):
                continue
        
        # Score filter
        if filters.get("min_score", 0) > 0:
            score = c.get("_final_score", 0)
            if score < filters["min_score"]:
                continue
        
        # Skill score filter
        if filters.get("min_skill_score", 0) > 0:
            skill_score_val = c.get("_skill_score", 0)
            if skill_score_val < filters["min_skill_score"]:
                continue
        
        # Response rate
        if filters.get("min_response_rate", 0) > 0:
            response_rate = c.get("response_rate", 0)
            if response_rate < filters["min_response_rate"]:
                continue
        
        # Notice days
        if filters.get("max_notice_days", 180) < 180:
            notice_days = c.get("notice_days", 0)
            if notice_days > filters["max_notice_days"]:
                continue
        
        # Open to work
        if filters.get("open_to_work_only"):
            if not c.get("open_to_work", False):
                continue
        
        filtered.append(c)
    
    return filtered


# ============================================================
# PROCESS CANDIDATES
# ============================================================

# Check for uploaded file (removed sample load)
uploaded = uploaded

if uploaded:
    with st.spinner("Loading and processing candidates..."):
        # Load data
        data = load_candidates_from_upload(uploaded)
        
        if not data:
            st.error("❌ Failed to parse uploaded file. Please check the format.")
            st.stop()
        
        # Show file info
        st.success(f"✅ Dataset successfully loaded. Identified {len(data)} candidate profiles.")
        
        # Limit for performance
        if len(data) > 100:
            st.warning(f"⚠️ Preview optimized for up to 100 profiles. Truncating dataset from {len(data)}.")
            data = data[:100]
        
        # Flatten candidates
        candidates = []
        for c in data:
            try:
                flat = flatten(c)
                candidates.append(flat)
            except Exception as e:
                st.warning(f"Failed to flatten candidate: {e}")
        
        # Hard filter
        eligible = [c for c in candidates if passes_hard_filter(c)]
        
        # Score candidates
        for c in eligible:
            try:
                # Calculate individual scores
                skill = skill_score(c)
                career = career_score(c)
                behavior = behavioral_score(c)
                
                c["_skill_score"] = skill
                c["_career_score"] = career
                c["_behavior_score"] = behavior
                
                compute_score(c)
            except Exception as e:
                c["_final_score"] = 0.01
        
        # Apply filters
        filters = {
            "title_contains": title_contains if title_contains else None,
            "exclude_titles": exclude_titles,
            "min_yoe": min_yoe,
            "max_yoe": max_yoe,
            "location_filter": location_filter,
            "only_india": only_india,
            "willing_relocate": willing_relocate,
            "min_score": min_score,
            "min_skill_score": min_skill_score,
            "min_response_rate": min_response_rate,
            "max_notice_days": max_notice_days,
            "open_to_work_only": open_to_work_only,
        }
        
        filtered_candidates = apply_filters(eligible, filters)
        
        # Rank candidates
        ranked = sorted(
            filtered_candidates,
            key=lambda x: (-x.get("_final_score", 0.0), x.get("candidate_id", ""))
        )
        
        top = ranked[:top_n]
        
        # Generate reasoning
        for i, c in enumerate(top):
            try:
                c["reasoning"] = generate_reasoning(c)
            except:
                c["reasoning"] = "Candidate matches job requirements."
        
        # Build rows for display
        rows = []
        for i, c in enumerate(top, start=1):
            rows.append({
                "rank": i,
                "candidate_id": c.get("candidate_id", ""),
                "score": c.get("_final_score", 0.0),
                "current_title": c.get("current_title", "Unknown").title(),
                "yoe": c.get("yoe", 0),
                "location": c.get("location", ""),
                "country": c.get("country", ""),
                "skill_score": c.get("_skill_score", 0.0),
                "career_score": c.get("_career_score", 0.0),
                "behavior_score": c.get("_behavior_score", 0.0),
                "modifier": c.get("_modifier", 1.0),
                "reasoning": c.get("reasoning", ""),
                "_raw": c
            })
        
        # Store in session state
        st.session_state["rows"] = rows
        st.session_state["eligible_count"] = len(eligible)
        st.session_state["filtered_count"] = len(filtered_candidates)
        st.session_state["anomaly_count"] = len(candidates) - len(eligible)
        st.session_state["candidates_count"] = len(candidates)
        st.session_state["total_data_count"] = len(data)
        
        # ============================================================
        # METRICS DASHBOARD
        # ============================================================
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Eligible Profiles</div>
                <div class="metric-val" style="color: #34d399;">{st.session_state["eligible_count"]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Anomalous Profiles</div>
                <div class="metric-val" style="color: #f87171;">{st.session_state["anomaly_count"]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Filtered Out</div>
                <div class="metric-val" style="color: #60a5fa;">{st.session_state["eligible_count"] - st.session_state["filtered_count"]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)
        
        # ============================================================
        # MAIN CONTENT - TABS
        # ============================================================
        
        tab1, tab2, tab3 = st.tabs([
            "📊 Rankings",
            "📈 Visualizations",
            "🔍 Profile Inspector"
        ])
        
        # ============================================================
        # TAB 1: RANKINGS
        # ============================================================
        
        with tab1:
            if rows:
                st.markdown('<div class="section-header">📊 Evaluation Shortlist</div>', unsafe_allow_html=True)
                
                # Display DataFrame
                display_cols = ["rank", "candidate_id", "score", "current_title", "yoe", "location"]
                df_display = pd.DataFrame([
                    {k: r[k] for k in display_cols} for r in rows
                ])
                
                # Format columns
                df_display["score"] = df_display["score"].map("{:.6f}".format)
                df_display.columns = ["Rank", "Candidate ID", "Score", "Current Title", "YOE", "Location"]
                
                st.dataframe(
                    df_display,
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )
                
                # Download ranked shortlist
                csv_buf = io.StringIO()
                writer = csv.DictWriter(csv_buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
                writer.writeheader()
                for r in rows:
                    writer.writerow({
                        "candidate_id": r["candidate_id"],
                        "rank": r["rank"],
                        "score": f"{r['score']:.6f}",
                        "reasoning": r["reasoning"],
                    })
                
                st.download_button(
                    label="💾 Download Ranked Shortlist (CSV)",
                    data=csv_buf.getvalue().encode("utf-8"),
                    file_name="ranked_candidates.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No candidates match the current filters.")
        
        # ============================================================
        # TAB 2: VISUALIZATIONS - FIXED
        # ============================================================
        
        with tab2:
            if rows:
                st.markdown('<div class="section-header">📈 Score Distribution</div>', unsafe_allow_html=True)
                
                # Prepare data for visualization
                df_viz = pd.DataFrame([
                    {
                        "Rank": r["rank"],
                        "Final Score": r["score"],
                        "Skill Score": r["skill_score"],
                        "Career Score": r["career_score"],
                        "Behavior Score": r["behavior_score"],
                        "Title": r["current_title"][:30]
                    }
                    for r in rows
                ])
                
                if not df_viz.empty:
                    # Score distribution bar chart
                    fig = px.bar(
                        df_viz,
                        x="Rank",
                        y="Final Score",
                        title="Score Distribution by Rank",
                        color="Final Score",
                        color_continuous_scale="Viridis",
                        labels={"Rank": "Rank", "Final Score": "Score"},
                        text_auto='.3f'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Component scores line chart
                    fig2 = px.line(
                        df_viz,
                        x="Rank",
                        y=["Skill Score", "Career Score", "Behavior Score"],
                        title="Component Scores by Rank",
                        labels={"value": "Score", "Rank": "Rank", "variable": "Component"},
                        color_discrete_map={
                            "Skill Score": "#8b5cf6",
                            "Career Score": "#34d399",
                            "Behavior Score": "#fbbf24"
                        },
                        markers=True
                    )
                    fig2.update_layout(height=400)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No data available for visualizations.")
                
                # Top skills - FIXED
                st.markdown("#### 🏷️ Top Skills in Top Candidates")
                all_skills = []
                for r in rows:
                    skills = r["_raw"].get("skill_names", [])
                    if skills:
                        all_skills.extend(skills)
                
                if all_skills:
                    skill_counts = pd.Series(all_skills).value_counts().head(15)
                    
                    if not skill_counts.empty:
                        fig3 = px.bar(
                            skill_counts,
                            x=skill_counts.values,
                            y=skill_counts.index,
                            orientation='h',
                            title=f"Most Common Skills (Top {len(skill_counts)})",
                            labels={"x": "Count", "y": "Skill"},
                            color=skill_counts.values,
                            color_continuous_scale="Viridis",
                            text_auto=True
                        )
                        fig3.update_layout(height=400)
                        st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.info("No skills found in candidate profiles.")
                else:
                    st.info("No skills data available.")
            else:
                st.info("No data available for visualizations.")
        
        # ============================================================
        # TAB 3: PROFILE INSPECTOR
        # ============================================================
        
        with tab3:
            if rows:
                st.markdown('<div class="section-header">🔍 Candidate Profile Inspector</div>', unsafe_allow_html=True)
                
                # Select candidate
                c_ids = [r["candidate_id"] for r in rows]
                selected_id = st.selectbox("Inspect profile details:", options=c_ids)
                
                # Find selected candidate
                selected_record = next(r for r in rows if r["candidate_id"] == selected_id)
                c = selected_record["_raw"]
                
                # Display profile card
                st.markdown(f"""
                <div class="profile-card">
                    <div class="profile-title">{selected_record['current_title']}</div>
                    <div class="profile-meta">
                        ID: <b>{c['candidate_id']}</b> &nbsp;|&nbsp; 
                        Experience: <b>{selected_record['yoe']:.1f} Years</b> &nbsp;|&nbsp; 
                        Location: <b>{selected_record['location']}, {c['country']}</b>
                    </div>
                    <div style="margin-bottom: 15px;">
                        <span class="score-badge">Final Score: {selected_record['score']:.6f}</span>
                        <span class="score-badge" style="background: rgba(16, 185, 129, 0.12); color: #34d399; border-color: rgba(16, 185, 129, 0.25);">
                            Skill Alignment: {selected_record['skill_score']:.2f}
                        </span>
                        <span class="score-badge" style="background: rgba(245, 158, 11, 0.12); color: #fbbf24; border-color: rgba(245, 158, 11, 0.25);">
                            Behavioral Mod: {selected_record['modifier']:.2f}
                        </span>
                    </div>
                    <div style="font-size: 0.95rem; line-height: 1.6; color: #cbd5e1; border-top: 1px solid rgba(255, 255, 255, 0.05); padding-top: 15px;">
                        <b>Factual Reasoning Justification:</b><br/>
                        <i>"{selected_record['reasoning']}"</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Skills
                st.markdown("#### 🏷️ Declared Skills")
                skills = c.get("skill_names", [])
                if skills:
                    skills_html = "".join([f'<span class="badge-tech">{s}</span>' for s in skills[:25]])
                    st.markdown(f'<div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 10px;">{skills_html}</div>', unsafe_allow_html=True)
                    if len(skills) > 25:
                        st.caption(f"... and {len(skills) - 25} more skills")
                else:
                    st.info("No skills listed")
                
                # Notice Period
                st.markdown("#### ⏰ Notice Period")
                st.write(f"**{c.get('notice_days', 0)} Days notice**")
                
                # Additional details
                with st.expander("📊 Behavioral Signals"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Response Rate",
                            f"{c.get('response_rate', 0):.0%}",
                            help="Recruiter response rate"
                        )
                        st.metric(
                            "Open to Work",
                            "✅" if c.get("open_to_work", False) else "❌",
                            help="Open to work flag"
                        )
                    with col2:
                        st.metric(
                            "GitHub Score",
                            f"{c.get('github_score', -1):.0f}" if c.get('github_score', -1) >= 0 else "N/A",
                            help="GitHub activity score"
                        )
                        st.metric(
                            "Willing to Relocate",
                            "✅" if c.get("willing_relocate", False) else "❌",
                            help="Willing to relocate"
                        )
                    with col3:
                        st.metric(
                            "Days Active",
                            f"{c.get('days_since_active', 9999)} days ago",
                            help="Days since last active"
                        )
                        st.metric(
                            "Interview Completion",
                            f"{c.get('interview_completion', 0):.0%}",
                            help="Interview completion rate"
                        )
                
                with st.expander("💼 Career History"):
                    career = c.get("career_raw", [])
                    if career:
                        for job in career[:5]:
                            title = job.get("title", "Unknown")
                            company = job.get("company", "Unknown")
                            duration = job.get("duration_months", 0)
                            st.markdown(f"- **{title}** at {company} ({duration} months)")
                        if len(career) > 5:
                            st.caption(f"... and {len(career) - 5} more roles")
                    else:
                        st.info("No career history available")
                
                with st.expander("📊 Score Breakdown"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Skill Score", f"{selected_record['skill_score']:.4f}")
                        st.metric("Career Score", f"{selected_record['career_score']:.4f}")
                    with col2:
                        st.metric("Behavior Score", f"{selected_record['behavior_score']:.4f}")
                        st.metric("Final Score", f"{selected_record['score']:.6f}")
            else:
                st.info("No candidates available for inspection.")
    
else:
    st.info("📂 Please upload a candidate profile file (.json or .jsonl) to start the evaluation pipeline.")


