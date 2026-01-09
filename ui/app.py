import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import os

# ===============================
# FASTAPI CONFIG
# ===============================
API_URL = os.getenv("API_URL", "http://localhost:8000")


# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# Custom CSS - Dark Professional Theme
# ===============================
st.markdown("""
<style>
:root {
    --bg: #0a0e27;
    --text: #ffffff;
    --accent: #6366f1;
    --success: #10b981;
    --danger: #ef4444;
    --warning: #f59e0b;
}

* {
    color: var(--text);
}

.metric-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(99, 102, 241, 0.05) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    backdrop-filter: blur(10px);
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 15px;
    margin: 20px 0;
}

.section-header {
    border-bottom: 3px solid var(--accent);
    padding-bottom: 10px;
    margin-bottom: 20px;
}

.download-section {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 12px;
    padding: 20px;
    margin: 20px 0;
}

.stDownloadButton > button {
    background: linear-gradient(90deg, var(--accent), var(--success));
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-weight: 600;
    width: 100%;
}

.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3);
}

.risk-badge-high {
    background: var(--danger);
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 12px;
    display: inline-block;
}

.risk-badge-medium {
    background: var(--warning);
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 12px;
    display: inline-block;
}

.risk-badge-low {
    background: var(--success);
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 12px;
    display: inline-block;
}

h1, h2, h3 {
    color: var(--text) !important;
}

.stTabs [data-baseweb="tab-list"] button {
    background: rgba(99, 102, 241, 0.1);
    border-radius: 8px;
    color: var(--text);
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background: var(--accent);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# Header
# ===============================
st.markdown("# 🎯 CHURN PREDICTION INTELLIGENCE")
st.markdown(
    '<p class="subtitle">AI-Powered Customer Retention Analytics Platform</p>',
    unsafe_allow_html=True
)

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.markdown("## ⚙️ DASHBOARD SETTINGS")
    st.markdown("---")
    
    # Display options
    show_advanced = st.checkbox("🔬 Advanced Analytics", value=True)
    show_risk_details = st.checkbox("📊 Risk Distribution Analysis", value=True)
    show_customer_segments = st.checkbox("👥 Customer Segments", value=True)
    
    st.markdown("---")
    
    # Export options
    st.markdown("### 📥 EXPORT OPTIONS")
    export_format = st.radio("Choose format:", ["CSV", "Excel"], horizontal=True)
    
    st.markdown("---")
    st.info("💡 Upload a CSV with customer data to generate predictions and insights.")

# ===============================
# File Upload
# ===============================
st.markdown("## 📤 DATA INGESTION")
uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])

# ===============================
# MAIN LOGIC
# ===============================
if uploaded_file is not None:

    with st.spinner("🚀 Calling FastAPI for predictions..."):
        response = requests.post(
            f"{API_URL}/predict_csv",
            files={"file": uploaded_file}
        )

    if response.status_code != 200:
        st.error("❌ FastAPI request failed")
        st.stop()

    result = response.json()

    predictions = pd.DataFrame(result["predictions"])
    kpis = result["kpis"]

    st.success("✅ Prediction completed successfully")

    # ===============================
    # EXECUTIVE KPI SUMMARY - Enhanced
    # ===============================
    st.markdown("<div class='section-header'><h2>📊 EXECUTIVE KPI SUMMARY</h2></div>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("👥 Total Customers", f"{kpis['total_customers']:,}")
    
    with col2:
        high_pct = (kpis['high_risk_customers'] / kpis['total_customers'] * 100) if kpis['total_customers'] > 0 else 0
        st.metric("🔴 High Risk", kpis['high_risk_customers'], f"{high_pct:.1f}%")
    
    with col3:
        med_pct = (kpis['medium_risk_customers'] / kpis['total_customers'] * 100) if kpis['total_customers'] > 0 else 0
        st.metric("🟡 Medium Risk", kpis['medium_risk_customers'], f"{med_pct:.1f}%")
    
    with col4:
        low_pct = (kpis['low_risk_customers'] / kpis['total_customers'] * 100) if kpis['total_customers'] > 0 else 0
        st.metric("🟢 Low Risk", kpis['low_risk_customers'], f"{low_pct:.1f}%")
    
    with col5:
        st.metric("📈 Avg Churn Prob.", f"{kpis['average_churn_probability']:.1%}")

    # ===============================
    # ADVANCED METRICS
    # ===============================
    if show_advanced:
        st.markdown("<div class='section-header'><h2>🔬 ADVANCED ANALYTICS</h2></div>", unsafe_allow_html=True)

        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
        
        with adv_col1:
            st.metric(
                "📌 Max Churn Risk",
                f"{predictions['churn_probability'].max():.1%}",
                "Highest risk detected"
            )
        
        with adv_col2:
            st.metric(
                "✅ Min Churn Risk",
                f"{predictions['churn_probability'].min():.1%}",
                "Lowest risk detected"
            )
        
        with adv_col3:
            std_dev = predictions['churn_probability'].std()
            st.metric(
                "📊 Risk Std Dev",
                f"{std_dev:.3f}",
                "Risk variance"
            )
        
        with adv_col4:
            median_risk = predictions['churn_probability'].median()
            st.metric(
                "🎯 Median Risk",
                f"{median_risk:.1%}",
                "Middle value"
            )

    # ===============================
    # VISUALIZATIONS - Enhanced
    # ===============================
    st.markdown("<div class='section-header'><h2>📉 CHURN ANALYTICS & VISUALIZATIONS</h2></div>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    # Churn Distribution Histogram
    with chart_col1:
        st.markdown("#### Churn Probability Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=predictions["churn_probability"],
            nbinsx=30,
            marker_color="#6366f1",
            name="Distribution"
        ))
        fig_hist.update_layout(
            template="plotly_dark",
            xaxis_title="Churn Probability",
            yaxis_title="Number of Customers",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        st.plotly_chart(fig_hist, use_container_width=True, key='histogram')

    # Risk Distribution Pie Chart
    with chart_col2:
        st.markdown("#### Risk Distribution")
        risk_counts = [
            kpis['high_risk_customers'],
            kpis['medium_risk_customers'],
            kpis['low_risk_customers']
        ]
        risk_labels = ['High Risk', 'Medium Risk', 'Low Risk']
        colors = ['#ef4444', '#f59e0b', '#10b981']
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=risk_labels,
            values=risk_counts,
            marker=dict(colors=colors),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>'
        )])
        fig_pie.update_layout(
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True, key='pie_chart')

    # Risk Distribution Analysis
    if show_risk_details:
        st.markdown("<div class='section-header'><h3>📊 Risk Distribution Analysis</h3></div>", unsafe_allow_html=True)
        
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        with risk_col1:
            st.markdown("##### High Risk Segment")
            st.markdown(f"**Count:** {kpis['high_risk_customers']:,}")
            st.markdown(f"**Percentage:** {(kpis['high_risk_customers']/kpis['total_customers']*100):.1f}%")
            st.markdown(f"**Priority:** 🔴 Critical")
        
        with risk_col2:
            st.markdown("##### Medium Risk Segment")
            st.markdown(f"**Count:** {kpis['medium_risk_customers']:,}")
            st.markdown(f"**Percentage:** {(kpis['medium_risk_customers']/kpis['total_customers']*100):.1f}%")
            st.markdown(f"**Priority:** 🟡 Important")
        
        with risk_col3:
            st.markdown("##### Low Risk Segment")
            st.markdown(f"**Count:** {kpis['low_risk_customers']:,}")
            st.markdown(f"**Percentage:** {(kpis['low_risk_customers']/kpis['total_customers']*100):.1f}%")
            st.markdown(f"**Priority:** 🟢 Monitor")

    # ===============================
    # CUSTOMER SEGMENTS
    # ===============================
    if show_customer_segments:
        st.markdown("<div class='section-header'><h2>👥 CUSTOMER SEGMENTS</h2></div>", unsafe_allow_html=True)
        
        segment_tabs = st.tabs(["High Risk", "Medium Risk", "Low Risk", "All Customers"])
        
        with segment_tabs[0]:
            st.markdown("### 🔴 High Risk Customers")
            high_risk_df = predictions[predictions["risk_level"] == "High"]
            
            if high_risk_df.empty:
                st.success("✨ No high-risk customers detected!")
            else:
                st.markdown(f"**Total:** {len(high_risk_df)} customers ({len(high_risk_df)/len(predictions)*100:.1f}%)")
                high_risk_sorted = high_risk_df.sort_values(by="churn_probability", ascending=False)
                st.dataframe(high_risk_sorted, width='stretch')
        
        with segment_tabs[1]:
            st.markdown("### 🟡 Medium Risk Customers")
            medium_risk_df = predictions[predictions["risk_level"] == "Medium"]
            
            if medium_risk_df.empty:
                st.info("No medium-risk customers detected.")
            else:
                st.markdown(f"**Total:** {len(medium_risk_df)} customers ({len(medium_risk_df)/len(predictions)*100:.1f}%)")
                medium_risk_sorted = medium_risk_df.sort_values(by="churn_probability", ascending=False)
                st.dataframe(medium_risk_sorted, width='stretch')
        
        with segment_tabs[2]:
            st.markdown("### 🟢 Low Risk Customers")
            low_risk_df = predictions[predictions["risk_level"] == "Low"]
            
            if low_risk_df.empty:
                st.info("No low-risk customers detected.")
            else:
                st.markdown(f"**Total:** {len(low_risk_df)} customers ({len(low_risk_df)/len(predictions)*100:.1f}%)")
                low_risk_sorted = low_risk_df.sort_values(by="churn_probability", ascending=False)
                st.dataframe(low_risk_sorted.head(50), width='stretch')
                if len(low_risk_sorted) > 50:
                    st.caption(f"Showing 50 of {len(low_risk_sorted)} low-risk customers")
        
        with segment_tabs[3]:
            st.markdown("### 📋 All Customers")
            st.markdown(f"**Total:** {len(predictions)} customers")
            st.dataframe(predictions.sort_values(by="churn_probability", ascending=False), width='stretch')

    # ===============================
    # DOWNLOAD SECTION - Enhanced
    # ===============================
    st.markdown("<div class='download-section'>", unsafe_allow_html=True)
    st.markdown("<h2>💾 EXPORT & DOWNLOAD</h2>", unsafe_allow_html=True)
    st.markdown("Download predictions and segmented data for further analysis", unsafe_allow_html=True)
    
    # Helper function to export data
    def export_data(df, format_type="CSV"):
        if format_type == "CSV":
            return df.to_csv(index=False).encode("utf-8"), "csv"
        else:  # Excel
            try:
                import openpyxl
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Predictions')
                return buffer.getvalue(), "xlsx"
            except:
                return df.to_csv(index=False).encode("utf-8"), "csv"
    
    download_col1, download_col2, download_col3 = st.columns(3)
    
    with download_col1:
        st.markdown("**📥 Full Dataset**")
        csv_data, ext = export_data(predictions, export_format)
        st.download_button(
            label="Download All Predictions",
            data=csv_data,
            file_name=f"churn_predictions_all.{ext}",
            mime=f"text/csv" if ext == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_all"
        )
        st.caption(f"📊 {len(predictions):,} records")
    
    with download_col2:
        st.markdown("**🔴 High Risk Only**")
        high_risk_data = predictions[predictions["risk_level"] == "High"]
        if len(high_risk_data) > 0:
            csv_data, ext = export_data(high_risk_data, export_format)
            st.download_button(
                label="Download High Risk",
                data=csv_data,
                file_name=f"churn_high_risk.{ext}",
                mime=f"text/csv" if ext == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_high"
            )
            st.caption(f"📊 {len(high_risk_data):,} records")
        else:
            st.info("No high-risk customers")
    
    with download_col3:
        st.markdown("**📋 Summary Report**")
        summary_df = pd.DataFrame({
            'Metric': ['Total Customers', 'High Risk', 'Medium Risk', 'Low Risk', 'Avg Churn Prob'],
            'Value': [
                kpis['total_customers'],
                kpis['high_risk_customers'],
                kpis['medium_risk_customers'],
                kpis['low_risk_customers'],
                f"{kpis['average_churn_probability']:.2%}"
            ]
        })
        csv_data, ext = export_data(summary_df, export_format)
        st.download_button(
            label="Download Summary",
            data=csv_data,
            file_name=f"churn_summary.{ext}",
            mime=f"text/csv" if ext == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_summary"
        )
        st.caption("Executive summary")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ===============================
    # HIGH RISK CUSTOMERS (Quick View)
    # ===============================
    st.markdown("<div class='section-header'><h2>🔴 IMMEDIATE ACTIONS REQUIRED</h2></div>", unsafe_allow_html=True)

    high_risk_df = predictions[predictions["risk_level"] == "High"]

    if high_risk_df.empty:
        st.success("✨ Excellent! No high-risk customers detected. 🎉")
    else:
        st.warning(f"⚠️ {len(high_risk_df)} customers require immediate attention")
        high_risk_display = high_risk_df.sort_values(by="churn_probability", ascending=False).head(10)
        st.dataframe(high_risk_display, width='stretch')
        if len(high_risk_df) > 10:
            st.caption(f"Showing top 10 of {len(high_risk_df)} high-risk customers")

else:
    st.info("Upload a CSV file to begin analysis.")
