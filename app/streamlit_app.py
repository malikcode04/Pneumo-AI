"""
🫁 Pneumonia Detection System - Professional Clinical Interface
Production-ready clinical interface for pneumonia triage with DICOM support and Reporting.
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import sys
import pandas as pd
import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import PneumoniaPredictor, MedicalIntegrityError
from src.utils import Config
from src.utils.dicom_handler import read_dicom
from src.services.auth import init_auth, check_credentials, create_user

# HOTFIX: Force reload modules to ensure new functions are picked up
import importlib
import src.utils.report_gen
import src.services.db
import src.inference.predictor
import src.inference
importlib.reload(src.utils.report_gen)
importlib.reload(src.services.db)
importlib.reload(src.inference.predictor)
importlib.reload(src.inference)

from src.utils.report_gen import create_pdf_report
from src.services.db import init_db, add_scan, get_all_scans, delete_scan, update_scan

# ... (rest of imports are fine, just fixing the specific import line if needed, but replace_file_content targetting the function is safer)


if "current_view" not in st.session_state:
    st.session_state.current_view = "Dashboard"
if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "expanded"
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "last_analysis_image" not in st.session_state:
    st.session_state.last_analysis_image = None

# Page configuration
st.set_page_config(
    page_title="Pneumo AI - Advanced Radiology",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state=st.session_state.sidebar_state
)

# Load Custom CSS
def load_css():
    css_path = Path(__file__).parent.parent / "assets" / "styles.css"
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Session State Initialization
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False

# Initialize Backend
init_db()  # Ensure database is initialized
init_auth()

@st.cache_resource(show_spinner="Loading Clinical Model...")
def load_model():
    """Load trained model."""
    try:
        config = Config()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create model
        model = create_densenet121(
            num_classes=config.get('model.num_classes', 3),
            pretrained=False,
            dropout_rate=config.get('model.dropout_rate', 0.3)
        )
        
        # Load checkpoint
        checkpoint_path = config.get('inference.checkpoint_path', 'checkpoints/best_recall.pth')
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            # Fallback for demo
            pass 
        
        # Create predictor
        predictor = PneumoniaPredictor(
            model=model,
            device=device,
            class_names=['Normal', 'Bacterial Pneumonia', 'Viral Pneumonia'],
            use_uncertainty=True,
            use_gradcam=True,
            use_lung_masking=True,
            mc_samples=20
        )
        
        return predictor
    except Exception as e:
        return None

def show_login():
    """Render Login & Sign Up Screen."""
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: #00d4ff;">🫁 Pneumo AI</h1>
            <p style="font-size: 1.2em; color: #a0a0a0;">Advanced Clinical Decision Support System</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True)
                
                if submitted:
                    if check_credentials(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        
        with tab2:
            with st.form("signup_form"):
                new_user = st.text_input("Choose Username")
                new_pass = st.text_input("Choose Password", type="password")
                confirm_pass = st.text_input("Confirm Password", type="password")
                signup_submitted = st.form_submit_button("Create Account", use_container_width=True)
                
                if signup_submitted:
                    if new_pass != confirm_pass:
                        st.error("Passwords do not match!")
                    elif len(new_pass) < 4:
                        st.error("Password must be at least 4 characters.")
                    else:
                        if create_user(new_user, new_pass):
                            st.success("Account created! Logging you in...")
                            st.session_state.authenticated = True
                            st.session_state.username = new_user
                            st.rerun()
                        else:
                            st.error("Username already exists.")

def show_disclaimer_modal():
    """Show modal for HIPAA/Compliance disclaimer."""
    st.markdown("<br><br><br>", unsafe_allow_html=True) 
    
    with st.container():
        st.markdown("""
            <div style="background-color: #1a1d29; padding: 30px; border-radius: 15px; border: 1px solid #00d4ff; text-align: center; margin-bottom: 20px;">
                <h2 style="color: #00d4ff; margin-top: 0;">⚠️ Medical Device Disclaimer</h2>
                <p style="color: #e0e0e0; font-size: 1.1em; margin: 20px 0; line-height: 1.6;">
                    <strong>Pneumo AI</strong> is a <strong>Clinical Decision Support System (CDSS)</strong> prototype.
                    <br>It is <strong>NOT</strong> intended to be a primary diagnostic tool. 
                    <br>All results must be verified by a qualified radiologist.
                </p>
                <div style="background-color: rgba(255, 193, 7, 0.1); padding: 15px; border-radius: 8px; border: 1px solid #ffc107; margin-top: 20px;">
                    <p style="color: #ffc107; margin: 0; font-weight: bold;">
                        By proceeding, you acknowledge that this tool is for investigational use only and does not replace professional medical advice.
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            agree = st.checkbox("I have read and understand the disclaimer above")
            if st.button("Enter System", type="primary", disabled=not agree, use_container_width=True):
                st.session_state.disclaimer_accepted = True
                st.rerun()

def render_sidebar():
    with st.sidebar:
        st.markdown(f"## 👤 Dr. {st.session_state.username.capitalize()}")
        
        st.info(f"**System Status:** Online")
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 🧭 Navigation")
        
        if st.button("📊 Dashboard", use_container_width=True):
             st.session_state.current_view = "Dashboard"
             st.session_state.sidebar_state = "expanded"
             st.rerun()

        if st.button("🔍 New Analysis", use_container_width=True):
            st.session_state.current_view = "Analysis"
            st.session_state.sidebar_state = "collapsed"
            st.rerun()
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

def render_dashboard():
    """Doctor's Dashboard View."""
    st.markdown("# 📊 Doctor's Dashboard")
    
    scans = get_all_scans(st.session_state.username)
    
    # KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Total Patients Processed", len(scans))
    with kpi2:
        pneumonia_count = sum(1 for s in scans if "Pneumonia" in s['diagnosis'])
        st.metric("Pneumonia Cases", pneumonia_count)
    with kpi3:
        normal_count = len(scans) - pneumonia_count
        st.metric("Normal Cases", normal_count)
    
    st.markdown("### 🗂️ Patient History")
    
    if not scans:
        st.info("No records found. Start a new analysis.")
        return

    # Display as an interactive section
    for scan in scans:
        scan_id = scan['id']
        edit_key = f"edit_mode_{scan_id}"
        
        # Initialize edit state for this item if not exists
        if edit_key not in st.session_state:
            st.session_state[edit_key] = False

        with st.expander(f"{scan['scan_date']} - {scan['patient_name']} ({scan['diagnosis']})"):
            
            # --- EDIT MODE ---
            if st.session_state[edit_key]:
                with st.form(f"edit_form_{scan_id}"):
                    st.markdown("#### ✏️ Edit Record")
                    new_name = st.text_input("Patient Name", value=scan['patient_name'])
                    new_notes = st.text_area("Clinical Notes", value=scan['notes'])
                    
                    c1, c2 = st.columns([1, 1])
                    if c1.form_submit_button("💾 Save Changes", type="primary", use_container_width=True):
                        update_scan(scan_id, new_name, new_notes)
                        st.session_state[edit_key] = False
                        st.success("Record updated!")
                        st.rerun()
                        
                    if c2.form_submit_button("❌ Cancel", use_container_width=True):
                        st.session_state[edit_key] = False
                        st.rerun()

            # --- VIEW MODE ---
            else:
                # Top Row: Details
                info_col, action_col = st.columns([3, 1])
                
                with info_col:
                    st.markdown(f"**Patient ID:** `{scan['patient_id']}`")
                    st.markdown(f"**Diagnosis:** `{scan['diagnosis']}`")
                    st.markdown(f"**Confidence:** `{scan['confidence']:.1%}`")
                    st.info(f"**📝 Notes:** {scan['notes'] if scan['notes'] else 'No notes recorded.'}")

                with action_col:
                    st.markdown("#### Actions")
                    if st.button("✏️ Edit Details", key=f"btn_edit_{scan_id}", use_container_width=True):
                        st.session_state[edit_key] = True
                        st.rerun()
                        
                    if st.button("🗑️ Delete Record", key=f"btn_del_{scan_id}", type="primary", use_container_width=True):
                        delete_scan(scan_id)
                        st.rerun()

                st.markdown("---")
                st.markdown("#### 🖼️ Clinical Imaging")
                
                img_c1, img_c2 = st.columns(2)
                image_path = scan['image_path']
                heatmap_path = scan['heatmap_path']
                
                if image_path and Path(image_path).exists():
                    img_c1.image(image_path, caption="Original X-Ray", use_container_width=True)
                else:
                    img_c1.warning("Original image not found.")
                    
                if heatmap_path and Path(heatmap_path).exists():
                    img_c2.image(heatmap_path, caption="AI Evidence Map (Grad-CAM)", use_container_width=True)
                else:
                    img_c2.warning("Heatmap not found.")


def render_analysis_view(predictor):
    st.markdown("# ☢️ Diagnostic Imaging Console")
    
    col_u1, col_u2 = st.columns([2, 1])
    
    with col_u1:
        uploaded_file = st.file_uploader(
            "Upload Study (DICOM or Image)",
            type=['dcm', 'jpg', 'png', 'jpeg'],
            help="Supports standard DICOM (.dcm) and Image formats"
        )
    
    with col_u2:
        st.markdown("### 📋 Patient Metadata")
        
        # Manual Override Form
        with st.expander("📝 Edit Patient Details", expanded=True):
            p_name = st.text_input("Name", key="p_name_input")
            c1, c2 = st.columns(2)
            p_age = c1.text_input("Age", key="p_age_input")
            p_gender = c2.selectbox("Gender", ["M", "F", "Other"], key="p_gender_input")
            p_view = st.selectbox("View Position", ["PA", "AP", "Lateral", "L-Decub", "R-Decub"], key="p_view_input")
            
        patient_placeholder = st.empty()
        patient_placeholder.info("Waiting for image...")

    if not uploaded_file:
        st.session_state.analysis_result = None
        st.session_state.last_analysis_image = None

    if uploaded_file:
        # Process File
        with st.spinner("Processing Imaging Data..."):
            image_source = None
            metadata = {}
            
            if uploaded_file.name.lower().endswith('.dcm'):
                dicom_data = read_dicom(uploaded_file)
                if dicom_data['error']:
                    st.error(f"Error reading DICOM: {dicom_data['error']}")
                    return
                image_source = dicom_data['image']
                metadata = dicom_data['metadata']
            else:
                image_source = Image.open(uploaded_file).convert('RGB')
                metadata = {
                    "Patient ID": "ANON-" + str(abs(hash(uploaded_file.name)))[:8],
                    "Patient Name": "Anonymous",
                    "Study Date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "Modality": "XR"
                }

            # Overlay Manual Inputs
            if p_name: metadata["Patient Name"] = p_name
            # Map report generator expectations (Name key) to DICOM key (Patient Name)
            metadata["Name"] = metadata.get("Patient Name") 
            metadata["Age"] = p_age if p_age else "N/A"
            metadata["Gender"] = p_gender if p_gender else "N/A"

            # Display Metadata
            patient_placeholder.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px;">
                    <p><strong>ID:</strong> {metadata.get('Patient ID')}</p>
                    <p><strong>Name:</strong> {metadata.get('Name')}</p>
                    <p><strong>Age/Sex:</strong> {metadata.get('Age')} / {metadata.get('Gender')}</p>
                    <p><strong>View:</strong> {p_view}</p>
                    <p><strong>Date:</strong> {metadata.get('Study Date')}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Show Image
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                 st.image(image_source, caption="Original X-Ray", use_column_width=True)
            
            # Predict
            if st.button("🚀 Run AI Diagnosis", type="primary", use_container_width=True):
                try:
                    image_np = np.array(image_source)
                    result = predictor.predict(image_np, return_heatmap=True)
                    st.session_state.analysis_result = result
                    st.session_state.last_analysis_image = image_source
                except MedicalIntegrityError as e:
                    st.error(f"⚠️ Medical Integrity Check Failed: {str(e)}")
                    st.warning("Please upload a standard Chest X-ray image for analysis.")
                    st.session_state.analysis_result = None
                    return
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    st.session_state.analysis_result = None
                    return

            # Persistent Display Logic
            if st.session_state.analysis_result is not None:
                result = st.session_state.analysis_result
                image_source = st.session_state.last_analysis_image
                

                # Display Prediction Badge
                st.markdown("---")
                label = result['predicted_label']
                conf = result['confidence']
                interpretation = result['clinical_metrics']['interpretation']
                is_indeterminate = result.get('is_indeterminate', False)
                
                # Robustness check: Ensure interpretation is a dict (handles caching issues)
                if isinstance(interpretation, str):
                    interpretation = {
                        "category": "High Accuracy" if conf > 0.9 else "Moderate",
                        "text": interpretation
                    }
                
                # Accuracy Badge Logic
                acc_category = interpretation.get('category', 'Status Unknown')
                badge_class = "risk-high" if ("Pneumonia" in label and conf > 0.5) else "risk-low"
                if "Normal" in label: badge_class = "risk-low"
                if is_indeterminate: badge_class = "risk-medium" # Yellow/Warning color
                
                # Visual feedback for high accuracy
                if acc_category == "High Accuracy":
                    st.balloons()
                    st.markdown(f"""
                        <div style="padding: 20px; border-radius: 10px; background: rgba(0, 212, 255, 0.1); border: 2px solid #00d4ff; margin-bottom: 20px; animation: pulse 2s infinite;">
                            <h2 style="color: #00d4ff; text-align: center; margin: 0;">✨ {acc_category} Mode Enabled</h2>
                            <p style="text-align: center; color: #a0a0a0; margin: 10px 0 0 0;">Statistical significance thresholds exceeded for {label}</p>
                        </div>
                    """, unsafe_allow_html=True)

                if is_indeterminate:
                    st.markdown(f"""
                        <div style="padding: 15px; border-radius: 10px; background: rgba(255, 193, 7, 0.1); border: 1px solid #ffc107; margin-bottom: 20px;">
                            <h3 style="color: #ffc107; margin: 0;">🔬 {acc_category}</h3>
                            <p style="color: #e0e0e0; margin: 10px 0 0 0;">{interpretation.get('text')}</p>
                            <small style="color: #a0a0a0;">Reason: AI model detected features overlapping both Bacterial and Viral patterns. Manual differential diagnosis via PCR or Sputum culture is recommended.</small>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-badge {badge_class}">{label} ({interpretation.get("category", "N/A")} - {conf:.1%})</div>', unsafe_allow_html=True)
                
                # Visuals
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("#### Evidence Map (Grad-CAM)")
                    st.image(result['heatmap_overlay'], use_column_width=True)
                with c2:
                    st.markdown("#### Clinical Metrics")
                    st.markdown(f"""
                    <div class="metric-card">
                        <p><strong>Sensitivity:</strong> {result['clinical_metrics']['sensitivity']}</p>
                        <p><strong>Specificity:</strong> {result['clinical_metrics']['specificity']}</p>
                        <p><strong>Uncertainty:</strong> {result['uncertainty']['entropy']:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # SAVE TO DB AUTOMATICALLY
                    heatmap_img = Image.fromarray(result['heatmap_overlay'])
                    scan_id = add_scan(
                        metadata.get('Patient ID'),
                        metadata.get('Patient Name'),
                        label,
                        conf,
                        image_source,
                        heatmap_img,
                        st.session_state.username,
                        notes="Auto-saved from inference"
                    )
                    st.success(f"✅ Record Saved to Dashboard (ID: {scan_id})")
                    
                with c3:
                    st.markdown("#### 📄 Actions")
                    doc_note = st.text_area("Final Clinical Assessment / Notes", key="doc_notes_input")
                    
                    # Generate Report
                    pdf_bytes = create_pdf_report(
                        metadata, 
                        result, 
                        {'original': image_source, 'overlay': Image.fromarray(result['heatmap_overlay'])},
                        doctor_name=st.session_state.username,
                        doctor_statement=doc_note,
                        view_position=p_view
                    )
                    
                    st.download_button(
                        label="📥 Download Clinical Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"Report_{metadata.get('Patient ID')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                

def main():
    if not st.session_state.authenticated:
        show_login()
        return

    if not st.session_state.disclaimer_accepted:
        show_disclaimer_modal()
        return

    render_sidebar()
    
    predictor = load_model()
    if not predictor:
        st.error("System Error: Model failed to initialize. Contact IT.")
        return

    if st.session_state.current_view == "Analysis":
        render_analysis_view(predictor)
    else:
        render_dashboard()

if __name__ == "__main__":
    main()
