# 🫁 Pneumo AI

**Clinical-grade AI system for Pneumonia triage from Chest X-Rays with adaptive interpretation and multi-tenant security.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![Live Demo](https://img.shields.io/badge/Live-Streamlit-FF4B4B.svg)](https://pneumo-ai.streamlit.app/)
[![Vercel Hub](https://img.shields.io/badge/Vercel-Landing_Page-000000.svg)](https://pneumo-ai-seven.vercel.app/)

---

## 🌐 Live Access

- **Clinical UI Dashboard**: [https://pneumo-ai.streamlit.app/](https://pneumo-ai.streamlit.app/)
- **Product Hub & API**: [https://pneumo-ai-seven.vercel.app/](https://pneumo-ai-seven.vercel.app/)

---

## � What is Pneumo AI?

**Pneumo AI** is a production-ready Clinical Decision Support System (CDSS) designed to assist radiologists and clinicians in the early detection and triage of pneumonia. Unlike standard classification models, Pneumo AI provides a multi-layered diagnostic experience focusing on **statistical significance**, **explainability**, and **data security**.

### 🌟 Key Enhancements
- 🎯 **Adaptive Accuracy Categorization**: Leverages statistical thresholds to categorize results into *High Accuracy*, *Moderate*, and *Review Required* modes.
- 🔒 **Multi-Tenant Data Isolation**: Secure doctor-level account isolation. Clinicians only see the patients and scans they have personally analyzed.
- 🎨 **Explainable AI (XAI)**: Dual-view clinical imaging featuring original X-rays alongside Grad-CAM++ evidence maps.
- 📄 **Clinical Reporting**: Automated PDF generation with doctor statements, image watermarking, and view-position metadata.

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Core Brain** | PyTorch, TorchVision, TIMM (Deep Learning Frameworks) |
| **Model** | DenseNet-121 (clinical-grade backbone) |
| **Explainability** | Grad-CAM++, Lung-Masking, MC Dropout |
| **Frontend UI** | Streamlit (Clinical Console), HTML5/CSS3 (Premium Landing Page) |
| **Backend API** | FastAPI, Uvicorn (RESTful Architecture) |
| **Database** | SQLite3 (Persistent Metadata), Local/Cloud Storage (Imaging) |
| **Deployment** | Docker, Vercel (Edge Functions), Streamlit Cloud |

---

## 🏗️ How it Works (The Pipeline)

Pneumo AI follows a rigorous clinical pipeline to ensure safety and precision:

### 1. Secure Authentication
Clinicians log into a secure environment. The system uses **Bcrypt** hashing for password protection. Upon signup, the system automatically initializes a private workspace for the doctor.

### 2. Diagnostic Console
The **Adaptive Sidebar** intelligently collapses during analysis to provide a focused environment. Clinicians can upload standard high-res images (JPG/PNG) or medical **DICOM** files.

### 3. Inference Engine
- **Preprocessing**: Images are normalized and resized to `224x224` using clinical-standard transforms.
- **Analysis**: The DenseNet-121 model performs a 3-class classification (Normal, Bacterial, Viral).
- **Uncertainty**: The system runs 20 forward passes (MC Dropout) to calculate entropy and confidence intervals.

### 4. Interpretation & Feedback
Results are pushed through a clinical significance filter:
- **High Accuracy Mode**: Triggered for results >90% confidence. Triggers visual confirmation and high-priority badges.
- **Evidence Mapping**: Grad-CAM++ identifies the exact pixels influencing the AI's decision.

### 5. Secure Archiving & Reporting
Scans are saved to the **Isolated Database**. Clinicians can add notes, edit patient metadata, and generate a **Clinical Grade PDF Report** which includes watermarked evidence maps and the doctor's assessment statement.

---

## � Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/malikcode04/Pneumo-AI.git
cd Pneumo-AI

# Create environment
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows

# Install optimized requirements
pip install -r requirements.txt
```

### Local Execution

- **Standard UI**: `streamlit run app/streamlit_app.py`
- **FastAPI API**: `uvicorn app.api.main:app --reload`

---

## 🏥 Clinical Validation

| Metric | Target |
|--------|--------|
| **Sensitivity (Recall)** | ≥ 96% |
| **Specificity** | ≥ 87% |
| **Explainability** | Grad-CAM++ Visualization |
| **Privacy** | Row-Level Data Isolation |

---

## ⚠️ Disclaimer

**Pneumo AI is a clinical-decision support tool, NOT a primary diagnostic device.** It is designed for research and educational purposes. All outputs MUST be verified by a board-certified radiologist.

---

**Developed for the future of radiology with ❤️ by Malik Code**
