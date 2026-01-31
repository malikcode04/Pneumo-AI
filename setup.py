"""
Pneumonia Detection System - Production-Ready Medical AI
A clinical-grade chest X-ray analysis system for pneumonia triage.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="pneumo-ai",
    version="1.1.0",
    author="Medical AI Team",
    author_email="contact@medical-ai.example.com",
    description="Pneumo AI: Clinical-grade pneumonia detection from chest X-rays",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/medical-ai/pneumonia-detector",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "timm>=0.9.12",
        "monai>=1.3.0",
        "opencv-python>=4.8.1.78",
        "Pillow>=10.1.0",
        "albumentations>=1.3.1",
        "pytorch-grad-cam>=1.4.8",
        "matplotlib>=3.8.2",
        "numpy>=1.24.3",
        "pandas>=2.1.4",
        "scikit-learn>=1.3.2",
        "streamlit>=1.29.0",
        "fastapi>=0.108.0",
        "uvicorn[standard]>=0.25.0",
        "reportlab>=4.0.7",
        "PyYAML>=6.0.1",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.1",
        "loguru>=0.7.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "flake8>=7.0.0",
            "mypy>=1.7.1",
        ],
        "export": [
            "onnx>=1.15.0",
            "onnxruntime>=1.16.3",
        ],
        "monitoring": [
            "tensorboard>=2.15.1",
            "wandb>=0.16.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "pneumonia-train=training.trainer:main",
            "pneumonia-predict=predict:main",
            "pneumonia-evaluate=evaluation.evaluate:main",
            "pneumonia-app=app.streamlit_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "configs/*.json"],
    },
    zip_safe=False,
)
