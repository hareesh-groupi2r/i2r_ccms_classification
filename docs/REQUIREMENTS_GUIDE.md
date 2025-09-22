# Requirements Files Guide

## Overview

The project has been reorganized with clean dependency management:

## 📦 Production Requirements

### `backend/requirements.txt`
**Use this for integration into other applications**
- Core dependencies only
- Clean, minimal set
- Production-ready
- Well-documented sections

### `requirements.txt` (root)
**Convenience file that points to backend requirements**
- Installs from `backend/requirements.txt`
- Single entry point for quick setup

## 🧪 Development Requirements

### `tests_standalone/requirements_*.txt`
**Various requirement files for different purposes:**

- `requirements_original.txt` - Original requirements backup
- `requirements_fixed.txt` - Fixed versions for reproducible builds
- `requirements_flexible.txt` - Flexible versions for compatibility
- `requirements_production.txt` - FastAPI-based production setup
- `requirements_py313.txt` - Python 3.13 compatibility
- `requirements_synthetic.txt` - Synthetic data generation dependencies

## 🚀 Quick Setup

### For Integration
```bash
cd backend
pip install -r requirements.txt
```

### For Development
```bash
# Use the root convenience file
pip install -r requirements.txt

# Or for specific needs
pip install -r tests_standalone/requirements_production.txt
```

## 🔧 Dependency Categories

### Core API Framework
- Flask for REST API
- CORS support
- Environment configuration

### Data Processing
- Pandas for data manipulation
- NumPy for numerical operations
- Scikit-learn for ML utilities

### AI/ML Components
- Sentence transformers for embeddings
- FAISS for vector search
- Transformers for language models
- PyTorch backend

### Document Processing
- PDFPlumber for text extraction
- Tesseract for OCR
- PIL for image processing

### LLM Integration
- OpenAI API client
- Anthropic Claude API
- Google Generative AI

## 🔄 Maintenance

When adding new dependencies:
1. Add to `backend/requirements.txt` for production features
2. Add to appropriate `tests_standalone/requirements_*.txt` for development/testing
3. Update this guide with rationale

## 🎯 Integration Notes

The `backend/requirements.txt` is designed to be:
- **Minimal**: Only essential dependencies
- **Stable**: Pinned versions for reproducibility  
- **Modular**: Optional dependencies clearly marked
- **Clean**: Well-organized and documented

This makes it easy to integrate the backend service into any application without dependency conflicts.