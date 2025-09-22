# CCMS Classification Backend Service

This directory contains the core backend service files needed for integration into other applications.

## 🚀 Quick Start

**For new developers - one command setup:**

```bash
./setup.sh
```

Copy .env.example to .env and update the keys

Then configure your API keys and run:

```bash
./start_ccms_backend.sh
```

## 🏗️ Core Components

### API Service
- `api/app.py` - Main Flask application
- `api/service_endpoints.py` - REST API endpoints

### Services
- `services/` - Core business logic services
  - `hybrid_rag_classification_service.py` - Main classification service
  - `category_mapping_service.py` - Category mapping logic
  - `document_processing_orchestrator.py` - Document processing workflow
  - `llm_service.py` - LLM integration
  - `ocr_service.py` - OCR processing
  - Other supporting services

### Classifier Modules (External)
- `../classifier/` - Core classification algorithms (shared at project root)
  - `hybrid_rag.py` - Hybrid RAG implementation  
  - `pure_llm.py` - Pure LLM classification
  - `pdf_extractor.py` - PDF processing
  - `embeddings.py` - Embedding management
  - Other classification modules

### Configuration
- `config.yaml` - Main configuration file
- `requirements.txt` - Python dependencies

## 📋 Prerequisites

### System Dependencies

Install these system packages before running setup:

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr poppler-utils
```

**macOS:**
```bash
brew install tesseract poppler
```

**Windows:**
- Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Install Poppler: https://blog.alivate.com.au/poppler-windows/

### Python Requirements

- Python 3.12+ (tested with 3.12.3)
- pip (latest version)

## 🔧 Manual Setup

If you prefer manual setup or the script fails:

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Run the service
cd api
python app.py
```

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```env
# Required for AI document processing
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional for advanced features
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
GOOGLE_CLOUD_PROJECT=your_project_id
```

### Getting API Keys

1. **Google Gemini API Key:**
   - Visit: https://aistudio.google.com/app/apikey
   - Create an API key
   - Add to `.env` file

2. **Google Cloud Document AI (Optional):**
   - Create a Google Cloud Project
   - Enable Document AI API
   - Create service account credentials
   - Download JSON key file

## 🏃‍♂️ Running the Service

### Development Mode (Recommended)

```bash
./start_ccms_backend.sh
```

The service will start at: http://localhost:5001

### Manual Run

```bash
source .venv/bin/activate
cd api
python app.py
```

## 🧪 Testing the Service

Once running, test the health endpoint:

```bash
curl http://localhost:5001/health
```

## 📁 Project Structure

```
backend/
├── .venv/                 # Virtual environment (auto-generated)
├── api/                   # Flask web application
│   ├── app.py            # Main Flask app
│   └── service_endpoints.py  # API routes
├── services/              # Document processing services
│   ├── hybrid_rag_classification_service.py  # Main classification service
│   ├── category_mapping_service.py          # Category mapping
│   ├── document_processing_orchestrator.py  # Processing workflow
│   ├── ocr_service.py    # OCR processing
│   ├── llm_service.py    # LLM integration
│   └── ...               # Other services
├── ../classifier/        # Core classification modules (shared at project root)
│   ├── hybrid_rag.py     # Hybrid RAG implementation
│   ├── pure_llm.py       # Pure LLM classification
│   ├── pdf_extractor.py  # PDF processing
│   ├── embeddings.py     # Embedding management
│   └── ...               # Other classification modules
├── ../data/               # Shared data directory (at project root)
├── requirements.txt       # Python dependencies
├── config.yaml           # Configuration
├── .env                  # Environment configuration
├── setup.sh              # One-command setup script
├── start_ccms_backend.sh    # Main service startup script
├── run_backend.sh        # Alternative run script
└── README.md             # This file
```

## 🔧 Integration Notes

- This backend service is self-contained and can be integrated into any application
- All imports are relative to the backend directory
- Configuration is centralized in `config.yaml`
- API endpoints are documented in `api/README_API.md`

## 🛠️ Development

### Installing New Dependencies

```bash
source .venv/bin/activate
pip install package_name
pip freeze > requirements.txt  # Update requirements
```

### Running in Production

For production, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn --bind 0.0.0.0:5001 --workers 4 api.app:app
```

## 🐛 Troubleshooting

### Common Issues

1. **ImportError: No module named 'cv2'**
   - Solution: System dependencies missing, install tesseract-ocr and poppler-utils

2. **API Key not configured**
   - Solution: Set `GOOGLE_API_KEY` in `.env` file

3. **Permission denied on scripts**
   - Solution: `chmod +x setup.sh run_backend.sh`

4. **Virtual environment issues**
   - Solution: Delete `.venv` folder and run `./setup.sh` again

### Logs

Check logs for debugging:
- Application logs: `backend.log`
- Flask development server logs in terminal

## 📚 API Documentation

Main endpoints:
- `GET /health` - Health check
- `POST /document/process` - Process document
- `POST /document/extract` - Extract text from PDF
- `POST /llm/classify` - Classify document type

For detailed API documentation, see the main project documentation.

## 🤝 Contributing

1. Follow the setup instructions above
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## 📄 License

This project is part of the CCMS application suite.