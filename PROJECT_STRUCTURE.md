# CCMS Classification Project Structure

## 📁 Clean Organized Structure

```
ccms_classification/
├── 🎯 backend/                    # CORE INTEGRATION FILES
│   ├── api/                      # REST API endpoints
│   ├── services/                 # Business logic (imports from ../classifier/)
│   ├── requirements.txt          # Production dependencies
│   ├── start_integrated_backend.sh
│   └── README.md                 # Integration guide
│
├── 🔧 classifier/                # SHARED CLASSIFICATION MODULES
│   ├── hybrid_rag.py            # Core classification algorithms
│   ├── embeddings.py            # Vector operations
│   ├── pdf_extractor.py         # PDF processing
│   └── ...                      # Other classification modules
│
├── 📚 docs/                      # DOCUMENTATION
│   ├── *.md                     # Technical documentation
│   └── REQUIREMENTS_GUIDE.md    # Dependency documentation
│
├── 🧪 tests_standalone/          # DEVELOPMENT & TESTING
│   ├── test_*.py                # Test files
│   ├── debug_*.py               # Debug utilities
│   ├── requirements_*.txt       # Development dependencies
│   └── original_tests/          # Original test structure
│
├── 🔧 temp_tests/               # TEMPORARY UTILITIES
│   ├── check_issue_types.py    # Utility scripts
│   ├── lot21_batch_test.py     # Batch processing tests
│   ├── process_*.sh            # Shell processing scripts
│   └── README.md               # Temp files documentation
│
├── 🗃️ data/                     # DATA & EMBEDDINGS
│   ├── embeddings/             # Vector embeddings
│   ├── synthetic/              # Synthetic training data
│   └── backups/                # Data backups
│
├── ⚙️ CONFIG & DEPLOYMENT
│   ├── config.yaml             # Main configuration
│   ├── batch_config.yaml       # Batch processing config
│   ├── Dockerfile              # Container configuration
│   ├── docker-compose.yml      # Multi-container setup
│   ├── service.sh              # Service management
│   └── start_integrated_backend.sh  # Main startup script
│
├── 📊 DATA FILES
│   ├── *.xlsx                  # Mapping and analysis files
│   └── issue_category_mapping_diffs/  # Mapping analysis
│
└── 🔗 INTEGRATION
    ├── requirements.txt        # Points to backend/requirements.txt
    ├── integrated_backend/     # Current working backend
    └── integrated_backend.backup/  # Backup of integrated backend
```

## 🎯 For Integration Use

**Essential Directories:**
- `backend/` - API and service layer
- `classifier/` - Core classification algorithms
- `data/` - Embeddings and training data (optional, can regenerate)

**Optional:**
- `docs/` - Documentation reference
- `config.yaml` - Configuration customization

## 🧪 For Development Use

**Development Directories:**
- `tests_standalone/` - Comprehensive testing utilities
- `temp_tests/` - Temporary scripts and batch processing
- `data/` - Training data and embeddings

## 🚀 Quick Commands

### Integration Setup
```bash
pip install -r requirements.txt    # Install dependencies
./start_integrated_backend.sh      # Start service
```

### Development Setup
```bash
pip install -r tests_standalone/requirements_*.txt  # Dev dependencies
python tests_standalone/test_*.py   # Run specific tests
```

### Service Management
```bash
./service.sh start    # Start service
./service.sh status   # Check status
./service.sh stop     # Stop service
```

## 📝 Maintenance Notes

- **Keep** `backend/` lean and focused on integration
- **Use** `tests_standalone/` for all development activities
- **Document** new features in `docs/`
- **Archive** completed utilities from `temp_tests/`

This structure provides clear separation of concerns and makes it easy to identify exactly what's needed for different use cases.