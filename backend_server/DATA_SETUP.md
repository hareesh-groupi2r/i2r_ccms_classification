# Data Setup Guide for CCMS Classification Backend

## 📁 Data Directory Structure

The data directory is located at the **project root level** (outside of backend_server/):

```
ccms_classification/
├── backend_server/       # Backend service code
├── classifier/           # 🔧 SHARED CLASSIFICATION MODULES
│   ├── hybrid_rag.py    # Core classification algorithms
│   ├── embeddings.py    # Vector operations
│   └── ...              # Other classification modules
├── data/                # 🎯 SHARED DATA DIRECTORY
│   ├── embeddings/      # Vector databases (FAISS)
│   │   ├── rag_index.faiss
│   │   └── rag_index.pkl
│   ├── synthetic/       # Training data
│   └── backups/         # Data backups
└── other_directories/
```

## 🔄 Why Data is External to Backend

### Advantages:
- **Shared Access**: Multiple services can use same embeddings
- **Size Management**: Vector databases can be 100MB+, kept separate from code
- **Flexible Integration**: Choose to include/exclude data based on needs
- **Persistent Storage**: Data survives backend updates/changes

## 🚀 Integration Options

### Option A: Copy Backend + Classifier Only (Minimal)
```bash
# Copy backend and shared classifier modules
cp -r backend_server/ your_project/ccms_backend/
cp -r classifier/ your_project/ccms_classifier/

# Update import paths in backend to point to classifier/
# Generate embeddings from your training data
cd your_project/ccms_backend/
python -c "
import sys; sys.path.append('../ccms_classifier')
from embeddings import EmbeddingsManager
em = EmbeddingsManager()
em.build_index(your_training_data)
"
```

### Option B: Copy Complete Structure (Full Setup)
```bash
# Copy backend, classifier, and data for immediate functionality
cp -r backend_server/ your_project/ccms_backend/
cp -r classifier/ your_project/ccms_classifier/ 
cp -r data/ your_project/ccms_data/

# The backend will automatically find classifier/ and data/ at the expected relative paths
cd your_project/ccms_backend/
./start_ccms_backend.sh  # Should work immediately
```

### Option C: External Vector Database
```bash
# Use external vector database (Pinecone, Qdrant, etc.)
cp -r backend_server/ your_project/ccms_backend/

# Configure external DB in config.yaml:
# vector_db: "pinecone"  # instead of "faiss"
```

## 🛠️ Setting Up Embeddings

### For New Users (No Existing Data)

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Create Data Directory**:
   ```bash
   mkdir -p ../data/embeddings
   ```

3. **Generate Embeddings**:
   ```python
   from classifier.embeddings import EmbeddingsManager
   from pathlib import Path
   
   # Initialize embeddings manager
   em = EmbeddingsManager(cache_dir='../data/embeddings')
   
   # Build index from your training data
   em.build_index_from_excel('your_training_data.xlsx')
   ```

### For Existing Users (With Data Directory)

The backend_server will automatically use existing embeddings in `../data/embeddings/`

## ⚙️ Configuration

### Default Paths (in config.yaml)
```yaml
hybrid_rag:
  vector_db: "faiss"
  # Embeddings automatically loaded from ../data/embeddings/
```

### Custom Paths
```yaml
# Override in your config.yaml if needed
embeddings:
  cache_dir: "/path/to/your/embeddings"
  faiss_index: "custom_rag_index.faiss"
  metadata: "custom_rag_index.pkl"
```

## 🔍 Verifying Setup

### Check Data Directory
```bash
ls -la ../data/embeddings/
# Should show: rag_index.faiss, rag_index.pkl
```

### Test Embeddings Loading
```python
from classifier.embeddings import EmbeddingsManager
em = EmbeddingsManager()
print("✅ Embeddings loaded successfully!")
```

### API Health Check
```bash
# Start backend service
./start_integrated_backend.sh

# Test embeddings endpoint
curl http://localhost:5001/api/services/hybrid-rag-classification/status
```

## 📦 Data Management

### Backup Embeddings
```bash
cp -r ../data/embeddings/ ../data/backups/embeddings_$(date +%Y%m%d)/
```

### Update Embeddings
```bash
# Regenerate with new training data
python -c "
from classifier.embeddings import EmbeddingsManager
em = EmbeddingsManager()
em.rebuild_index('updated_training_data.xlsx')
"
```

### Monitor Embedding Size
```bash
du -sh ../data/embeddings/
# Typical size: 50-200MB depending on training data size
```

## 🚨 Troubleshooting

### Missing Embeddings Error
```
FileNotFoundError: rag_index.faiss not found
```
**Solution**: Generate embeddings or copy from data/ directory

### Memory Issues
```
RuntimeError: FAISS index too large
```
**Solution**: Use smaller embedding model or reduce training data size

### Path Issues
```
No such file or directory: ./data/embeddings
```
**Solution**: Ensure you're running from backend/ directory or update paths in config

## 🎯 Production Recommendations

1. **Keep data/ external** for easier maintenance
2. **Backup embeddings regularly** (they take time to regenerate)
3. **Monitor embedding size** for deployment constraints
4. **Use external vector DB** for scalable production deployments
5. **Version your training data** to track embedding changes