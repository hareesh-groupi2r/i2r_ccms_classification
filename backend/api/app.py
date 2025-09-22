"""
Flask Application for Modular Document Processing Services
Serves individual service endpoints and orchestrator
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from parent directory
parent_dir = Path(__file__).parent.parent
env_path = parent_dir / '.env'
load_dotenv(dotenv_path=env_path)

# Add parent directory to path to import services
sys.path.append(str(parent_dir))

from service_endpoints import service_api


def create_app(config_name='default'):
    """Application factory"""
    app = Flask(__name__)
    
    # Enable CORS for all routes
    cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
    CORS(app, origins=cors_origins)
    
    # Configuration from environment variables
    max_content = int(os.getenv('MAX_CONTENT_LENGTH', 200)) * 1024 * 1024
    app.config['MAX_CONTENT_LENGTH'] = max_content
    app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', '/tmp')
    
    # Logging configuration with timestamps
    if config_name != 'testing':
        # Configure logging format with timestamps
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        app.logger.setLevel(logging.INFO)
        
        # Force stdout/stderr to be unbuffered for real-time logging
        import sys
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    
    # Register blueprints
    app.register_blueprint(service_api)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": "Endpoint not found",
            "message": "The requested endpoint does not exist"
        }), 404
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            "error": "Bad request",
            "message": "The request was malformed or invalid"
        }), 400
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            "error": "File too large",
            "message": "The uploaded file exceeds the maximum size limit (200MB)"
        }), 413
    
    @app.errorhandler(500)
    def internal_server_error(error):
        app.logger.error(f"Internal server error: {error}")
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }), 500
    
    # Root endpoint
    @app.route('/')
    def index():
        return jsonify({
            "name": "CCMS Document Processing Services",
            "version": "1.0.0",
            "description": "Modular document processing services for Contract Correspondence Management System",
            "services": {
                "document_type": "Document type classification",
                "ocr": "Text extraction from documents", 
                "llm": "Structured data extraction and content classification",
                "category_mapping": "Issue to category mapping",
                "hybrid_rag_classification": "Hybrid RAG document classification with categories and issue types",
                "orchestrator": "Coordinated document processing pipeline"
            },
            "endpoints": {
                "health": "/api/services/health",
                "config": "/api/services/config/*",
                "services": "/api/services/*"
            }
        })
    
    # API documentation endpoint
    @app.route('/api')
    def api_documentation():
        return jsonify({
            "api_version": "v1",
            "base_url": "/api/services",
            "endpoints": {
                "document_type": {
                    "classify": "POST /document-type/classify - Classify document from file",
                    "classify_text": "POST /document-type/classify-text - Classify document from text"
                },
                "ocr": {
                    "extract": "POST /ocr/extract - Extract text from document",
                    "extract_pages": "POST /ocr/extract-pages - Extract text from specific pages",
                    "methods": "GET /ocr/methods - Get available OCR methods"
                },
                "llm": {
                    "extract_structured": "POST /llm/extract-structured - Extract structured data",
                    "classify_content": "POST /llm/classify-content - Classify content",
                    "status": "GET /llm/status - Get LLM service status"
                },
                "category_mapping": {
                    "map_issue": "POST /category-mapping/map-issue - Map issue to category",
                    "bulk_classify": "POST /category-mapping/bulk-classify - Classify multiple issues",
                    "categories": "GET /category-mapping/categories - Get available categories",
                    "statistics": "GET /category-mapping/statistics - Get mapping statistics",
                    "add_mapping": "POST /category-mapping/add-mapping - Add new mapping",
                    "remove_mapping": "DELETE /category-mapping/remove-mapping - Remove mapping"
                },
                "hybrid_rag_classification": {
                    "classify_document": "POST /hybrid-rag-classification/classify-document - Classify document by ID",
                    "classify_text": "POST /hybrid-rag-classification/classify-text - Classify text content",
                    "classify_batch": "POST /hybrid-rag-classification/classify-batch - Batch classify multiple texts",
                    "categories": "GET /hybrid-rag-classification/categories - Get available categories",
                    "issue_types": "GET /hybrid-rag-classification/issue-types - Get available issue types",
                    "status": "GET /hybrid-rag-classification/status - Get classification service status"
                },
                "issue_types": {
                    "list": "GET /issue-types - Get all issue types",
                    "get": "GET /issue-types/{id} - Get specific issue type"
                },
                "issue_categories": {
                    "list": "GET /issue-categories - Get all issue categories",
                    "get": "GET /issue-categories/{id} - Get specific issue category",
                    "by_issue_type": "GET /issue-categories/by-issue-type/{issue_type_id} - Get categories for issue type"
                },
                "orchestrator": {
                    "process_document": "POST /orchestrator/process-document - Full document processing",
                    "process_partial": "POST /orchestrator/process-partial - Partial processing",
                    "status": "GET /orchestrator/status - Get orchestrator status"
                },
                "config": {
                    "validate": "GET /config/validate - Validate all configurations",
                    "services": "GET /config/services - Get all service configs",
                    "service_config": "GET/PUT /config/services/{service_name} - Get/Update service config"
                },
                "health": "GET /health - Health check for all services"
            }
        })
    
    return app


# For running directly
if __name__ == '__main__':
    app = create_app()
    # Get host and port from environment variables with defaults
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5001))
    debug = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'
    
    # Disable reloader to avoid dual processes in production-like environments
    use_reloader = os.getenv('FLASK_USE_RELOADER', 'false').lower() == 'true'
    
    app.run(debug=debug, host=host, port=port, use_reloader=use_reloader)