#!/usr/bin/env python3
"""
Production startup script for Contract Correspondence Classification API
Handles initialization, health checks, and graceful startup
"""

import asyncio
import sys
import os
import time
import logging
from pathlib import Path
import uvicorn
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_requirements():
    """Check if all required dependencies are installed."""
    logger.info("🔍 Checking system requirements...")
    
    required_modules = [
        'fastapi', 'uvicorn', 'pandas', 'numpy', 'sklearn', 
        'sentence_transformers', 'openai', 'transformers'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"❌ Missing required modules: {missing_modules}")
        logger.error("Install with: pip install -r requirements_production.txt")
        return False
    
    logger.info("✅ All required modules found")
    return True


def check_data_files():
    """Check if required data files exist."""
    logger.info("📁 Checking data files...")
    
    required_files = [
        './data/synthetic/combined_training_data.xlsx',
        './data/raw/Consolidated_labeled_data.xlsx',
        './config.yaml',
        './.env'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    # At least one training data file should exist
    training_files_exist = any(
        Path(f).exists() for f in [
            './data/synthetic/combined_training_data.xlsx',
            './data/raw/Consolidated_labeled_data.xlsx'
        ]
    )
    
    if not training_files_exist:
        logger.error("❌ No training data found. Please ensure one of these exists:")
        logger.error("  - ./data/synthetic/combined_training_data.xlsx")
        logger.error("  - ./data/raw/Consolidated_labeled_data.xlsx")
        return False
    
    if not Path('./.env').exists():
        logger.warning("⚠️  .env file not found. Create one with your API keys:")
        logger.warning("  OPENAI_API_KEY=your-key-here")
        logger.warning("  ANTHROPIC_API_KEY=your-key-here")
    
    logger.info("✅ Essential data files found")
    return True


def setup_directories():
    """Create necessary directories."""
    logger.info("📁 Setting up directories...")
    
    directories = [
        'logs',
        'data/embeddings',
        'data/models',
        'data/backups',
        'data/processed'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Directories created")


def check_api_keys():
    """Check if API keys are configured."""
    logger.info("🔑 Checking API configuration...")
    
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not openai_key and not anthropic_key:
        logger.warning("⚠️  No LLM API keys found. Only RAG-based classification will work.")
        logger.warning("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file for full functionality")
        return False
    
    if openai_key and openai_key.startswith('sk-'):
        logger.info("✅ OpenAI API key found")
    
    if anthropic_key and anthropic_key.startswith('sk-'):
        logger.info("✅ Anthropic API key found")
    
    return True


def pre_startup_checks():
    """Run all pre-startup checks."""
    logger.info("🚀 Starting Contract Correspondence Classification API...")
    logger.info("=" * 60)
    
    checks = [
        ("Requirements", check_requirements),
        ("Data Files", check_data_files),
        ("Directories", setup_directories),
        ("API Keys", check_api_keys)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        try:
            if callable(check_func):
                result = check_func()
                if result is False:
                    all_passed = False
            else:
                check_func()
        except Exception as e:
            logger.error(f"❌ {check_name} check failed: {e}")
            all_passed = False
    
    logger.info("=" * 60)
    
    if not all_passed:
        logger.error("❌ Pre-startup checks failed. Please fix the issues above.")
        return False
    
    logger.info("✅ All pre-startup checks passed!")
    return True


def get_server_config():
    """Get server configuration based on environment."""
    env = os.getenv('ENVIRONMENT', 'development')
    
    if env == 'production':
        return {
            "host": "0.0.0.0",
            "port": int(os.getenv('PORT', 8000)),
            "workers": int(os.getenv('WORKERS', 4)),
            "log_level": "info",
            "access_log": True,
            "loop": "auto",
            "reload": False
        }
    else:
        return {
            "host": "127.0.0.1",
            "port": 8000,
            "log_level": "debug",
            "reload": True,
            "access_log": True
        }


def main():
    """Main startup function."""
    try:
        # Run pre-startup checks
        if not pre_startup_checks():
            sys.exit(1)
        
        # Get server configuration
        config = get_server_config()
        
        logger.info(f"🌐 Starting server on {config['host']}:{config['port']}")
        logger.info("📖 API Documentation will be available at:")
        logger.info(f"   http://{config['host']}:{config['port']}/docs")
        logger.info(f"   http://{config['host']}:{config['port']}/redoc")
        
        # Start the server
        uvicorn.run(
            "production_api:app",
            **config
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Received shutdown signal")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()