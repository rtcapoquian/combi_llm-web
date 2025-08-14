#!/usr/bin/env python3
"""
Startup script for Eco-Sort AI Web Interface
This script starts the Flask backend server that serves the web interface
and handles YOLO + LLM inference requests.
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path
import argparse

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'web_backend.py',
        'index.html',
        'css/styles.css', 
        'js/script.js',
        'yolov11_infer_openvino.py',
        'tools.py',
        'app.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files found")
    return True

def check_models():
    """Check if required models exist"""
    model_paths = [
        'best.pt',  # YOLO model
        'model/qwen2-1.5B-INT4',  # LLM model
        'model/bge-small-FP32'    # Embedding model
    ]
    
    missing_models = []
    for model_path in model_paths:
        if not Path(model_path).exists():
            missing_models.append(model_path)
    
    if missing_models:
        print("⚠️  Some models are missing (will use fallback mode):")
        for model_path in missing_models:
            print(f"   - {model_path}")
    else:
        print("✅ All models found")
    
    return len(missing_models) == 0

def install_requirements():
    """Install Python requirements"""
    requirements = [
        'flask',
        'flask-cors', 
        'pillow',
        'numpy',
        'websockets',
        'ultralytics',
        'openvino'
    ]
    
    print("📦 Installing Python requirements...")
    try:
        for req in requirements:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def start_backend(host='127.0.0.1', port=5000, debug=False):
    """Start the Flask backend server"""
    print(f"🚀 Starting Eco-Sort AI Backend on {host}:{port}")
    print("📊 Loading AI models (this may take a few minutes)...")
    
    try:
        # Import and run the backend
        from web_backend import EcoSortBackend
        
        backend = EcoSortBackend()
        
        # Open web browser after a short delay
        if not debug:
            def open_browser():
                time.sleep(2)
                url = f"http://{host}:{port}"
                print(f"🌐 Opening web browser: {url}")
                webbrowser.open(url)
            
            import threading
            threading.Timer(2.0, open_browser).start()
        
        print("✅ Backend server starting...")
        backend.run(host=host, port=port, debug=debug)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Eco-Sort AI: Verbose Real-Time Waste Classification Web Interface'
    )
    parser.add_argument('--host', default='127.0.0.1', 
                       help='Host to bind server to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, 
                       help='Port to bind server to (default: 5000)')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode (default: False)')
    parser.add_argument('--install-deps', action='store_true',
                       help='Install Python dependencies before starting')
    parser.add_argument('--skip-browser', action='store_true',
                       help='Skip opening web browser automatically')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🌱 Eco-Sort AI: Verbose Real-Time Waste Classification")
    print("   Using YOLOv11 + OpenVINO + Qwen2 + RAG")
    print("=" * 60)
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_requirements():
            return 1
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please ensure all required files are present")
        return 1
    
    # Check models (warnings only)
    check_models()
    
    # Set environment variables for better performance
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['OPENVINO_LOG_LEVEL'] = '1'  # Reduce OpenVINO logging
    
    print(f"\n🎯 Configuration:")
    print(f"   - Host: {args.host}")
    print(f"   - Port: {args.port}")
    print(f"   - Debug: {args.debug}")
    print(f"   - Auto-open browser: {not args.skip_browser}")
    
    print(f"\n📋 Features:")
    print(f"   - 📷 Real-time camera capture")
    print(f"   - 🖼️  Image upload support")
    print(f"   - 🤖 YOLOv11 + OpenVINO inference")
    print(f"   - 🧠 LLM-powered guidance")
    print(f"   - 📝 RAG-enhanced responses")
    print(f"   - ♻️  Recycling tracking")
    
    print(f"\n⚡ Starting server...")
    
    try:
        success = start_backend(
            host=args.host, 
            port=args.port, 
            debug=args.debug
        )
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0

if __name__ == '__main__':
    sys.exit(main())
