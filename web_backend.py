#!/usr/bin/env python3
"""
Flask Backend for Eco-Sort AI Web Interface
Integrates YOLOv11 OpenVINO inference with LLM guidance using existing components.
"""

import os
import sys
import json
import time
import base64
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from io import BytesIO
import asyncio
import websockets
from threading import Thread
import uuid

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np

# Import existing modules
from yolov11_infer_openvino import predict_and_show, load_optimized_model
from tools import WasteClassifier, RecyclingCart
from app import setup_models, setup_tools, load_documents
from llama_index.core import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EcoSortBackend:
    def __init__(self):
        self.app = Flask(__name__, static_folder='.')
        CORS(self.app)
        
        # Initialize models
        self.yolo_model = None
        self.llm_agent = None
        self.llm = None
        self.embedding = None
        self.vector_tool = None
        self.waste_tools = None
        self.is_initialized = False
        
        # WebSocket connections
        self.ws_connections = set()
        
        self.setup_routes()
        self.setup_websocket_server()
    
    def initialize_models(self):
        """Initialize YOLO and LLM models"""
        try:
            logger.info("Initializing models...")
            
            # Initialize YOLO model with OpenVINO
            self.yolo_model, self.engine_type = load_optimized_model(
                "best.pt", 
                use_openvino=True
            )
            logger.info(f"YOLO model loaded with {self.engine_type}")
            
            # Initialize LLM models and agent
            try:
                llm_model_path = Path("model/qwen2-1.5B-INT4")
                embedding_model_path = Path("model/bge-small-FP32") 
                data_folder_path = Path("data")
                
                if llm_model_path.exists() and embedding_model_path.exists():
                    self.llm, self.embedding = setup_models(
                        llm_model_path, 
                        embedding_model_path, 
                        "CPU"
                    )
                    
                    Settings.embed_model = self.embedding
                    Settings.llm = self.llm
                    
                    # Setup tools and agent
                    self.waste_tools = setup_tools()
                    index = load_documents(data_folder_path)
                    
                    # Import necessary components for agent
                    from llama_index.core.tools import QueryEngineTool, ToolMetadata
                    from llama_index.core.agent import ReActAgent
                    from llama_index.core import PromptTemplate
                    from llama_index.core.agent import ReActChatFormatter
                    from system_prompt import react_system_header_str
                    import nest_asyncio
                    
                    # Create vector search tool
                    self.vector_tool = QueryEngineTool(
                        index.as_query_engine(streaming=True),
                        metadata=ToolMetadata(
                            name="vector_search",
                            description="""Use this tool for ANY question about waste management, recycling procedures, disposal guidelines, or environmental information from the knowledge base.""",
                        ),
                    )
                    
                    nest_asyncio.apply()
                    
                    # Create the ReActAgent with all tools
                    all_tools = list(self.waste_tools) + [self.vector_tool]
                    self.llm_agent = ReActAgent.from_tools(
                        all_tools,
                        llm=self.llm,
                        max_iterations=5,
                        verbose=True,
                        react_chat_formatter=ReActChatFormatter.from_defaults(),
                    )
                    
                    # Set custom system prompt
                    react_system_prompt = PromptTemplate(react_system_header_str)
                    self.llm_agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})
                    self.llm_agent.reset()
                    
                    logger.info("LLM models and agent initialized successfully")
                    self.is_initialized = True
                else:
                    logger.warning("LLM model paths not found, using basic classification only")
                    self.is_initialized = True
                    
            except Exception as e:
                logger.error(f"Failed to initialize LLM models: {e}")
                logger.info("Continuing with YOLO-only mode")
                self.is_initialized = True
                
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return send_from_directory('.', 'index.html')
        
        @self.app.route('/<path:filename>')
        def static_files(filename):
            return send_from_directory('.', filename)
        
        @self.app.route('/api/status', methods=['GET'])
        def get_status():
            return jsonify({
                'yolo_ready': self.yolo_model is not None,
                'llm_ready': self.llm_agent is not None,
                'initialized': self.is_initialized,
                'engine_type': getattr(self, 'engine_type', 'Not loaded')
            })
        
        @self.app.route('/api/classify', methods=['POST'])
        def classify_waste():
            try:
                if not self.is_initialized:
                    return jsonify({'error': 'Models not initialized'}), 500
                
                # Get image from request
                if 'image' not in request.files:
                    return jsonify({'error': 'No image provided'}), 400
                
                image_file = request.files['image']
                if image_file.filename == '':
                    return jsonify({'error': 'No image selected'}), 400
                
                # Process image
                start_time = time.time()
                
                # Save temporary image
                temp_path = f"temp_{uuid.uuid4().hex}.jpg"
                image_file.save(temp_path)
                
                try:
                    # Run YOLO inference
                    detections = self.run_yolo_inference(temp_path)
                    
                    processing_time = (time.time() - start_time) * 1000
                    
                    return jsonify({
                        'success': True,
                        'detections': detections,
                        'processing_time_ms': processing_time,
                        'engine_type': self.engine_type
                    })
                
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            except Exception as e:
                logger.error(f"Classification error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/guidance', methods=['POST'])
        def get_guidance():
            try:
                data = request.get_json()
                detections = data.get('detections', [])
                
                if not detections:
                    return jsonify({'error': 'No detections provided'}), 400
                
                guidance = self.get_llm_guidance(detections)
                
                return jsonify({
                    'success': True,
                    'guidance': guidance
                })
            
            except Exception as e:
                logger.error(f"Guidance error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/chat', methods=['POST'])
        def chat_with_llm():
            try:
                data = request.get_json()
                query = data.get('query', '')
                
                if not query:
                    return jsonify({'error': 'No query provided'}), 400
                
                if self.llm_agent:
                    # Use the LLM agent to answer the question
                    try:
                        response = self.llm_agent.chat(query)
                        response_text = str(response.response) if hasattr(response, 'response') else str(response)
                        
                        return jsonify({
                            'success': True,
                            'response': response_text,
                            'source': 'LLM Agent with RAG'
                        })
                    except Exception as e:
                        logger.error(f"LLM chat error: {e}")
                        return jsonify({'error': f'LLM processing failed: {str(e)}'}), 500
                else:
                    return jsonify({
                        'success': False,
                        'error': 'LLM agent not available',
                        'response': 'The AI assistant is not currently available. Please try again later.'
                    }), 503
            
            except Exception as e:
                logger.error(f"Chat error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/recycling/add', methods=['POST'])
        def add_to_recycling():
            try:
                data = request.get_json()
                item_name = data.get('item_name')
                category = data.get('category')
                quantity = data.get('quantity', 1)
                notes = data.get('notes', '')
                
                if not item_name or not category:
                    return jsonify({'error': 'Missing required fields'}), 400
                
                # Add to recycling cart
                result = RecyclingCart.add_to_recycling_list(
                    item_name, category, quantity, notes
                )
                
                return jsonify({
                    'success': True,
                    'message': result,
                    'recycling_list': RecyclingCart.get_recycling_items()
                })
            
            except Exception as e:
                logger.error(f"Add to recycling error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/recycling/list', methods=['GET'])
        def get_recycling_list():
            try:
                items = RecyclingCart.get_recycling_items()
                return jsonify({
                    'success': True,
                    'items': items
                })
            
            except Exception as e:
                logger.error(f"Get recycling list error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/recycling/clear', methods=['POST'])
        def clear_recycling_list():
            try:
                result = RecyclingCart.clear_recycling_list()
                return jsonify({
                    'success': True,
                    'message': result
                })
            
            except Exception as e:
                logger.error(f"Clear recycling list error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def run_yolo_inference(self, image_path: str) -> List[Dict]:
        """Run YOLO inference on image and return detections"""
        try:
            if not self.yolo_model:
                raise ValueError("YOLO model not initialized")
            
            # Run inference
            results = self.yolo_model(
                image_path,
                imgsz=640,
                conf=0.25,
                iou=0.5,
                device='cpu',
                verbose=False
            )
            
            detections = []
            
            # Process results
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    cls = r.boxes.cls.int().cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    
                    for i, (box, class_id, conf) in enumerate(zip(xyxy, cls, confs)):
                        class_name = r.names[int(class_id)]
                        
                        # Map YOLO classes to waste categories
                        category = self.map_class_to_category(class_name)
                        
                        detection = {
                            'class': class_name,
                            'confidence': float(conf),
                            'bbox': [float(x) for x in box],  # [x1, y1, x2, y2]
                            'category': category
                        }
                        
                        detections.append(detection)
            
            logger.info(f"YOLO inference completed: {len(detections)} detections")
            return detections
        
        except Exception as e:
            logger.error(f"YOLO inference error: {e}")
            raise
    
    def map_class_to_category(self, class_name: str) -> str:
        """Map YOLO class names to waste categories"""
        # Define mapping from YOLO classes to waste categories
        class_mapping = {
            'bottle': 'Recyclable Plastic',
            'plastic': 'Recyclable Plastic',
            'can': 'Metal Recyclable',
            'phone': 'E-Waste',
            'laptop': 'E-Waste',
            'computer': 'E-Waste',
            'electronic': 'E-Waste',
            'paper': 'Paper Recyclable',
            'cardboard': 'Paper Recyclable',
            'glass': 'Glass Recyclable',
            'battery': 'Hazardous Waste',
            'organic': 'Organic Waste',
            'food': 'Organic Waste'
        }
        
        # Check for partial matches
        class_lower = class_name.lower()
        for key, category in class_mapping.items():
            if key in class_lower:
                return category
        
        return 'General Waste'
    
    def get_llm_guidance(self, detections: List[Dict]) -> List[Dict]:
        """Get LLM guidance for detected items using the actual ReActAgent"""
        try:
            guidance_list = []
            
            for detection in detections:
                class_name = detection['class']
                category = detection['category']
                confidence = detection['confidence'] * 100
                
                logger.info(f"Processing guidance for: {class_name} (LLM agent available: {self.llm_agent is not None})")
                
                if self.llm_agent is not None:
                    # Use the actual LLM agent to generate intelligent responses
                    try:
                        # Construct a detailed query for the LLM agent
                        query = f"""I have detected a {class_name} (classified as {category}) with {confidence:.1f}% confidence in an image for waste classification. 

Please provide comprehensive and specific guidance for this waste item. I need detailed information including:

1. **Recycling Instructions**: Specific steps for properly recycling this exact type of {class_name}
2. **Disassembly Steps**: If applicable, provide detailed step-by-step instructions for safely disassembling this {class_name} 
3. **Safety Precautions**: Specific safety warnings and protective measures for handling this {class_name}
4. **Environmental Impact**: Information about the environmental benefits of proper disposal
5. **Local Disposal**: Recommendations for where to dispose of this type of waste
6. **Special Requirements**: Any special handling, preparation, or processing requirements

Please make your response detailed, practical, and specific to this exact type of waste item ({class_name}). Use any relevant information from the knowledge base about waste management regulations and procedures."""
                        
                        # Get response from the LLM agent
                        logger.info(f"Querying LLM agent for detailed guidance on {class_name}...")
                        response = self.llm_agent.chat(query)
                        
                        # Extract the response text
                        response_text = str(response.response) if hasattr(response, 'response') else str(response)
                        logger.info(f"LLM agent response length: {len(response_text)} characters")
                        
                        # Parse the LLM response to extract structured information
                        guidance = self.parse_llm_response(detection, response_text)
                        guidance['source'] = 'LLM Agent with RAG (Qwen2)'
                        
                        logger.info(f"Successfully generated LLM guidance for {class_name}")
                        
                    except Exception as e:
                        logger.error(f"LLM agent error for {class_name}: {e}")
                        logger.info(f"Falling back to enhanced basic guidance for {class_name}")
                        # Fall back to enhanced basic guidance
                        guidance = self.get_enhanced_basic_guidance(detection)
                        guidance['source'] = 'Enhanced Basic Guidance (LLM Failed)'
                else:
                    logger.warning(f"No LLM agent available, using enhanced basic guidance for {class_name}")
                    # No LLM agent available, use enhanced basic guidance
                    guidance = self.get_enhanced_basic_guidance(detection)
                    guidance['source'] = 'Enhanced Basic Guidance (No LLM)'
                
                guidance_list.append(guidance)
            
            logger.info(f"Generated LLM guidance for {len(guidance_list)} items")
            return guidance_list
        
        except Exception as e:
            logger.error(f"LLM guidance error: {e}")
            # Return enhanced basic guidance on error
            return [self.get_enhanced_basic_guidance(det) for det in detections]
    
    def parse_llm_response(self, detection: Dict, response_text: str) -> Dict:
        """Parse LLM response into structured guidance format"""
        class_name = detection['class'].replace('_', ' ').title()
        
        logger.info(f"Parsing LLM response for {class_name}")
        logger.info(f"Response text preview: {response_text[:200]}..." if len(response_text) > 200 else f"Response text: {response_text}")
        
        # Split the response into sections (basic parsing)
        sections = response_text.split('\n\n')
        
        # Extract disassembly steps (look for numbered lists or bullet points)
        disassembly_steps = []
        safety_warnings = []
        main_content = response_text
        
        # Simple parsing to extract lists
        lines = response_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Identify section headers
            if any(keyword in line.lower() for keyword in ['disassembly', 'steps', 'instructions']):
                current_section = 'disassembly'
                logger.debug(f"Found disassembly section: {line}")
                continue
            elif any(keyword in line.lower() for keyword in ['safety', 'precaution', 'warning']):
                current_section = 'safety'
                logger.debug(f"Found safety section: {line}")
                continue
            elif line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•', '*')):
                if current_section == 'disassembly':
                    disassembly_steps.append(line.lstrip('123456789.-•* '))
                    logger.debug(f"Added disassembly step: {line}")
                elif current_section == 'safety':
                    safety_warnings.append(line.lstrip('123456789.-•* '))
                    logger.debug(f"Added safety warning: {line}")
        
        # If no specific steps found, provide general ones
        if not disassembly_steps:
            disassembly_steps = ['Follow the detailed guidance provided above']
            logger.info(f"No specific disassembly steps found, using default")
            
        if not safety_warnings:
            safety_warnings = ['Follow standard safety precautions as detailed in the guidance']
            logger.info(f"No specific safety warnings found, using default")
        
        logger.info(f"Parsed {len(disassembly_steps)} disassembly steps and {len(safety_warnings)} safety warnings")
        
        return {
            'detection': detection,
            'title': f'{class_name} - AI-Generated Disposal Guide',
            'content': response_text,
            'disassembly': disassembly_steps[:10],  # Limit to 10 steps
            'safety': safety_warnings[:10],  # Limit to 10 warnings
            'category': detection['category'],
            'recycling_instructions': 'See detailed AI guidance above',
            'source': 'LLM Agent with RAG'
        }
    
    def get_enhanced_basic_guidance(self, detection: Dict) -> Dict:
        """Enhanced basic guidance when LLM is not available"""
        class_name = detection['class']
        category = detection['category']
        confidence = detection['confidence'] * 100
        
        # Get basic classification info from tools
        classification_result = WasteClassifier.classify_waste_type(
            class_name, 
            material_type=category
        )
        
        # Get disassembly guidance if applicable
        disassembly_result = WasteClassifier.get_disassembly_guidance(
            class_name,
            safety_level="detailed"
        )
        
        # Enhanced content generation
        class_title = class_name.replace('_', ' ').title()
        
        content = f"""🎯 **AI Detection Results**: Detected {class_title} with {confidence:.1f}% confidence
        
♻️ **Waste Category**: {category}

📋 **Specific Instructions**: {classification_result.get('recycling_instructions', 'Check local guidelines for this specific item type')}

🔍 **Item Analysis**: This {class_title} requires {category.lower()} processing procedures.

🌍 **Environmental Benefits**: Proper recycling of {class_title} helps:
- Reduce landfill waste and environmental pollution
- Conserve natural resources and energy
- Support circular economy principles
- Minimize carbon footprint

💡 **Local Resources**: Contact your municipal waste management for {category.lower()} disposal locations."""
        
        # Add component info if available
        if disassembly_result.get('recyclable_components'):
            content += f"\n\n🔧 **Recoverable Materials**: {', '.join(disassembly_result['recyclable_components'])}"
        
        return {
            'detection': detection,
            'title': f'{class_title} - Enhanced Disposal Guide',
            'content': content,
            'disassembly': disassembly_result.get('disassembly_steps', [
                f'Research {class_title}-specific disassembly procedures',
                'Gather appropriate safety equipment and tools',
                'Work in a well-ventilated, safe environment',
                'Separate components by material type for recycling'
            ]),
            'safety': disassembly_result.get('safety_warnings', [
                'Use appropriate personal protective equipment',
                f'Handle {class_title} with care to avoid injury',
                'Be aware of potential hazardous materials',
                'Follow local safety regulations and guidelines'
            ]),
            'category': category,
            'recycling_instructions': classification_result.get('recycling_instructions', 'Consult local recycling guidelines'),
            'source': 'Enhanced Classification System'
        }
    
    
    def setup_websocket_server(self):
        """Setup WebSocket server for real-time communication"""
        async def handle_websocket(websocket, path):
            self.ws_connections.add(websocket)
            logger.info(f"WebSocket client connected: {websocket.remote_address}")
            
            try:
                async for message in websocket:
                    await self.handle_websocket_message(websocket, message)
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
            finally:
                self.ws_connections.discard(websocket)
        
        def start_websocket_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            start_server = websockets.serve(handle_websocket, "localhost", 8765)
            logger.info("WebSocket server started on ws://localhost:8765")
            loop.run_until_complete(start_server)
            loop.run_forever()
        
        # Start WebSocket server in separate thread
        ws_thread = Thread(target=start_websocket_server, daemon=True)
        ws_thread.start()
    
    async def handle_websocket_message(self, websocket, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'analyze_image':
                # Process image analysis request
                image_data = data.get('image')
                if image_data:
                    await self.process_websocket_image(websocket, image_data)
            
        except Exception as e:
            logger.error(f"WebSocket message error: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    async def process_websocket_image(self, websocket, image_data):
        """Process image analysis via WebSocket"""
        try:
            # Send status update
            await websocket.send(json.dumps({
                'type': 'status_update',
                'message': 'Processing image...',
                'progress': 10
            }))
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            temp_path = f"temp_ws_{uuid.uuid4().hex}.jpg"
            
            with open(temp_path, 'wb') as f:
                f.write(image_bytes)
            
            try:
                # Update status
                await websocket.send(json.dumps({
                    'type': 'status_update',
                    'message': 'Running YOLO inference...',
                    'progress': 30
                }))
                
                # Run YOLO inference
                detections = self.run_yolo_inference(temp_path)
                
                # Send detection results
                await websocket.send(json.dumps({
                    'type': 'detection_result',
                    'detections': detections
                }))
                
                # Update status
                await websocket.send(json.dumps({
                    'type': 'status_update',
                    'message': 'Getting AI guidance...',
                    'progress': 70
                }))
                
                # Get LLM guidance
                guidance = self.get_llm_guidance(detections)
                
                # Send guidance results
                await websocket.send(json.dumps({
                    'type': 'guidance_result',
                    'guidance': guidance
                }))
                
                # Final status
                await websocket.send(json.dumps({
                    'type': 'status_update',
                    'message': 'Analysis complete',
                    'progress': 100
                }))
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        except Exception as e:
            logger.error(f"WebSocket image processing error: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Run the Flask application"""
        logger.info(f"Starting Eco-Sort AI Backend on {host}:{port}")
        
        # Initialize models in a separate thread to avoid blocking startup
        init_thread = Thread(target=self.initialize_models, daemon=True)
        init_thread.start()
        
        self.app.run(host=host, port=port, debug=debug, threaded=True)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Eco-Sort AI Backend Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create and run backend
    backend = EcoSortBackend()
    backend.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
