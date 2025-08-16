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
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# Import existing modules
from yolov11_infer_openvino import predict_and_show, load_optimized_model
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
        
        # Create annotated images folder
        self.annotated_folder = Path("annotated_images")
        self.annotated_folder.mkdir(exist_ok=True)
        
        self.setup_routes()
        self.setup_websocket_server()
        self.cleanup_old_files()
    
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
                    
                    # Create the ReActAgent with just the vector tool for simplicity
                    self.llm_agent = ReActAgent.from_tools(
                        [self.vector_tool],  # Only use vector search tool
                        llm=self.llm,
                        max_iterations=5,  # Increase iterations
                        verbose=True,     # Reduce verbosity
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
        
        @self.app.route('/annotated_images/<filename>')
        def annotated_images(filename):
            return send_from_directory('annotated_images', filename)
        
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
                    
                    # Create annotated image if detections found
                    annotated_image_path = None
                    if detections:
                        annotated_image_path = self.create_annotated_image(temp_path, detections)
                        
                        # Add annotated image path to each detection for frontend access
                        for detection in detections:
                            if annotated_image_path:
                                # Extract just the filename from the full path
                                filename = Path(annotated_image_path).name
                                detection['annotated_image_path'] = filename
                            else:
                                detection['annotated_image_path'] = None
                    
                    processing_time = time.time() - start_time  # seconds
                    
                    return jsonify({
                        'success': True,
                        'detections': detections,
                        'processing_time_s': processing_time,
                        'engine_type': self.engine_type,
                        'annotated_image': f"/annotated/{Path(annotated_image_path).name}" if annotated_image_path else None
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
                
                # Add to in-memory recycling list
                if not hasattr(self, 'recycling_items'):
                    self.recycling_items = []
                
                # Check if item already exists
                existing_item = None
                for item in self.recycling_items:
                    if item['item_name'] == item_name and item['category'] == category:
                        existing_item = item
                        break
                
                if existing_item:
                    existing_item['quantity'] += quantity
                    if notes:
                        existing_item['notes'] = f"{existing_item.get('notes', '')}; {notes}".strip('; ')
                    message = f"Updated {item_name} quantity to {existing_item['quantity']}"
                else:
                    new_item = {
                        'item_name': item_name,
                        'category': category,
                        'quantity': quantity,
                        'notes': notes,
                        'date_added': time.strftime('%Y-%m-%d %H:%M')
                    }
                    self.recycling_items.append(new_item)
                    message = f"Added {quantity} {item_name} to recycling list"
                
                return jsonify({
                    'success': True,
                    'message': message,
                    'recycling_list': self.recycling_items
                })
            
            except Exception as e:
                logger.error(f"Add to recycling error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/recycling/list', methods=['GET'])
        def get_recycling_list():
            try:
                if not hasattr(self, 'recycling_items'):
                    self.recycling_items = []
                
                return jsonify({
                    'success': True,
                    'items': self.recycling_items
                })
            
            except Exception as e:
                logger.error(f"Get recycling list error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/recycling/clear', methods=['POST'])
        def clear_recycling_list():
            try:
                if hasattr(self, 'recycling_items'):
                    self.recycling_items = []
                
                return jsonify({
                    'success': True,
                    'message': 'Recycling list cleared'
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
    
    def cleanup_old_files(self):
        """Clean up old temporary and annotated files"""
        try:
            current_time = time.time()
            
            # Clean up temp files older than 1 hour
            temp_folder = Path("temp")
            if temp_folder.exists():
                for file in temp_folder.glob("*"):
                    if current_time - file.stat().st_mtime > 3600:  # 1 hour
                        file.unlink()
                        logger.info(f"Cleaned up old temp file: {file}")
            
            # Clean up annotated images older than 24 hours
            if self.annotated_folder.exists():
                for file in self.annotated_folder.glob("annotated_*.jpg"):
                    if current_time - file.stat().st_mtime > 86400:  # 24 hours
                        file.unlink()
                        logger.info(f"Cleaned up old annotated image: {file}")
                        
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def create_annotated_image(self, image_path: str, detections: List[Dict]) -> str:
        """Create annotated image with bounding boxes and save it"""
        try:
            # Load the original image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image for drawing
            pil_image = Image.fromarray(image_rgb)
            draw = ImageDraw.Draw(pil_image)
            
            # Try to load a font, fall back to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Color map for different categories
            color_map = {
                'Recyclable Plastic': '#4CAF50',  # Green
                'E-Waste': '#FF9800',            # Orange
                'Metal Recyclable': '#2196F3',   # Blue
                'Paper Recyclable': '#9C27B0',   # Purple
                'Glass Recyclable': '#00BCD4',   # Cyan
                'Hazardous Waste': '#F44336',    # Red
                'Organic Waste': '#8BC34A',     # Light Green
                'General Waste': '#9E9E9E'      # Gray
            }
            
            # Draw bounding boxes and labels
            for i, detection in enumerate(detections):
                bbox = detection['bbox']  # [x1, y1, x2, y2]
                class_name = detection['class']
                confidence = detection['confidence']
                category = detection['category']
                
                # Get color for this category
                color = color_map.get(category, '#9E9E9E')
                
                # Convert color to RGB tuple
                color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                
                # Draw bounding box
                draw.rectangle(bbox, outline=color_rgb, width=3)
                
                # Prepare label text
                label = f"{class_name} ({confidence * 100:.1f}%)"

                
                # Get text size
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
                
                # Calculate label position
                label_x = bbox[0]
                label_y = bbox[1] - text_height - 5
                
                # Ensure label stays within image bounds
                if label_y < 0:
                    label_y = bbox[1] + 5
                
                # Draw label background
                draw.rectangle([label_x, label_y, label_x + text_width + 10, label_y + text_height + 5], 
                             fill=color_rgb, outline=color_rgb)
                
                # Draw label text
                draw.text((label_x + 5, label_y + 2), label, fill='white', font=font)
            
            # Save annotated image in the folder
            annotated_filename = f"annotated_{uuid.uuid4().hex}.jpg"
            annotated_path = self.annotated_folder / annotated_filename
            pil_image.save(annotated_path, quality=85)
            
            logger.info(f"Created annotated image: {annotated_path}")
            return str(annotated_path)
            
        except Exception as e:
            logger.error(f"Error creating annotated image: {e}")
            return None
    
    def cleanup_old_files(self):
        """Clean up old temporary and annotated files"""
        try:
            import glob
            import os
            import time
            
            # Clean up temporary files older than 1 hour
            temp_files = glob.glob("temp_*.jpg")
            current_time = time.time()
            
            for temp_file in temp_files:
                try:
                    file_age = current_time - os.path.getmtime(temp_file)
                    if file_age > 3600:  # 1 hour
                        os.remove(temp_file)
                        logger.info(f"Cleaned up old temp file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean temp file {temp_file}: {e}")
            
            # Clean up annotated images older than 24 hours
            if self.annotated_folder.exists():
                for annotated_file in self.annotated_folder.glob("annotated_*.jpg"):
                    try:
                        file_age = current_time - annotated_file.stat().st_mtime
                        if file_age > 86400:  # 24 hours
                            annotated_file.unlink()
                            logger.info(f"Cleaned up old annotated image: {annotated_file}")
                    except Exception as e:
                        logger.warning(f"Failed to clean annotated file {annotated_file}: {e}")
                        
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    def get_direct_llm_response(self, query: str) -> str:
        """Get direct response from LLM without agent wrapper"""
        try:
            if hasattr(self, 'llm') and self.llm:
                # Create a very simple, concise query
                simple_query = f"How to disassemble {query}? Give 2 short sentences only."
                
                response = self.llm.complete(simple_query)
                response_text = str(response.text) if hasattr(response, 'text') else str(response)
                
                # Clean up the response and keep it concise
                response_text = response_text.strip()
                
                # Remove repetitive text and keep only the first few sentences
                sentences = response_text.split('. ')
                if len(sentences) > 3:
                    response_text = '. '.join(sentences[:3]) + '.'
                
                # Remove common repetitive phrases
                repetitive_phrases = [
                    'please note that this answer assumes',
                    'if you have different questions',
                    'please let me know',
                    'thank you for your question',
                    'have a great day',
                    'thank you for using my service',
                    'feel free to ask'
                ]
                
                for phrase in repetitive_phrases:
                    response_text = response_text.lower().replace(phrase.lower(), '')
                
                # Capitalize first letter and clean up
                response_text = response_text.strip('. ').capitalize()
                if response_text and not response_text.endswith('.'):
                    response_text += '.'
                
                if not response_text or len(response_text) < 20:
                    # Simple fallback
                    item_name = query.replace("recycle", "").strip()
                    return f"Clean the {item_name} thoroughly and place in appropriate recycling bin. Check local recycling guidelines for specific requirements."
                
                return response_text
            else:
                # Simple fallback
                item_name = query.replace("recycle", "").strip()
                return f"Clean the {item_name} thoroughly and place in appropriate recycling bin. Check local recycling guidelines for specific requirements."
                
        except Exception as e:
            logger.error(f"Direct LLM query failed: {e}")
            item_name = query.replace("recycle", "").strip()
            return f"Clean the {item_name} thoroughly and place in appropriate recycling bin. Check local recycling guidelines for specific requirements."

    def get_llm_guidance(self, detections: List[Dict]) -> List[Dict]:
        """Get LLM guidance for detected items - processing each item individually for better results"""
        try:
            if not detections:
                return []
            
            logger.info(f"Processing guidance for {len(detections)} items individually")
            
            # Process each item individually instead of batch processing
            guidance_list = []
            for detection in detections:
                try:
                    class_name = detection['class']
                    
                    # Simple individual query
                    query = f"recycle {class_name.replace('_', ' ')}"
                    
                    # Try agent first, then fallback to direct LLM
                    if self.llm_agent is not None:
                        try:
                            simple_query = f"How to recycle {class_name.replace('_', ' ')}?"
                            response = self.llm_agent.chat(simple_query)
                            response_text = str(response.response) if hasattr(response, 'response') else str(response)
                            
                            # Check if response is adequate
                            if (len(response_text) < 20 or 
                                any(phrase in response_text.lower() for phrase in ['unable to assist', 'no tool available', 'cannot help', 'sorry'])):
                                logger.warning(f"Agent response inadequate for {class_name}, using direct LLM")
                                response_text = self.get_direct_llm_response(query)
                        except Exception as e:
                            logger.error(f"Agent query failed for {class_name}: {e}")
                            response_text = self.get_direct_llm_response(query)
                    else:
                        response_text = self.get_direct_llm_response(query)
                    
                    # Parse individual response
                    guidance = self.parse_individual_llm_response(detection, response_text)
                    guidance['source'] = 'AI Assistant'
                    guidance_list.append(guidance)
                    
                except Exception as e:
                    logger.error(f"Error processing {detection['class']}: {e}")
                    # Use basic guidance as fallback
                    guidance = self.get_enhanced_basic_guidance(detection)
                    guidance_list.append(guidance)
            
            logger.info(f"Successfully generated guidance for {len(guidance_list)} items")
            return guidance_list
                
        except Exception as e:
            logger.error(f"LLM guidance error: {e}")
            return [self.get_enhanced_basic_guidance(det) for det in detections]
    
    def parse_batch_llm_response(self, detections: List[Dict], response_text: str) -> List[Dict]:
        """Parse batch LLM response and split into individual item guidance"""
        try:
            guidance_list = []
            
            # Split response by item markers
            item_sections = []
            current_section = ""
            
            lines = response_text.split('\n')
            current_item_index = -1
            
            for line in lines:
                # Look for item markers (Item 1:, Item 2:, etc.)
                if any(marker in line.lower() for marker in ['item 1', 'item 2', 'item 3', 'item 4', 'item 5', 
                                                             'item 6', 'item 7', 'item 8', 'item 9', 'item 10']):
                    # Save previous section
                    if current_section.strip() and current_item_index >= 0:
                        item_sections.append((current_item_index, current_section.strip()))
                    
                    # Start new section
                    current_section = line + '\n'
                    # Extract item number
                    for i in range(10):
                        if f'item {i+1}' in line.lower():
                            current_item_index = i
                            break
                else:
                    current_section += line + '\n'
            
            # Add the last section
            if current_section.strip() and current_item_index >= 0:
                item_sections.append((current_item_index, current_section.strip()))
            
            # If no clear item sections found, split roughly
            if not item_sections and len(detections) > 1:
                # Split response roughly by detection count
                section_length = len(response_text) // len(detections)
                for i, detection in enumerate(detections):
                    start_idx = i * section_length
                    end_idx = (i + 1) * section_length if i < len(detections) - 1 else len(response_text)
                    section_text = response_text[start_idx:end_idx]
                    item_sections.append((i, section_text))
            
            # Process each item section
            for item_index, section_text in item_sections:
                if item_index < len(detections):
                    detection = detections[item_index]
                    guidance = self.parse_individual_llm_response(detection, section_text)
                    guidance_list.append(guidance)
            
            # If we couldn't parse sections properly, use full response for each item
            if not guidance_list:
                logger.warning("Could not parse batch response properly, using full response for each item")
                for detection in detections:
                    guidance = self.parse_individual_llm_response(detection, response_text)
                    guidance_list.append(guidance)
            
            return guidance_list
            
        except Exception as e:
            logger.error(f"Error parsing batch LLM response: {e}")
            # Fall back to individual responses
            return [self.parse_individual_llm_response(det, response_text) for det in detections]
    
    def parse_individual_llm_response(self, detection: Dict, response_text: str) -> Dict:
        """Parse individual LLM response into clean, structured guidance format"""
        class_name = detection['class'].replace('_', ' ').title()
        confidence = detection['confidence'] * 100
        category = detection['category']
        
        # Clean the response text
        clean_response = self.clean_llm_response(response_text)
        
        # Simple content formatting without title and category info
        enhanced_content = f"""{clean_response}"""
        
        return {
            'detection': detection,
            'title': '',  # Remove title
            'content': enhanced_content,
            'disassembly': [],  # Remove disassembly steps
            'safety': [],  # Remove safety warnings
            'category': category,
            'recycling_instructions': clean_response,
            'source': 'AI Assistant'
        }
    
    def clean_llm_response(self, response_text: str) -> str:
        """Clean and simplify LLM response text"""
        if not response_text:
            return "Follow local recycling guidelines for proper disposal."
        
        # Remove repetitive and verbose phrases
        repetitive_phrases = [
            'please note that this answer assumes',
            'if you have different questions',
            'please let me know',
            'thank you for your question',
            'have a great day',
            'thank you for using my service',
            'feel free to ask',
            'if you have any further questions',
            'please note that',
            'this answer assumes',
            'i can assist you',
            'do my best to provide',
            'helpful answers',
            'you have any further questions',
            'thank you for using my service',
            'have a great day'
        ]
        
        # Clean the text
        clean_text = response_text.lower()
        for phrase in repetitive_phrases:
            clean_text = clean_text.replace(phrase, '')
        
        # Split into sentences and keep only the first 3 meaningful sentences
        sentences = clean_text.split('. ')
        meaningful_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (sentence and 
                len(sentence) > 10 and 
                not any(skip in sentence for skip in ['thank you', 'please note', 'feel free'])):
                meaningful_sentences.append(sentence.capitalize())
                if len(meaningful_sentences) >= 3:
                    break
        
        if meaningful_sentences:
            result = '. '.join(meaningful_sentences)
            if not result.endswith('.'):
                result += '.'
            return result
        else:
            return "Follow local recycling guidelines for proper disposal."
    
    def enhance_llm_content(self, detection: Dict, response_text: str, code_snippets: List[str]) -> str:
        """Enhance LLM content with simple, clean formatting"""
        class_name = detection['class'].replace('_', ' ').title()
        confidence = detection['confidence'] * 100
        category = detection['category']
        
        # Clean the response text first
        clean_response = self.clean_llm_response(response_text)
        
        # Create simple, clean content without verbose footers
        enhanced_content = f"""**{class_name} Recycling Guide**

{clean_response}

**Category**: {category}
**Detection Confidence**: {confidence:.1f}%"""
        
        # Add code section if available (but keep it simple)
        if code_snippets:
            enhanced_content += f"""

**Additional Procedures**:
{chr(10).join(code_snippets[:2])}"""
        
        return enhanced_content
    
    def get_individual_llm_guidance(self, detections: List[Dict]) -> List[Dict]:
        """Fallback method to process items individually"""
        guidance_list = []
        
        for detection in detections:
            try:
                class_name = detection['class']
                category = detection['category']
                confidence = detection['confidence'] * 100
                
                # Individual query - very simple
                query = f"How to recycle {class_name}?"
                
                try:
                    response = self.llm_agent.chat(query)
                    response_text = str(response.response) if hasattr(response, 'response') else str(response)
                    
                    # Check if response is adequate
                    if len(response_text) < 30 or any(phrase in response_text.lower() for phrase in ['unable to assist', 'no tool available']):
                        response_text = self.get_direct_llm_response(f"recycle {class_name}")
                
                except Exception as e:
                    logger.error(f"Agent query failed: {e}")
                    response_text = self.get_direct_llm_response(f"recycle {class_name}")
                
                guidance = self.parse_individual_llm_response(detection, response_text)
                guidance['source'] = 'LLM Agent Individual Processing'
                guidance_list.append(guidance)
                
            except Exception as e:
                logger.error(f"Individual LLM processing error for {detection['class']}: {e}")
                guidance = self.get_enhanced_basic_guidance(detection)
                guidance_list.append(guidance)
        
        return guidance_list
    
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
        
        # Enhanced content generation
        class_title = class_name.replace('_', ' ').title()
        
        # Generate appropriate recycling instructions based on category
        if 'plastic' in category.lower():
            basic_instructions = "Check recycling number on bottom. Rinse clean before recycling."
        elif 'electronic' in category.lower() or 'e-waste' in category.lower():
            basic_instructions = "Take to certified e-waste facility. May contain valuable materials for recovery."
        elif 'glass' in category.lower():
            basic_instructions = "Remove lids/caps. Rinse clean. Check local glass recycling guidelines."
        elif 'paper' in category.lower():
            basic_instructions = "Keep dry. Remove any plastic components or tape."
        elif 'metal' in category.lower():
            basic_instructions = "Rinse clean. Remove labels if possible. Check for recycling codes."
        else:
            basic_instructions = "Check local waste management guidelines for proper disposal."
        
        content = f"""🎯 **Detection**: {class_title} ({confidence:.1f}% confidence)
♻️ **Category**: {category}

📋 **Instructions**: {basic_instructions}

🌍 **Impact**: Proper recycling helps reduce waste and conserve resources."""
        
        return {
            'detection': detection,
            'title': f'{class_title} - Recycling Guide',
            'content': content,
            'disassembly': [
                f'Clean the {class_title.lower()} thoroughly',
                'Remove any non-recyclable parts',
                'Place in appropriate recycling container'
            ],
            'safety': [
                'Handle with care to avoid injury',
                'Follow local safety guidelines'
            ],
            'category': category,
            'recycling_instructions': basic_instructions,
            'source': 'Basic Classification'
        }
    
    
    def setup_websocket_server(self):
        """Setup WebSocket server for real-time communication"""
        try:
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
                try:
                    # Create new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    start_server = websockets.serve(handle_websocket, "localhost", 8765)
                    logger.info("WebSocket server starting on ws://localhost:8765")
                    
                    loop.run_until_complete(start_server)
                    loop.run_forever()
                except Exception as e:
                    logger.warning(f"WebSocket server failed to start: {e}")
            
            # Start WebSocket server in separate thread
            ws_thread = Thread(target=start_websocket_server, daemon=True)
            ws_thread.start()
            
        except Exception as e:
            logger.warning(f"WebSocket setup failed: {e}. Continuing without WebSocket support.")
    
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
