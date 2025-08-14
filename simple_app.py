import argparse
import logging
import sys
from pathlib import Path
from collections.abc import Iterator

import gradio as gr
import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.streams as streams
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from llama_index.llms.openvino import OpenVINOLLM
from llama_index.core.chat_engine.types import ChatMode

# Initialize logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Filter unnecessary warnings for demonstration
import warnings
warnings.filterwarnings("ignore")

ov_config = {
    hints.performance_mode(): hints.PerformanceMode.LATENCY,
    streams.num(): "1",
    props.cache_dir(): ""
}

def setup_models(
    llm_model_path: Path,
    embedding_model_path: Path,
    device: str) -> tuple[OpenVINOLLM, OpenVINOEmbedding]:
    """
    Sets up LLM and embedding models using OpenVINO.
    
    Args:
        llm_model_path: Path to the LLM model
        embedding_model_path: Path to the embedding model
        device: Target device for inference ("CPU", "GPU", "NPU", etc.)
        
    Returns:
        Tuple of (llm, embedding) models
    """
    
    # Check if model paths exist
    if not llm_model_path.exists():
        log.error(f"LLM model not found at {llm_model_path}.")
        sys.exit(1)

    if not embedding_model_path.exists():
        log.error(f"Embedding model not found at {embedding_model_path}.")
        sys.exit(1)

    log.info(f"Loading models on {device}...")
    
    # Load LLM model
    llm = OpenVINOLLM(
        model_id_or_path=str(llm_model_path),
        context_window=4096,  # Reduced context window to save memory
        max_new_tokens=256,   # Reduced max tokens to save memory
        model_kwargs={"ov_config": ov_config},
        generate_kwargs={"do_sample": False, "temperature": 0.1},        
        device_map=device,
    )

    # Load embedding model
    embedding = OpenVINOEmbedding(model_id_or_path=str(embedding_model_path), device=device)

    return llm, embedding


def load_documents(data_folder_path: Path) -> VectorStoreIndex:
    """
    Loads all PDF documents from the specified data folder for RAG functionality
    """
    
    if not data_folder_path.exists():
        log.error(f"Data folder not found at {data_folder_path}")
        sys.exit(1)
    
    # Find all PDF files in the data folder
    pdf_files = list(data_folder_path.glob("*.pdf"))
    
    if not pdf_files:
        log.warning(f"No PDF files found in {data_folder_path}")
        return VectorStoreIndex.from_documents([])
    
    log.info(f"Found {len(pdf_files)} PDF files for RAG:")
    for pdf_file in pdf_files:
        log.info(f"  - {pdf_file.name}")
    
    # Load all PDF documents
    reader = SimpleDirectoryReader(input_dir=str(data_folder_path), required_exts=[".pdf"])
    documents = reader.load_data()
    
    log.info(f"Loaded {len(documents)} document chunks from {len(pdf_files)} PDF files")
    
    # Create vector index from all documents
    index = VectorStoreIndex.from_documents(documents)
    return index


def run_app(llm, index, public_interface: bool = False) -> None:
    """
    Launches the simple RAG application.
    """
    
    # Create chat engine from index
    chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT, 
        llm=llm,
        streaming=True,
        system_prompt="""
        You are Eco-Sort AI, an expert in waste classification, recycling, and environmental sustainability.
        
        Focus on providing accurate, safety-conscious waste management guidance based on the knowledge base.
        Always prioritize environmental responsibility and human safety in your recommendations.
        Use waste/recycling emojis (♻️ 🌱 🗂️ 🔋 📱 🌍) to make interactions engaging.
        Encourage proper waste sorting and responsible disposal practices.
        When safety is involved (e-waste, hazardous materials), always emphasize precautions.
        
        Answer questions using the provided context about waste management, recycling procedures, 
        and environmental guidelines. If you don't know something from the context, say so.
        """
    )

    def _handle_user_message(user_message, history):
        return "", [*history, (user_message, None)]

    def _generate_response(chat_history: list) -> Iterator[list]:
        """Generate a streaming response from the chat engine."""
        if not chat_history:
            return
            
        user_message = chat_history[-1][0]
        
        # Get streaming response from chat engine
        try:
            response = chat_engine.stream_chat(user_message)
            chat_history[-1][1] = ""
            
            for token in response.response_gen:
                if token:
                    chat_history[-1][1] += token
                yield chat_history
                    
        except Exception as e:
            log.error(f"Error generating response: {e}")
            chat_history[-1][1] = "Sorry, I encountered an error processing your request. Please try again."
            yield chat_history

    def _reset_chat() -> tuple[str, list]:
        """Resets the chat interface."""
        return "", []

    # Set up Gradio interface
    theme = gr.themes.Default(
        primary_hue="green",
        font=[gr.themes.GoogleFont("Montserrat"), "ui-sans-serif", "sans-serif"],
    )

    with gr.Blocks(theme=theme) as demo:
        header = gr.HTML(
            "<div style='text-align: center; padding: 20px;'>"
            "<h1>🤖♻️ Eco-Sort AI: Waste Classification & Recycling Guidance 🌱</h1>"
            "<p>Ask questions about waste disposal, recycling, and environmental sustainability</p>"
            "</div>"
        )

        chat_window = gr.Chatbot(
            label="Eco-Sort AI Assistant",
            avatar_images=(None, "🤖"),
            height=500,
        )            

        with gr.Row():
            message = gr.Textbox(
                label="Ask Eco-Sort AI", 
                scale=4, 
                placeholder="Ask about waste disposal, recycling, or environmental guidelines..."
            )
            with gr.Column(scale=1):
                submit_btn = gr.Button("Submit", variant="primary")
                clear = gr.ClearButton()
                      
        sample_questions = [
            "How should I dispose of my old smartphone?",
            "What type of plastic is recyclable?",
            "Can you help me understand e-waste disposal?", 
            "What safety precautions should I take with batteries?",
            "Where can I recycle electronics?",
            "What are the best materials to recycle?",
            "How do I safely disassemble electronics?",
            "Tell me about sustainable waste management"              
        ]
        gr.Examples(
            examples=sample_questions,
            inputs=message, 
            label="Example Questions"
        )                     
        
        # Set up event handlers
        message.submit(
            _handle_user_message,
            inputs=[message, chat_window],
            outputs=[message, chat_window],
            queue=False                
        ).then(
            _generate_response,
            inputs=[chat_window],
            outputs=[chat_window],
        )

        submit_btn.click(
            _handle_user_message,
            inputs=[message, chat_window],
            outputs=[message, chat_window],
            queue=False,
        ).then(
            _generate_response,
            inputs=[chat_window],
            outputs=[chat_window],
        )
        
        clear.click(_reset_chat, None, [message, chat_window])

        gr.Markdown("---")
        gr.Markdown("**Eco-Sort AI** - Simple RAG-powered waste management assistant")

    print("Demo is ready!", flush=True)
    demo.queue().launch(share=public_interface)


def run(chat_model: Path, embedding_model: Path, data_folder: Path, device: str, public_interface: bool = False):
    """
    Initializes and runs the simple RAG solution
    """
    # Load models
    llm, embedding = setup_models(chat_model, embedding_model, device)

    Settings.embed_model = embedding
    Settings.llm = llm

    # Load all PDF documents from the data folder
    index = load_documents(data_folder)
    log.info(f"Vector index created from documents in {data_folder}")
 
    # Run the application
    run_app(llm, index, public_interface)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat_model", type=str, default="model/qwen2-1.5B-INT4", help="Path to the chat model directory")
    parser.add_argument("--embedding_model", type=str, default="model/bge-small-FP32", help="Path to the embedding model directory")
    parser.add_argument("--data_folder", type=str, default="data", help="Path to the folder containing PDF files for RAG functionality")    
    parser.add_argument("--device", type=str, default="CPU", help="Device for inferencing (CPU,GPU)")
    parser.add_argument("--public", default=False, action="store_true", help="Whether interface should be available publicly")

    args = parser.parse_args()
    run(Path(args.chat_model), Path(args.embedding_model), Path(args.data_folder), args.device, args.public)
