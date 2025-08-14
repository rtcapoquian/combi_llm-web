
# Proxy fix for corporate networks - patch httpx before any network requests
import httpx
import os

# Disable proxy for httpx requests
original_httpx_client = httpx.Client
original_httpx_async_client = httpx.AsyncClient

def patched_client(*args, **kwargs):
    # Remove proxy-related kwargs that might not be supported
    kwargs.pop('proxies', None)
    try:
        return original_httpx_client(*args, **kwargs)
    except TypeError:
        # Fallback without any kwargs that might cause issues
        return original_httpx_client()

def patched_async_client(*args, **kwargs):
    # Remove proxy-related kwargs that might not be supported
    kwargs.pop('proxies', None)
    try:
        return original_httpx_async_client(*args, **kwargs)
    except TypeError:
        # Fallback without any kwargs that might cause issues
        return original_httpx_async_client()

# Monkey patch httpx to disable proxy
httpx.Client = patched_client
httpx.AsyncClient = patched_async_client

# Set environment variables to bypass proxy
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0,api.gradio.app'
os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0,api.gradio.app'
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

import argparse
import io
import logging
import sys
import time
import warnings
from io import StringIO
from pathlib import Path
from typing import Tuple, List
from collections.abc import Iterator

import gradio as gr
import nest_asyncio
import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.streams as streams
from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from llama_index.llms.openvino import OpenVINOLLM
from llama_index.core.agent import ReActChatFormatter
from llama_index.core.llms import MessageRole
from llama_index.core.callbacks import CallbackManager
from llama_index.core.chat_engine.types import ChatMode
# Agent tools
from tools import WasteClassifier, RecyclingCart
from system_prompt import react_system_header_str

# Initialize logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

#Filter unnecessary warnings for demonstration
warnings.filterwarnings("ignore")

ov_config = {
    hints.performance_mode(): hints.PerformanceMode.LATENCY,
    streams.num(): "1",
    props.cache_dir(): ""
}

def setup_models(
    llm_model_path: Path,
    embedding_model_path: Path,
    device: str) -> Tuple[OpenVINOLLM, OpenVINOEmbedding]:
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
        log.error(f"LLM model not found at {llm_model_path}. Please run convert_and_optimize_llm.py to download the model first.")
        sys.exit(1)

    if not embedding_model_path.exists():
        log.error(f"Embedding model not found at {embedding_model_path}. Please run convert_and_optimize_llm.py to download the model first.")
        sys.exit(1)

    # NPU hybrid approach: embedding on NPU, LLM on CPU due to I64 input limitations
    if device.upper() == "NPU":
        log.info("Using NPU hybrid mode: Embedding model on NPU, LLM on CPU")
        
        # LLM on CPU (NPU doesn't support I64 inputs used by language models)
        log.info("Loading LLM on CPU (NPU doesn't support I64 inputs for language models)...")
        llm = OpenVINOLLM(
            model_id_or_path=str(llm_model_path),
            context_window=8192,
            max_new_tokens=500,
            model_kwargs={"ov_config": ov_config},
            generate_kwargs={"do_sample": False, "temperature": 0.1, "top_p": 0.8},        
            device_map="CPU",
        )
        
        # Try embedding model on NPU (this should work as it was optimized for NPU)
        try:
            log.info("Loading embedding model on NPU...")
            embedding = OpenVINOEmbedding(model_id_or_path=str(embedding_model_path), device="NPU")
            log.info("✅ Embedding model successfully loaded on NPU!")
        except Exception as e:
            log.warning(f"NPU loading failed for embedding model: {e}")
            log.info("Falling back to CPU for embedding model...")
            embedding = OpenVINOEmbedding(model_id_or_path=str(embedding_model_path), device="CPU")
            
    else:
        # Standard device loading
        log.info(f"Loading models on {device}...")
        llm = OpenVINOLLM(
            model_id_or_path=str(llm_model_path),
            context_window=8192,
            max_new_tokens=500,
            model_kwargs={"ov_config": ov_config},
            generate_kwargs={"do_sample": False, "temperature": 0.1, "top_p": 0.8},        
            device_map=device,
        )

        embedding = OpenVINOEmbedding(model_id_or_path=str(embedding_model_path), device=device)

    return llm, embedding


def setup_tools()-> Tuple[FunctionTool, FunctionTool, FunctionTool, FunctionTool, FunctionTool, FunctionTool]:

    """
    Sets up and returns a collection of tools for waste classification, recycling guidance, and item tracking.
    
    Returns:
        Tuple containing tools for waste classification, disassembly guidance, recycling list management,
        viewing recycling items, clearing list, and finding disposal locations
    """

    waste_classifier_tool = FunctionTool.from_defaults(
        fn=WasteClassifier.classify_waste_type,
        name="classify_waste",
        description="ALWAYS use this tool when users ask about classifying waste items or need recycling guidance. Required inputs: item_description (str), optional: material_type (str)"
    )

    disassembly_guidance_tool = FunctionTool.from_defaults(
        fn=WasteClassifier.get_disassembly_guidance,
        name="get_disassembly_guidance",
        description="Use this tool when users need instructions for safely disassembling items before recycling. Required input: item_type (str), optional: safety_level (str - 'basic', 'detailed', 'professional')"
    )

    add_to_recycling_tool = FunctionTool.from_defaults(
        fn=RecyclingCart.add_to_recycling_list,
        name="add_to_recycling_list",
        description="""
        Use this tool WHENEVER a user wants to add items to their recycling tracking list.
        
        PARAMETERS:
        - item_name (string): Name/description of the waste item (e.g., "Old smartphone", "Plastic bottles")
        - category (string): Recycling category (e.g., "E-Waste", "Recyclable Plastic", "Metal Recyclable")
        - quantity (int): Number of items, default is 1
        - notes (string): Optional additional notes or special handling instructions
        
        RETURNS:
        - A confirmation message and updated recycling list
        
        EXAMPLES:
        To add e-waste: add_to_recycling_list(item_name="iPhone 12", category="E-Waste", quantity=1, notes="Screen cracked, battery needs special handling")
        """
    )
    
    view_recycling_list_tool = FunctionTool.from_defaults(
        fn=RecyclingCart.get_recycling_items,
        name="view_recycling_list",
        description="""
        Use this tool when a user wants to see what's in their recycling tracking list.
        No parameters are required.
        
        RETURNS:
        - A list of all items currently in the recycling list with their details
        
        EXAMPLES:
        To view the current recycling list: view_recycling_list()
        """
    )
    
    clear_recycling_list_tool = FunctionTool.from_defaults(
        fn=RecyclingCart.clear_recycling_list,
        name="clear_recycling_list",
        description="""
        Use this tool when a user asks to empty or clear their recycling tracking list.
        No parameters are required.
        
        RETURNS:
        - A confirmation message that the recycling list has been cleared
        
        EXAMPLES:
        To clear the recycling list: clear_recycling_list()
        """
    )

    disposal_locations_tool = FunctionTool.from_defaults(
        fn=RecyclingCart.get_disposal_locations,
        name="find_disposal_locations",
        description="""
        Use this tool when users need to find where to dispose of or recycle specific types of waste.
        
        PARAMETERS:
        - category (string): Waste category (e.g., "E-Waste", "Hazardous", "General Recyclables")
        - location (string): Optional geographic location for local recommendations
        
        RETURNS:
        - List of suggested disposal/recycling locations and contact information
        
        EXAMPLES:
        To find e-waste disposal: find_disposal_locations(category="E-Waste", location="urban area")
        """
    )
    
    return waste_classifier_tool, disassembly_guidance_tool, add_to_recycling_tool, view_recycling_list_tool, clear_recycling_list_tool, disposal_locations_tool


def load_documents(data_folder_path: Path) -> VectorStoreIndex:
    """
    Loads all PDF documents from the specified data folder for RAG functionality
    
    Args:
        data_folder_path: Path to the folder containing PDF documents
        
    Returns:
        VectorStoreIndex for all loaded documents
    """
    
    # Ensure the data folder exists
    if not data_folder_path.exists():
        log.error(f"Data folder not found at {data_folder_path}")
        sys.exit(1)
    
    # Find all PDF files in the data folder
    pdf_files = list(data_folder_path.glob("*.pdf"))
    
    if not pdf_files:
        log.warning(f"No PDF files found in {data_folder_path}")
        # Create a default document if no PDFs are found
        default_content = "No documents available for RAG functionality."
        index = VectorStoreIndex.from_documents([])
        return index
    
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

def custom_handle_reasoning_failure(callback_manager: CallbackManager, exception: Exception):
    """
    Provides custom error handling for agent reasoning failures.
    
    Args:
        callback_manager: The callback manager instance for event handling
        exception: The exception that was raised during reasoning
    """
    return "Hmm...I didn't quite that. Could you please rephrase your question to be simpler?"


def run_app(agent: ReActAgent, public_interface: bool = False) -> None:
    """
    Launches the application with the specified agent and interface settings.
    
    Args:
        agent: The ReActAgent instance configured with tools
        public_interface: Whether to launch with a public-facing Gradio interface
    """
    class Capturing(list):
        """A context manager that captures stdout output into a list."""
        def __enter__(self):
            """
            Redirects stdout to a StringIO buffer and returns self.
            Called when entering the 'with' block.
            """
            self._stdout = sys.stdout
            sys.stdout = self._stringio = StringIO()
            return self
        def __exit__(self, *args):
            """
            Stores captured output in this list and restores stdout.
            Called when exiting the 'with' block.
            """
            self.extend(self._stringio.getvalue().splitlines())
            del self._stringio
            sys.stdout = self._stdout        

    def _handle_user_message(user_message, history):
        return "", [*history, (user_message, None)]

    def update_recycling_display()-> str:
        """
        Generates an HTML representation of the recycling tracking list contents.
        
        Retrieves current recycling items and creates a formatted HTML table
        showing item details, categories, quantities, and notes.
        If the list is empty, returns a message indicating this.
        
        Returns:
            str: Markdown-formatted HTML table of recycling list contents
                or message indicating empty list
        """
        recycling_items = RecyclingCart.get_recycling_items()
        if not recycling_items:
            return "### ♻️ Your Recycling List is Empty"
            
        table = "### ♻️ Your Recycling Tracking List\n\n"
        table += "<table>\n"
        table += "  <thead>\n"
        table += "    <tr>\n"
        table += "      <th>Item</th>\n"
        table += "      <th>Category</th>\n"
        table += "      <th>Qty</th>\n"
        table += "      <th>Notes</th>\n"
        table += "    </tr>\n"
        table += "  </thead>\n"
        table += "  <tbody>\n"
            
        for item in recycling_items:
            table += "    <tr>\n"
            table += f"      <td>{item['item_name']}</td>\n"
            table += f"      <td><span style='color: #2E8B57; font-weight: bold;'>{item['category']}</span></td>\n"
            table += f"      <td>{item['quantity']}</td>\n"
            table += f"      <td>{item.get('notes', 'None')}</td>\n"
            table += "    </tr>\n"
            
        table += "  </tbody>\n"
        table += "</table>\n"
        
        total_items = sum(item["quantity"] for item in recycling_items)
        categories = set(item["category"] for item in recycling_items)
        table += f"\n**Total Items: {total_items} | Categories: {len(categories)}**"
        return table

    def _generate_response(chat_history: list, log_history: list | None = None) -> Iterator[Tuple[str, str, str]]:
        """
        Generate a streaming response from the agent with formatted thought process logs.
        
        This function:
        1. Captures the agent's thought process
        2. Formats the thought process into readable logs
        3. Streams the agent's response token by token
        4. Tracks performance metrics for thought process and response generation
        5. Updates the shopping cart display
        
        Args:
            chat_history: List of conversation messages
            log_history: List to store logs, will be initialized if None
            
        Yields:
            tuple: (chat_history, formatted_log_history, cart_content)
                - chat_history: Updated with agent's response
                - formatted_log_history: String of joined logs
                - cart_content: HTML representation of the shopping cart
        """
        log.info(f"log_history {log_history}")           
        
        if not isinstance(log_history, list):
            log_history = []

        # Capture time for thought process
        start_thought_time = time.time()

        # Capture the thought process output
        with Capturing() as output:
            try:
                response = agent.stream_chat(chat_history[-1][0])
            except ValueError:
                response = agent.stream_chat(chat_history[-1][0])
        formatted_output = []
        for line in output:
            if "Thought:" in line:
                formatted_output.append("\n🤔 **Thought:**\n" + line.split("Thought:", 1)[1])
            elif "Action:" in line:
                formatted_output.append("\n🔧 **Action:**\n" + line.split("Action:", 1)[1])
            elif "Action Input:" in line:
                formatted_output.append("\n📥 **Input:**\n" + line.split("Action Input:", 1)[1])
            elif "Observation:" in line:
                formatted_output.append("\n📋 **Result:**\n" + line.split("Observation:", 1)[1])
            else:
                formatted_output.append(line)
        end_thought_time = time.time()
        thought_process_time = end_thought_time - start_thought_time

        # After response is complete, show the captured logs in the log area
        log_entries = "\n".join(formatted_output)
        log_history.append("### 🤔 Agent's Thought Process")
        thought_process_log = f"Thought Process Time: {thought_process_time:.2f} seconds"
        log_history.append(f"{log_entries}\n{thought_process_log}")
        cart_content = update_recycling_display() # update recycling list
        yield chat_history, "\n".join(log_history), cart_content  # Yield after the thought process time is captured

        # Now capture response generation time
        start_response_time = time.time()

        # Gradually yield the response from the agent to the chat
        # Quick fix for agent occasionally repeating the first word of its response
        last_token = "Dummy Token"
        i = 0
        chat_history[-1][1] = ""
        for token in response.response_gen:
            if i == 0:
                last_token = token
            # Safe token processing to avoid index errors
            if i == 1 and token and last_token:
                try:
                    token_words = token.split()
                    last_token_words = last_token.split()
                    if token_words and last_token_words and token_words[0] == last_token_words[0]:
                        if len(token_words) > 1:
                            chat_history[-1][1] += " ".join(token_words[1:]) + " "
                        continue
                except (IndexError, AttributeError):
                    pass  # If there's any issue, just add the token normally
            
            chat_history[-1][1] += token if token else ""
            yield chat_history, "\n".join(log_history), cart_content  # Ensure log_history is a string
            if i <= 2: 
                i += 1

        end_response_time = time.time()
        response_time = end_response_time - start_response_time

        # Log tokens per second along with the device information
        tokens = len(chat_history[-1][1].split(" ")) * 4 / 3  # Convert words to approx token count
        response_log = f"Response Time: {response_time:.2f} seconds ({tokens / response_time:.2f} tokens/s)"

        log.info(response_log)

        # Append the response time to log history
        log_history.append(response_log)
        yield chat_history, "\n".join(log_history), cart_content  # Join logs into a string for display

    def _reset_chat()-> tuple[str, list, str, str]:
        """
        Resets the chat interface and agent state to initial conditions.
        
        This function:
        1. Resets the agent's internal state
        2. Clears all items from the recycling tracking list
        3. Returns values needed to reset the UI components
        
        Returns:
            tuple: Values to reset UI components
                - Empty string: Clears the message input
                - Empty list: Resets chat history
                - Default log heading: Sets initial log area text
                - Empty recycling display: Shows empty recycling list
        """
        agent.reset()
        RecyclingCart._recycling_items = []
        return "", [], "🤔 Agent's Thought Process", update_recycling_display()

    def run()-> None:
        """
        Sets up and launches the Gradio web interface for the Smart Retail Assistant.
        
        This function:
        1. Loads custom CSS styling if available
        2. Configures the Gradio theme and UI components
        3. Sets up the chat interface with agent interaction
        4. Configures event handlers for user inputs
        5. Adds example prompts for users
        6. Launches the web interface
        
        The interface includes:
        - Chat window for user-agent conversation
        - Log window to display agent's thought process
        - Shopping cart display
        - Text input for user messages
        - Submit and Clear buttons
        - Sample questions for easy access
        """
        custom_css = ""
        try:
            with open("css/gradio.css", "r") as css_file:
                custom_css = css_file.read()            
        except Exception as e:            
            log.warning(f"Could not load CSS file: {e}")

        theme = gr.themes.Default(
            primary_hue="blue",
            font=[gr.themes.GoogleFont("Montserrat"), "ui-sans-serif", "sans-serif"],
        )

        with gr.Blocks(theme=theme, css=custom_css) as demo:

            header = gr.HTML(
                        "<div class='intel-header-wrapper'>"
                        "  <div class='intel-header'>"
                        "    <img src='https://www.intel.com/content/dam/logos/intel-header-logo.svg' class='intel-logo'></img>"
                        "    <div class='intel-title'>Eco-Sort AI 🤖♻️: Waste Classification & Recycling Guidance 🌱</div>"
                        "  </div>"
                        "</div>"
            )

            with gr.Row():
                chat_window = gr.Chatbot(
                    label="Eco-Sort AI Assistant - Waste Classification & Recycling Guide",
                    avatar_images=(None, "https://docs.openvino.ai/2024/_static/favicon.ico"),
                    height=400,  # Adjust height as per your preference
                    scale=2  # Set a higher scale value for Chatbot to make it wider
                    #autoscroll=True,  # Enable auto-scrolling for better UX
                )            
                log_window = gr.Markdown(                                                                    
                        show_label=True,                        
                        value="### 🤔 Agent's Thought Process",
                        height=400,                        
                        elem_id="agent-steps"
                )
                recycling_display = gr.Markdown(
                    value=update_recycling_display(),
                    elem_id="recycling-list",
                    height=400
                )

            with gr.Row():
                message = gr.Textbox(label="Ask Eco-Sort AI �♻️", scale=4, placeholder="Describe your waste item or ask about recycling...")

                with gr.Column(scale=1):
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear = gr.ClearButton()
                          
            sample_questions = [
                "How should I dispose of my old smartphone?",
                "What type of plastic is this water bottle?",
                "Can you help me disassemble a laptop for recycling?",
                "Add old iPhone to my recycling list",
                "What are the safety precautions for e-waste?",
                "Where can I recycle batteries in my area?",
                "Show me what's in my recycling list",
                "Clear my recycling tracking list",
                "Is this item hazardous waste?",
                "I have several electronics to recycle, help me sort them"              
            ]
            gr.Examples(
                examples=sample_questions,
                inputs=message, 
                label="Examples"
            )                     
            
            # Ensure that individual components are passed
            message.submit(
                _handle_user_message,
                inputs=[message, chat_window],
                outputs=[message, chat_window],
                queue=False                
            ).then(
                _generate_response,
                inputs=[chat_window, log_window],
                outputs=[chat_window, log_window, recycling_display],
            )

            submit_btn.click(
                _handle_user_message,
                inputs=[message, chat_window],
                outputs=[message, chat_window],
                queue=False,
            ).then(
                _generate_response,
                inputs=[chat_window, log_window],
                outputs=[chat_window, log_window, recycling_display],
            )
            clear.click(_reset_chat, None, [message, chat_window, log_window, recycling_display])

            gr.Markdown("------------------------------")

        print("Demo is ready!", flush=True)  # Required for the CI to detect readiness
        
        # Additional proxy bypass for Gradio analytics
        try:
            import gradio.analytics
            gradio.analytics._do_analytics = lambda *args, **kwargs: None
        except:
            pass
            
        # Launch with proxy bypass
        demo.queue().launch(
            share=public_interface,
            quiet=True
        )

    run()


def run(chat_model: Path, embedding_model: Path, data_folder: Path, device: str, public_interface: bool = False):
    """
    Initializes and runs the agentic rag solution
    
    Args:
        chat_model: Path to the LLM chat model
        embedding_model: Path to the embedding model
        data_folder: Path to the folder containing PDF files for RAG functionality
        device: Target device for model inference ("CPU", "GPU", "GPU.1", "NPU")
        public_interface: Whether to expose a public-facing interface
    """
    # Load models and embedding based on parsed arguments
    llm, embedding = setup_models(chat_model, embedding_model, device)

    Settings.embed_model = embedding
    Settings.llm = llm

    # Set up tools
    waste_classifier_tool, disassembly_guidance_tool, add_to_recycling_tool, view_recycling_list_tool, clear_recycling_list_tool, disposal_locations_tool = setup_tools()
    
    # Load all PDF documents from the data folder
    index = load_documents(data_folder)
    log.info(f"Vector index created from documents in {data_folder}")
 
    vector_tool = QueryEngineTool(
        index.as_query_engine(streaming=True),
        metadata=ToolMetadata(
            name="vector_search",
            description="""            
            Use this tool for ANY question about waste management, recycling procedures, disposal guidelines, or environmental information from the knowledge base.
            
            The knowledge base contains multiple PDF documents that may include:
            - Item-specific disassembly guides and instructions
            - Material-type sorting and classification guidelines  
            - Municipal and regional waste disposal regulations
            - E-waste and appliance recycling procedures
            - Safety guidelines and hazardous material handling
            - Recycling myths, misconceptions, and best practices
            - Multilingual recycling and environmental guides
            
            WHEN TO USE:
            - User asks about specific waste disposal procedures
            - User needs detailed disassembly or safety instructions
            - User has questions about local recycling regulations
            - User asks about environmental impact or sustainability
            - User needs multilingual waste management information
            
            EXAMPLES:
            - "How do I safely disassemble a CRT monitor?"
            - "What are the municipal guidelines for e-waste disposal?"
            - "Is lithium from batteries recyclable?"
            - "What safety equipment do I need for appliance disassembly?"
            - "Tell me about electronic waste recycling procedures"
            """,
        ),
    )
    
    nest_asyncio.apply()
 
    # Define agent and available tools
    agent = ReActAgent.from_tools(
        [waste_classifier_tool, disassembly_guidance_tool, add_to_recycling_tool, view_recycling_list_tool, clear_recycling_list_tool, disposal_locations_tool, vector_tool],
        llm=llm,
        max_iterations=5,  # Set a max_iterations value
        handle_reasoning_failure_fn=custom_handle_reasoning_failure,
        verbose=True,
        react_chat_formatter=ReActChatFormatter.from_defaults(),
    ) 
    react_system_prompt = PromptTemplate(react_system_header_str)
    agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})  
    agent.reset()                     
    run_app(agent, public_interface)


if __name__ == "__main__":
    # Define the argument parser at the end
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat_model", type=str, default="model/qwen2-7B-INT4", help="Path to the chat model directory")
    parser.add_argument("--embedding_model", type=str, default="model/bge-large-FP32", help="Path to the embedding model directory")
    parser.add_argument("--data_folder", type=str, default="data", help="Path to the folder containing PDF files for RAG functionality")    
    parser.add_argument("--device", type=str, default="AUTO:GPU,CPU", help="Device for inferencing (CPU,GPU,GPU.1,NPU)")
    parser.add_argument("--public", default=False, action="store_true", help="Whether interface should be available publicly")

    args = parser.parse_args()

    run(Path(args.chat_model), Path(args.embedding_model), Path(args.data_folder), args.device, args.public)
