import os
import json
import streamlit as st
import time
import logging
import uuid
from typing import (
    List, 
    Dict, 
    Any, 
    Optional
)
import lancedb
from openai import OpenAI

from galileo_api_helper import (
    get_galileo_project_id, 
    get_galileo_log_stream_id
)
from galileo import GalileoLogger

import copy

# Import Galileo Protect functionality
from galileo.handlers.langchain import GalileoCallback
from galileo.handlers.langchain.tool import ProtectTool, ProtectParser
from galileo_core.schemas.protect.ruleset import Ruleset
from galileo_core.schemas.protect.action import OverrideAction
from galileo.projects import get_project
from galileo.stages import get_protect_stage

from langchain_openai import ChatOpenAI

import logging
logging.getLogger("watchdog").setLevel(logging.WARNING)
logging.getLogger("watchdog.observers.inotify_buffer").setLevel(logging.ERROR)


# Import tools
from log_hallucination import log_hallucination
from tools.get_ticker_symbol import get_ticker_symbol
from tools.get_stock_price import get_stock_price
from tools.purchase_stocks import purchase_stocks
from tools.sell_stocks import sell_stocks
from tools.tool_definitions import (
    AMBIGUOUS_TOOL_NAMES, 
    AMBIGUOUS_TOOL_DESCRIPTIONS, 
    AMBIGUOUS_PARAMETER_DESCRIPTIONS,
    tools)

# Import LanceDB setup functions
from data.vectordb_setup import check_and_populate_lancedb, populate_lancedb_automatically, add_document_to_lancedb

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger_debug = logging.getLogger(__name__)

# Environment variable fallbacks for production
def get_secret(key, default=None):
    """Get secret from st.secrets or environment variable."""
    try:
        return st.secrets[key]
    except:
        return os.getenv(key, default)

# Set environment variables with fallbacks
os.environ["GALILEO_API_KEY"] = get_secret("galileo_api_key", os.getenv("GALILEO_API_KEY"))
os.environ["GALILEO_PROJECT"] = get_secret("galileo_project", os.getenv("GALILEO_API_KEY"))
os.environ["GALILEO_LOG_STREAM_NAME"] = get_secret("galileo_log_stream", os.getenv("GALILEO_LOG_STREAM_NAME"))
os.environ["GALILEO_CONSOLE_URL"] = get_secret("galileo_console_url", os.getenv("GALILEO_CONSOLE_URL"))
os.environ["OPENAI_API_KEY"] = get_secret("openai_api_key", os.getenv("OPENAI_API_KEY"))

system_prompt = """
You are "Galileo TradeDesk," a STOCKS-ONLY agent.

SCOPE: U.S. equities, watchlists, quotes, basic TA (SMA/EMA/RSI), paper trades, order simulations, P&L, compliance-safe explanations.
OUT-OF-SCOPE: essays, recipes, general trivia, non-equity assets, medical/legal advice. If asked, DECLINE briefly and redirect to scope.

STYLE: concise, risk-aware, no hype, ≤120 words unless asked "detail".
WHEN ASKED "what can you do?" -> return the curated capability list verbatim.
"""

stage_id = get_secret("galileo_stage_agent_id", "No agent ID found. Please set galileo_stage_agent_id.")
project = get_project(name=os.environ["GALILEO_PROJECT"])
project_name = "protect_demo"
project_id = "edd72199-5b8f-44a1-a228-6ab0712fe2c2"
model_option = "gpt-4o-mini" 
log_steam_name = os.environ["GALILEO_LOG_STREAM_NAME"]
namespace = "documents"  # Default namespace for LanceDB
top_k = 10

logger_debug.debug("\n" + "-" * 60)
logger_debug.debug("💓 Galileo TradeDesk Config 💓 ")
logger_debug.debug(f"{'✅' if os.getenv('GALILEO_API_KEY') else '❌'} : Galileo API Key")
logger_debug.debug(f"{'✅' if get_secret('openai_api_key', os.getenv('OPENAI_API_KEY')) else '❌'} : OpenAI API Key")
logger_debug.debug(f"{'✅' if os.path.exists('./data') else '❌'} : LanceDB Data Directory")
logger_debug.debug(f"{'✅' if os.path.exists('./data') else '❌'} : LanceDB Tables")
logger_debug.debug(f"{os.getenv('GALILEO_CONSOLE_URL', '❌')}")
logger_debug.debug(f"⬆️ Galileo Console URL ⬆️")
logger_debug.debug(f"{os.getenv('GALILEO_PROJECT', '❌')}: Galileo Project")
logger_debug.debug(f"{model_option} \n" + "-" * 60)

openai_client = OpenAI(api_key= os.getenv('OPENAI_API_KEY'))
# Initialize LanceDB with absolute path
import pathlib
data_path = pathlib.Path(__file__).parent / "data"
# Ensure data directory exists
data_path.mkdir(exist_ok=True)
db = lancedb.connect(str(data_path))

# Auto-populate LanceDB on startup if needed
try:
    check_and_populate_lancedb()
    logger_debug.info("LanceDB setup completed successfully")
except Exception as e:
    logger_debug.warning(f"LanceDB setup failed on startup: {str(e)}")
    logger_debug.info("LanceDB will be set up when first needed")

# Define RAG response type
class RagResponse:
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents

def get_rag_response(query: str, top_k: int) -> Optional[RagResponse]:
    """Get RAG response using LanceDB vector store."""
    try:
        logger_debug.info(f"Making RAG request - Query: {query}, Top K: {top_k}")
        
        # Get embeddings for the query
        logger_debug.info("Getting embeddings for query")
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = embedding_response.data[0].embedding
        logger_debug.debug(f"Generated embedding of length: {len(query_embedding)}")
        
        # Get or create table
        table_name = "documents"
        try:
            table = db.open_table(table_name)
            logger_debug.info(f"Successfully opened table: {table_name}")
        except Exception as e:
            logger_debug.warning(f"Table {table_name} not found, creating empty table. Error: {e}")
            # Create table with schema
            import pyarrow as pa
            schema = pa.schema([
                pa.field("id", pa.string(), nullable=False),
                pa.field("text", pa.large_string()),
                pa.field("embedding", pa.list_(pa.float32(), 1536), nullable=False),
                pa.field("metadata", pa.struct([
                    pa.field("company", pa.string(), nullable=True),
                    pa.field("ticker", pa.string(), nullable=True),
                    pa.field("sector", pa.string(), nullable=True),
                    pa.field("type", pa.string(), nullable=True),
                    pa.field("index", pa.string(), nullable=True),
                    pa.field("topic", pa.string(), nullable=True)
                ]))
            ])
            table = db.create_table(table_name, schema=schema)
            logger_debug.info(f"Created new table: {table_name}")
            return None  # No documents to return for new table
        
        # Query LanceDB
        logger_debug.info(f"Querying LanceDB table: {table_name}")
        try:
            # Vector similarity search
            results = table.search(query_embedding).metric("cosine").limit(top_k).to_pandas()
            
            if results.empty:
                logger_debug.warning("No matches found in LanceDB")
                return None
                
            logger_debug.info(f"Found {len(results)} matches in LanceDB")
            
            # Format documents
            documents = []
            for _, row in results.iterrows():
                doc = {
                    "content": row.get("text", ""),
                    "metadata": {
                        "score": row.get("_distance", 0.0),
                        "id": row.get("id", ""),
                        **row.get("metadata", {})
                    }
                }
                documents.append(doc)
            
            logger_debug.info(f"Formatted {len(documents)} documents for response")
            if documents:
                logger_debug.debug(f"First document content preview: {documents[0]['content'][:200]}")
            
            return RagResponse(documents=documents)
            
        except Exception as e:
            logger_debug.error(f"Error querying LanceDB: {str(e)}", exc_info=True)
            return None
        
    except Exception as e:
        logger_debug.error(f"Error in RAG request: {str(e)}", exc_info=True)
        return None

def create_protected_chain(model: str = "gpt-4o", temperature: float = 0.7, timeout: int = 10):
    """Create a protected chain using Galileo Protect.
    
    Args:
        model: The OpenAI model to use
        temperature: The temperature for the model
        timeout: Timeout for the protect tool
        
    Returns:
        A tuple of (protected_chain, galileo_callback)
    """
    logger_debug.info(f"Creating protected chain with model: {model}, timeout: {timeout}")
    
    # Create a callback handler
    galileo_callback = GalileoCallback()
    
    # Create LangChain LLM
    llm = ChatOpenAI(model=model, temperature=temperature)
    
    # Get project name with detailed logging
    project_name = os.getenv("GALILEO_PROJECT")
    try:
        # Check for the "agent off" stage
        # Also check for specific stage ID
        stage = get_protect_stage(project_id=project_id,
                                 stage_id=stage_id)
        if stage:
            stage_status = stage.paused # if false then agent disabled
        if stage_status==False:
            logger_debug.info("Agent is disabled - returning fallback message")
            
            class AgentOffChain:
                def __init__(self):
                    self.content = "Sorry, this agent has been disabled."
                
                def invoke(self, input_data, config=None):
                    class MockResponse:
                        def __init__(self):
                            self.content = "Sorry, this agent has been disabled."
                    return MockResponse()
            
            return AgentOffChain(), galileo_callback
        else:
            logger_debug.info("Agent enabled.")
            
    except Exception as e:
        logger_debug.warning(f"Error checking for agent-off stage: {str(e)}")
        logger_debug.info("Proceeding with protection due to stage check error")
    
    # Create a ProtectParser instance, passing the LLM as the chain to be invoked
    protect_parser = ProtectParser(chain=llm)
    protect_tool = ProtectTool(
        stage_name="Finance Chat Protection",
        project_name=project_name,
        prioritized_rulesets=[
            Ruleset(rules=[
                {
                    "metric": "prompt_injection", # context_adherence_luna, prompt_injection, pii
                    "operator": "eq", # gt, eq, contains
                    "target_value": ["new_context"], # 0.6, new_context, address
                },
                {
                    "metric": "prompt_injection", # context_adherence_luna, prompt_injection, pii
                    "operator": "eq", # gt, eq, contains
                    "target_value": ["impersonation"], # 0.6, new_context, address
                }
            ]),
        ],
        action = OverrideAction(
        choices=[
            "Please rephrase your input."
            ]
        )
        )
    
    # Define the chain with Protect
    protected_chain = protect_tool | protect_parser.parser
    
    logger_debug.info("Protected chain created successfully")
    return protected_chain, galileo_callback

# Format tools for OpenAI API
openai_tools = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": tool["description"],
            "parameters": tool["parameters"]
        }
    }
    for name, tool in tools.items()
]

def escape_dollar_signs(text: str) -> str:
    """Escape dollar signs in text to prevent LaTeX interpretation."""
    return text.replace('$', '\\$')

def format_message(role: str, content: str = None, tool_calls=None, tool_call_id=None) -> dict:
    """Format a message for the chat.
    
    Args:
        role: The role of the message (system, user, assistant, tool)
        content: The content of the message
        tool_calls: Tool calls for assistant messages
        tool_call_id: Tool call ID for tool messages
        
    Returns:
        A properly formatted message dictionary
    """
    message = {"role": role}
    
    if content is not None:
        message["content"] = content
        
    if role == "assistant" and tool_calls is not None:
        message["tool_calls"] = [{
            "id": tool_call.get("id", f"toolcall-{i}"),
            "type": tool_call.get("type", "function"),
            "function": {
                "name": tool_call.get("function", {}).get("name", ""),
                "arguments": tool_call.get("function", {}).get("arguments", "{}")
            }
        } for i, tool_call in enumerate(tool_calls)]
        
    if role == "tool" and tool_call_id is not None:
        message["tool_call_id"] = tool_call_id
        
    return message

def handle_tool_call(tool_call, tool_result, description, messages_to_use, logger, is_streamlit=True):
    """Handle a tool call and its response.
    
    Args:
        tool_call: The tool call object from OpenAI
        tool_result: The result from executing the tool
        description: Human-readable description of what the tool is doing
        messages_to_use: The message history to append to
        logger: The Galileo logger
        is_streamlit: Whether to use Streamlit-specific code
    """
    # Create tool call data
    tool_call_data = {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments
        }
    }
    
    # Add tool response to messages
    messages_to_use.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [tool_call_data]
    })
    
    messages_to_use.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": tool_result
    })
    
    if is_streamlit:
        # Display the tool response in chat
        st.session_state.messages.append(format_message(
            role="assistant", 
            content=description,
            tool_calls=[tool_call_data]
        ))
        
        st.session_state.messages.append(format_message(
            role="tool", 
            content=tool_result,
            tool_call_id=tool_call.id
        ))
        
        with st.chat_message("assistant"):
            st.markdown(escape_dollar_signs(description))
        
        with st.chat_message("tool"):
            st.markdown(escape_dollar_signs(tool_result))

async def process_chat_message(
    prompt: str,
    message_history: List[Dict[str, Any]], 
    model: str = model_option, 
    system_prompt: str = None,
    use_rag: bool = True,
    top_k: int = 3,
    galileo_logger = None,
    is_streamlit=True,
    ambiguous_tool_names: bool = False,
    use_protection: bool = True
) -> Dict[str, Any]:
    """Process a chat message independently of Streamlit UI.
    
    Args:
        prompt (str): The user's message/prompt
        message_history (List[Dict[str, Any]]): Previous message history
        model (str): The OpenAI model to use
        system_prompt (str): System prompt to use
        use_rag (bool): Whether to use RAG for context
        namespace (str): Namespace for RAG
        top_k (int): Number of top documents to retrieve for RAG
        galileo_logger: Optional Galileo logger for observability
        is_streamlit: Whether to use Streamlit-specific code
        ambiguous_tool_names: Whether to use ambiguous tool names
        use_protection: Whether to use Galileo Protect for query protection

    Returns:
        Dict containing:
            - response_message: The final response message from the model
            - updated_history: The updated message history
            - rag_documents: Any RAG documents retrieved (if RAG was used)
    """
    return process_chat_message_sync(
        prompt=prompt,
        message_history=message_history,
        model=model,
        system_prompt=system_prompt,
        use_rag=use_rag,
        top_k=top_k,
        galileo_logger=galileo_logger,
        is_streamlit=is_streamlit,
        ambiguous_tool_names=ambiguous_tool_names,
        use_protection=use_protection
    )

def process_chat_message_sync(prompt: str,
    message_history: List[Dict[str, Any]], 
    model: str = model_option, 
    system_prompt: str = None,
    use_rag: bool = True,
    top_k: int = 3,
    galileo_logger = None,
    is_streamlit=True,
    ambiguous_tool_names: bool = False,
    use_protection: bool = True) -> Dict[str, Any]:
    start_time = time.time()
    logger_debug.info(f"Processing chat message: {prompt}")
    
    # Start Galileo trace if available
    if galileo_logger and not galileo_logger.current_parent():
        logger_debug.info("Starting new Galileo trace")
        trace = galileo_logger.start_trace(
            input=prompt,
            name="Chat Workflow",
            tags=["chat"],
        )
    
    try:
        # Copy message history to avoid modifying the original
        messages_to_use = message_history.copy()
        
        # Add user message to history
        messages_to_use.append(format_message("user", prompt))
        
        rag_documents = []
        
        # Handle RAG if enabled
        if use_rag:
            logger_debug.info("RAG enabled, fetching relevant documents")
            rag_response = get_rag_response(prompt, top_k)
            
            if rag_response and rag_response.documents:
                logger_debug.info(f"RAG returned {len(rag_response.documents)} documents")
                rag_documents = rag_response.documents
                
                # Log RAG retrieval to Galileo if available
                if galileo_logger:
                    galileo_logger.add_retriever_span(
                        input=prompt,
                        output=[doc['content'] for doc in rag_response.documents],
                        name="RAG Retriever",
                        duration_ns=int((time.time() - start_time) * 1000000),
                        metadata={
                            "document_count": str(len(rag_response.documents))
                        }
                    )
                
                # Add context to system message
                context = "\n\n".join(doc['content'] for doc in rag_response.documents)
                logger_debug.debug(f"Adding RAG context to messages: {context[:200]}...")
                
                messages_to_use = [
                    {
                        "role": "system",
                        "content": f"{system_prompt}\n\nHere is the relevant context that you should use to answer the user's questions:\n\n{context}\n\nMake sure to use this context when answering questions. Keep answers brief; under 50 words."
                    },
                    *messages_to_use
                ]
            else:
                logger_debug.warning("No RAG documents found for query")
                if system_prompt:
                    messages_to_use = [
                        {"role": "system", "content": system_prompt},
                        *messages_to_use
                    ]
        elif system_prompt:
            logger_debug.info("Adding system prompt without RAG")
            messages_to_use = [
                {"role": "system", "content": system_prompt},
                *messages_to_use
            ]
        
        # Check if we need to use protection
        if use_protection:
            logger_debug.info("Using Galileo Protect for query protection")
            try:
                # Create protected chain
                protected_chain, galileo_callback = create_protected_chain(model=model)
                
                # Prepare input for protected chain
                chain_input = {"input": prompt}
                
                # Run through protected chain
                response = protected_chain.invoke(
                    chain_input,
                    config={"callbacks": [galileo_callback]}
                )

                print("Response:", response) # does this respond with/ what?
                logger_debug.info(f"Response: {response}")
                

                # Handle protected response
                logger_debug.info(f"Protection response type: {type(response)}")
                logger_debug.info(f"Protection response content: {response}")
                
                if hasattr(response, 'content') and response.content:
                    # LLM was executed - create a mock response object
                    logger_debug.info("✅ Protection allowed - LLM Response generated")
                    response_message = type('obj', (object,), {
                        'content': response.content,
                        'role': 'assistant',
                        'tool_calls': None
                    })
                elif isinstance(response, str):
                    # ProtectTool intervened
                    logger_debug.info(f"🛡️ Protection intercepted/modified query")
                    response_message = type('obj', (object,), {
                        'content': f"Sorry, something went wrong. Please reword your query and try again.",
                        'role': 'assistant',
                        'tool_calls': None
                    })
                elif hasattr(response, 'content') and not response.content:
                    # Empty response - likely blocked
                    logger_debug.info(f"🛡️ Protection blocked query (empty response)")
                    response_message = type('obj', (object,), {
                        'content': f"Sorry, something went wrong. Please reword your query and try again.",
                        'role': 'assistant',
                        'tool_calls': None
                    })
                else:
                    # Unexpected response type
                    logger_debug.warning(f"❓ Unexpected protection response type: {type(response)}")
                    response_message = type('obj', (object,), {
                        'content': "I encountered an error processing your request.",
                        'role': 'assistant',
                        'tool_calls': None
                    })
                
                # Skip tool processing for protected responses
                tool_results = []
                
                # Calculate token counts for protected response
                input_tokens = len(prompt.split()) if prompt else 0
                output_tokens = len(response_message.content.split()) if response_message.content else 0
                total_tokens = input_tokens + output_tokens
                
            except Exception as e:
                logger_debug.error(f"Error in protection chain: {str(e)}", exc_info=True)
                # Fall back to regular processing
                use_protection = False
                logger_debug.info("Falling back to regular processing due to protection error")
                logger_debug.info(f"Error details: {type(e).__name__}: {str(e)}")
        
        # Regular processing (if not using protection or protection failed)
        if not use_protection:
            # Get response from OpenAI
            logger_debug.info(f"Calling OpenAI API with model {model}")
            logger_debug.debug(f"Messages being sent to OpenAI: {json.dumps([format_message(msg['role'], msg['content']) for msg in messages_to_use], indent=2)}")
            
            # Check if we need to use ambiguous tool names
            tools_to_use = openai_tools
            if ambiguous_tool_names:
                logger_debug.info("Using ambiguous tool names")
                tools_to_use = []
                
                for tool in openai_tools:
                    function_name = tool["function"]["name"]
                    function_desc = tool["function"]["description"]
                    function_params = tool["function"]["parameters"]
                    
                    # Create a copy of the tool
                    modified_tool = {
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "description": function_desc,
                            "parameters": copy.deepcopy(function_params)
                        }
                    }
                    
                    # Modify tools with ambiguous names and descriptions
                    if function_name in AMBIGUOUS_TOOL_NAMES:
                        ambiguous_name = AMBIGUOUS_TOOL_NAMES[function_name]
                        modified_tool["function"]["name"] = ambiguous_name
                        modified_tool["function"]["description"] = AMBIGUOUS_TOOL_DESCRIPTIONS[ambiguous_name]
                        
                        # Change parameter descriptions to be ambiguous
                        for param_name in modified_tool["function"]["parameters"]["properties"]:
                            if param_name in AMBIGUOUS_PARAMETER_DESCRIPTIONS[ambiguous_name]:
                                modified_tool["function"]["parameters"]["properties"][param_name]["description"] = \
                                    AMBIGUOUS_PARAMETER_DESCRIPTIONS[ambiguous_name][param_name]
                    
                    tools_to_use.append(modified_tool)
            
            logger_debug.debug(f"Tools being sent to OpenAI: {json.dumps(tools_to_use, indent=2)}")
            
            # Use OpenAI
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages_to_use,
                tools=tools_to_use,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            logger_debug.info("Received response from OpenAI")
            
            # Calculate token counts safely
            input_tokens = len(prompt.split()) if prompt else 0
            output_tokens = len(response_message.content.split()) if response_message.content else 0
            total_tokens = input_tokens + output_tokens
            
            # Log the API call to Galileo if available
            if galileo_logger:
                logger_debug.info("Logging API call to Galileo")
                
                # Prepare tools list for logging
                tools_for_logging = []
                for name, tool in tools.items():
                    # Determine the tool name based on whether ambiguous names are enabled
                    tool_name = name
                    if ambiguous_tool_names:
                        if name in AMBIGUOUS_TOOL_NAMES:
                            tool_name = AMBIGUOUS_TOOL_NAMES[name]
                    
                    tools_for_logging.append({
                        "name": tool_name,
                        "parameters": list(tool["parameters"]["properties"].keys())
                    })
                
                galileo_logger.add_llm_span(
                    input=[format_message(msg["role"], msg["content"]) for msg in messages_to_use],
                    output={
                        "role": response_message.role,
                        "content": response_message.content,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": call.type,
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments
                                }
                            } for call in (response_message.tool_calls or [])
                        ] if response_message.tool_calls else None
                    },
                    model=model,
                    name="OpenAI API Call",
                    tools=tools_for_logging,
                    duration_ns=int((time.time() - start_time) * 1000000),
                    metadata={"temperature": "0.7", "model": model},
                    tags=["api-call"],
                    num_input_tokens=input_tokens,
                    num_output_tokens=output_tokens,
                    total_tokens=total_tokens
                )
            
            # Handle tool calls if present
            tool_results = []
            if response_message.tool_calls:
                logger_debug.info("Processing tool calls")
                continue_conversation = True
                
                while continue_conversation and response_message.tool_calls:
                    current_tool_calls = []
                    
                    # Process each tool call and its response
                    for tool_call in response_message.tool_calls:
                        tool_result = None
                        original_function_name = tool_call.function.name
                        
                        # Map ambiguous tool names back to the original function names
                        if ambiguous_tool_names:
                            # Create a reverse mapping of ambiguous tool names to original names
                            ambiguous_to_original = {v: k for k, v in AMBIGUOUS_TOOL_NAMES.items()}
                            
                            if original_function_name in ambiguous_to_original:
                                original_function_name = ambiguous_to_original[original_function_name]
                                logger_debug.info(f"Mapping ambiguous tool name '{tool_call.function.name}' to '{original_function_name}'")
                        
                        if original_function_name == "getTickerSymbol":
                            company = json.loads(tool_call.function.arguments)["company"]
                            ticker = get_ticker_symbol(company, galileo_logger)
                            logger_debug.info(f"Got ticker symbol for {company}: {ticker}")
                            tool_result = ticker
                            description = f"Looking up ticker symbol for {company}..."
                            
                        elif original_function_name == "getStockPrice":
                            ticker = json.loads(tool_call.function.arguments)["ticker"]
                            result = get_stock_price(ticker, galileo_logger=galileo_logger)
                            logger_debug.info(f"Got stock price for {ticker}")
                            tool_result = result
                            description = f"Getting current price for {ticker}..."
                            
                        elif original_function_name == "purchaseStocks":
                            args = json.loads(tool_call.function.arguments)
                            result = purchase_stocks(
                                ticker=args["ticker"],
                                quantity=args["quantity"],
                                price=args["price"],
                                galileo_logger=galileo_logger
                            )
                            logger_debug.info(f"Processed stock purchase for {args['ticker']}")
                            tool_result = result
                            description = f"Processing purchase of {args['quantity']} shares of {args['ticker']}..."
                        elif original_function_name == "sellStocks":
                            args = json.loads(tool_call.function.arguments)
                            result = sell_stocks(
                                ticker=args["ticker"],
                                quantity=args["quantity"],
                                price=args["price"],
                                galileo_logger=galileo_logger
                            )
                            logger_debug.info(f"Processed stock sale for {args['ticker']}")
                            
                            # Handle tool call and response
                            handle_tool_call(
                                tool_call=tool_call,
                                tool_result=result,
                                description=f"Processing sale of {args['quantity']} shares of {args['ticker']}...",
                                messages_to_use=messages_to_use,
                                logger=galileo_logger,
                                is_streamlit=is_streamlit
                            )
                        if tool_result:
                            # Create tool call data for tracking
                            current_tool_calls.append({
                                "tool_call": tool_call,
                                "result": tool_result,
                                "description": description
                            })
                            
                            # Add tool call and response to messages
                            messages_to_use.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments
                                    }
                                }]
                            })
                            
                            messages_to_use.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_result
                            })
                            
                            # Track all tool results for return
                            tool_results.append({
                                "name": tool_call.function.name,
                                "arguments": json.loads(tool_call.function.arguments),
                                "result": tool_result,
                                "description": description
                            })
                    
                    # Get a new response from OpenAI with the tool results
                    logger_debug.info("Getting follow-up response with tool results")
                    follow_up_response = openai_client.chat.completions.create(
                        model=model,
                        messages=messages_to_use,
                        tools=tools_to_use,
                        tool_choice="auto"
                    )
                    
                    response_message = follow_up_response.choices[0].message
                    logger_debug.debug(f"Received follow-up response from OpenAI")
                    
                    # Calculate token counts for follow-up response
                    follow_up_input_tokens = sum(len(msg.get("content", "").split()) for msg in messages_to_use if msg.get("content"))
                    follow_up_output_tokens = len(response_message.content.split()) if response_message.content else 0
                    follow_up_total_tokens = follow_up_input_tokens + follow_up_output_tokens
                    
                    # Log the follow-up API call to Galileo if available
                    if galileo_logger:
                        logger_debug.info("Logging follow-up API call to Galileo")
                        galileo_logger.add_llm_span(
                            input=[format_message(msg["role"], msg["content"]) for msg in messages_to_use],
                            output={
                                "role": response_message.role,
                                "content": response_message.content,
                                "tool_calls": [
                                    {
                                        "id": call.id,
                                        "type": call.type,
                                        "function": {
                                            "name": call.function.name,
                                            "arguments": call.function.arguments
                                        }
                                    } for call in (response_message.tool_calls or [])
                                ] if response_message.tool_calls else None
                            },
                            model=model,
                            name="Follow-up OpenAI API Call",
                            tools=tools_for_logging,
                            duration_ns=int((time.time() - start_time) * 1000000),
                            metadata={"temperature": "0.7", "model": model},
                            tags=["api-call", "follow-up"],
                            num_input_tokens=follow_up_input_tokens,
                            num_output_tokens=follow_up_output_tokens,
                            total_tokens=follow_up_total_tokens
                        )
                    
                    # If no more tool calls, end the conversation
                    if not response_message.tool_calls:
                        continue_conversation = False
        logger_debug.info("Received response from OpenAI")
        
        # Calculate token counts safely
        input_tokens = len(prompt.split()) if prompt else 0
        output_tokens = len(response_message.content.split()) if response_message.content else 0
        total_tokens = input_tokens + output_tokens
        
        # Log the API call to Galileo if available
        if galileo_logger:
            logger_debug.info("Logging API call to Galileo")
            
            # Prepare tools list for logging
            tools_for_logging = []
            for name, tool in tools.items():
                # Determine the tool name based on whether ambiguous names are enabled
                tool_name = name
                if ambiguous_tool_names:
                    if name in AMBIGUOUS_TOOL_NAMES:
                        tool_name = AMBIGUOUS_TOOL_NAMES[name]
                
                tools_for_logging.append({
                    "name": tool_name,
                    "parameters": list(tool["parameters"]["properties"].keys())
                })
            
            galileo_logger.add_llm_span(
                input=[format_message(msg["role"], msg["content"]) for msg in messages_to_use],
                output={
                    "role": response_message.role,
                    "content": response_message.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": call.type,
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments
                            }
                        } for call in (response_message.tool_calls or [])
                    ] if response_message.tool_calls else None
                },
                model=model,
                name="OpenAI API Call",
                tools=tools_for_logging,
                duration_ns=int((time.time() - start_time) * 1000000),
                metadata={"temperature": "0.7", "model": model},
                tags=["api-call"],
                num_input_tokens=input_tokens,
                num_output_tokens=output_tokens,
                total_tokens=total_tokens
            )
        
        # Handle tool calls if present
        tool_results = []
        if response_message.tool_calls:
            logger_debug.info("Processing tool calls")
            continue_conversation = True
            
            while continue_conversation and response_message.tool_calls:
                current_tool_calls = []
                
                # Process each tool call and its response
                for tool_call in response_message.tool_calls:
                    tool_result = None
                    original_function_name = tool_call.function.name
                    
                    # Map ambiguous tool names back to the original function names
                    if ambiguous_tool_names:
                        # Create a reverse mapping of ambiguous tool names to original names
                        ambiguous_to_original = {v: k for k, v in AMBIGUOUS_TOOL_NAMES.items()}
                        
                        if original_function_name in ambiguous_to_original:
                            original_function_name = ambiguous_to_original[original_function_name]
                            logger_debug.info(f"Mapping ambiguous tool name '{tool_call.function.name}' to '{original_function_name}'")
                    
                    if original_function_name == "getTickerSymbol":
                        company = json.loads(tool_call.function.arguments)["company"]
                        ticker = get_ticker_symbol(company, galileo_logger)
                        logger_debug.info(f"Got ticker symbol for {company}: {ticker}")
                        tool_result = ticker
                        description = f"Looking up ticker symbol for {company}..."
                        
                    elif original_function_name == "getStockPrice":
                        ticker = json.loads(tool_call.function.arguments)["ticker"]
                        result = get_stock_price(ticker, galileo_logger=galileo_logger)
                        logger_debug.info(f"Got stock price for {ticker}")
                        tool_result = result
                        description = f"Getting current price for {ticker}..."
                        
                    elif original_function_name == "purchaseStocks":
                        args = json.loads(tool_call.function.arguments)
                        result = purchase_stocks(
                            ticker=args["ticker"],
                            quantity=args["quantity"],
                            price=args["price"],
                            galileo_logger=galileo_logger
                        )
                        logger_debug.info(f"Processed stock purchase for {args['ticker']}")
                        tool_result = result
                        description = f"Processing purchase of {args['quantity']} shares of {args['ticker']}..."
                    elif original_function_name == "sellStocks":
                        args = json.loads(tool_call.function.arguments)
                        result = sell_stocks(
                            ticker=args["ticker"],
                            quantity=args["quantity"],
                            price=args["price"],
                            galileo_logger=galileo_logger
                        )
                        logger_debug.info(f"Processed stock sale for {args['ticker']}")
                        
                        # Handle tool call and response
                        handle_tool_call(
                            tool_call=tool_call,
                            tool_result=result,
                            description=f"Processing sale of {args['quantity']} shares of {args['ticker']}...",
                            messages_to_use=messages_to_use,
                            logger=galileo_logger,
                            is_streamlit=is_streamlit
                        )
                    if tool_result:
                        # Create tool call data for tracking
                        current_tool_calls.append({
                            "tool_call": tool_call,
                            "result": tool_result,
                            "description": description
                        })
                        
                        # Add tool call and response to messages
                        messages_to_use.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }]
                        })
                        
                        messages_to_use.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        })
                        
                        # Track all tool results for return
                        tool_results.append({
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
                            "result": tool_result,
                            "description": description
                        })
                
                # Get a new response from OpenAI with the tool results
                logger_debug.info("Getting follow-up response with tool results")
                follow_up_response = openai_client.chat.completions.create(
                    model=model,
                    messages=messages_to_use,
                    tools=tools_to_use,
                    tool_choice="auto"
                )
                
                response_message = follow_up_response.choices[0].message
                logger_debug.debug(f"Received follow-up response from OpenAI")
                
                # Calculate token counts for follow-up response
                follow_up_input_tokens = sum(len(msg.get("content", "").split()) for msg in messages_to_use if msg.get("content"))
                follow_up_output_tokens = len(response_message.content.split()) if response_message.content else 0
                follow_up_total_tokens = follow_up_input_tokens + follow_up_output_tokens
                
                # Log the follow-up API call to Galileo if available
                if galileo_logger:
                    logger_debug.info("Logging follow-up API call to Galileo")
                    galileo_logger.add_llm_span(
                        input=[format_message(msg["role"], msg["content"]) for msg in messages_to_use],
                        output={
                            "role": response_message.role,
                            "content": response_message.content,
                            "tool_calls": [
                                {
                                    "id": call.id,
                                    "type": call.type,
                                    "function": {
                                        "name": call.function.name,
                                        "arguments": call.function.arguments
                                    }
                                } for call in (response_message.tool_calls or [])
                            ] if response_message.tool_calls else None
                        },
                        model=model,
                        name="Follow-up OpenAI API Call",
                        tools=tools_for_logging,
                        duration_ns=int((time.time() - start_time) * 1000000),
                        metadata={"temperature": "0.7", "model": model},
                        tags=["api-call", "follow-up"],
                        num_input_tokens=follow_up_input_tokens,
                        num_output_tokens=follow_up_output_tokens,
                        total_tokens=follow_up_total_tokens
                    )
                
                # If no more tool calls, end the conversation
                if not response_message.tool_calls:
                    continue_conversation = False
        
        # Add final assistant response to history
        if response_message.content:
            messages_to_use.append(format_message("assistant", response_message.content))
        
        # Conclude the Galileo trace if available
        if galileo_logger and is_streamlit:
            logger_debug.info("Concluding Galileo trace")
            galileo_logger.conclude(
                output=response_message.content,
                duration_ns=int((time.time() - start_time) * 1000000),
                status_code=200
            )
            galileo_logger.flush()
        
        # Return the results
        return {
            "response_message": response_message,
            "updated_history": messages_to_use,
            "rag_documents": rag_documents,
            "tool_results": tool_results,
            "total_tokens": total_tokens
        }
        
    except Exception as e:
        logger_debug.error(f"Error processing chat message: {str(e)}", exc_info=True)
        
        # Log error to Galileo if available
        if galileo_logger and is_streamlit:
            logger_debug.info("Logging error to Galileo")
            galileo_logger.conclude(
                output=f"Error: {str(e)}",
                duration_ns=int((time.time() - start_time) * 1000000),
                status_code=500
            )
        
        # Re-raise the exception
        raise






async def main():
    st.title("Galileo TradeDesk")
    logger_debug.debug("Starting application...")
    
    # Initialize session state variables if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_active" not in st.session_state:
        st.session_state.session_active = False
    
    if "galileo_session_id" not in st.session_state:
        st.session_state.galileo_session_id = None
    
    if "ambiguous_tool_names" not in st.session_state:
        st.session_state.ambiguous_tool_names = False
    

    
    # Initialize use_protection session state
    if "use_protection" not in st.session_state:
        st.session_state.use_protection = True
    
    # Initialize session ID for external tracking
    if "_session_id" not in st.session_state:
        st.session_state._session_id = str(uuid.uuid4())
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Galileo Configuration")

        if not "galileo_logger" in st.session_state:
            # Add Galileo configuration fields
            galileo_log_stream = st.text_input(
                "Galileo Log Stream",
                value=log_steam_name,
                help="The name of your Galileo log stream"
            )
        

        
        # Add checkbox for ambiguous tool names
        ambiguous_tool_names = st.checkbox(
            "Ambiguous Tool Names", 
            value=st.session_state.ambiguous_tool_names,
            help="Makes sell / buy functions ambiguous to induce poor tool selection"
        )
        # Update the session state with the checkbox value
        st.session_state.ambiguous_tool_names = ambiguous_tool_names    
            
        use_protection = st.checkbox(
            "Enable Galileo Protect", 
            value=st.session_state.use_protection,
            help="Enable query protection using Galileo Protect to detect and block harmful or drifting inputs"
        )
        # Update the session state with the checkbox value
        st.session_state.use_protection = use_protection
        
        # Model selection
        model_option = st.selectbox(
            "Select Model",
            options=[
                "gpt-4o-mini", 
                "gpt-4o",
                "gpt-4",
                "gpt-3.5-turbo"
            ],
            index=0,
            help="Choose your preferred OpenAI model."
        )
        use_rag = st.checkbox("Use RAG", value=True)
        
        # Agent Health Check in sidebar
        st.markdown("---")  # Add separator
        st.markdown("### 🔍 Agent Status")
        
        if st.button("Check Agent Health", type="secondary", use_container_width=True):
            # Check stage status
            try:
                agent_off_stage = get_protect_stage(stage_id=stage_id)
                if agent_off_stage==False:
                    st.warning("Agent is disabled.")
                else:
                    st.success("Agent is enabled.")
            except Exception as e:
                st.warning(f"⚠️ Could not check stage status: {str(e)}")
        
        if not st.session_state.session_active:
            st.info("⏸️ No active session")
        else:
            st.success("✅ Session Active")
        if "galileo_logger" in st.session_state:
            hallucination_button = st.button(
                "Log Sample Hallucination", 
                type="primary", 
            )

            if hallucination_button:
                log_hallucination(st.session_state.galileo_logger.project_name, st.session_state.galileo_logger.log_stream_name)


    
    # Start Trading Chat Button (main area)
    if not st.session_state.session_active:
        st.markdown("---")  # Add separator
        
        # Center the button with custom styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 🚀 Ready to Chat?")
            if st.button("🎯 **START TRADING CHAT** 🎯", 
                       type="primary", 
                       use_container_width=True,
                       help="Click to begin your financial analysis session!"):
                
                st.session_state.galileo_logger = GalileoLogger(
                                                    project=project_name,
                                                    log_stream=galileo_log_stream) 

                logger = st.session_state.galileo_logger

                st.session_state.session_active = True
                st.session_state.messages = []  # Clear any previous messages
                
                # Start a new Galileo session
                logger_debug.info("Starting new Galileo session")
                try:
                    # start_session doesn't return a session ID
                    st.session_state.galileo_logger.start_session(
                        name=f"Chat Session {time.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    # Generate our own session ID for reference
                    st.session_state.galileo_session_id = f"session-{time.time()}"
                    logger_debug.info(f"Started Galileo session with reference ID: {st.session_state.galileo_session_id}")
                except Exception as e:
                    logger_debug.error(f"Error starting Galileo session: {str(e)}")
                    st.session_state.galileo_session_id = None
                
                st.rerun()  # Rerun to update UI
        
        st.markdown("---")  # Add separator
    

    
    # Display chat messages
    for message in st.session_state.messages:
        # Skip system messages - they should not be displayed in the UI
        if message["role"] == "system":
            continue
        
        if message["role"] == "tool":
            # Display tool response
            with st.chat_message("tool"):
                st.markdown(escape_dollar_signs(message["content"]))
        elif message["role"] == "assistant" and "tool_calls" in message and message["tool_calls"]:
            # Display assistant tool call
            with st.chat_message("assistant"):
                st.markdown(escape_dollar_signs(message["content"]))
        else:
            # Display regular message
            with st.chat_message(message["role"]):
                st.markdown(escape_dollar_signs(message["content"]))
    
    # Only show chat input when session is active
    if st.session_state.session_active:
        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            logger_debug.info(f"Received user input: {prompt}")
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(escape_dollar_signs(prompt))
            
            try:
                # Process the chat message using standard approach
                chat_result = await process_chat_message(
                    prompt=prompt,
                    message_history=st.session_state.messages,
                    model=model_option,
                    system_prompt=system_prompt,
                    use_rag=use_rag,
                    top_k=top_k,
                    galileo_logger=st.session_state.galileo_logger,
                    is_streamlit=True,
                    ambiguous_tool_names=st.session_state.ambiguous_tool_names,
                    use_protection=st.session_state.use_protection
                )
                
                # Update the message history
                st.session_state.messages = []  # Clear and rebuild to ensure proper order
                
                # First, add all messages up to the user's message
                for msg in chat_result["updated_history"]:
                    if msg["role"] == "user" and msg["content"] == prompt:
                        st.session_state.messages.append(msg)
                        break
                    st.session_state.messages.append(msg)
                
                # Display tool calls and results if any were used
                if chat_result["tool_results"]:
                    for tool_result in chat_result["tool_results"]:
                        # Create tool call data
                        tool_call = {
                            "id": f"call-{time.time()}-{tool_result['name']}",
                            "type": "function",
                            "function": {
                                "name": tool_result["name"],
                                "arguments": json.dumps(tool_result["arguments"])
                            }
                        }
                        
                        # Display the tool call
                        with st.chat_message("assistant"):
                            if "description" in tool_result:
                                description = tool_result["description"]
                            else:
                                if tool_result["name"] == "getTickerSymbol":
                                    description = f"Looking up ticker symbol for {tool_result['arguments']['company']}..."
                                elif tool_result["name"] == "getStockPrice":
                                    description = f"Getting current price for {tool_result['arguments']['ticker']}..."
                                elif tool_result["name"] == "purchaseStocks":
                                    args = tool_result["arguments"]
                                    description = f"Processing purchase of {args['quantity']} shares of {args['ticker']}..."
                                else:
                                    description = f"Using tool: {tool_result['name']}..."
                            
                            st.markdown(escape_dollar_signs(description))
                        
                        # Add tool call message to session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": description,
                            "tool_calls": [tool_call]
                        })
                        
                        # Display the tool result
                        with st.chat_message("tool"):
                            st.markdown(escape_dollar_signs(tool_result["result"]))
                        
                        # Add tool result message to session state
                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": tool_result["result"]
                        })
                
                # Display the final assistant response
                response_message = chat_result["response_message"]
                if response_message.content:
                    # Add final response to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_message.content
                    })
                    
                    # Display the final response
                    with st.chat_message("assistant"):
                        st.markdown(escape_dollar_signs(response_message.content))
                

                

                

                
                # Display Galileo link if available
                if "galileo_logger" in st.session_state:
                    api_key = st.secrets["galileo_api_key"]
                    project_id = get_galileo_project_id(api_key, st.session_state.galileo_logger.project_name)
                    log_stream_id = get_galileo_log_stream_id(api_key, project_id, st.session_state.galileo_logger.log_stream_name) if project_id else None
                    
                    if project_id and log_stream_id:
                        project_url = f"{st.secrets['galileo_console_url']}/project/{project_id}/log-streams/{log_stream_id}"
                        # Add a small icon with tooltip in the sidebar
                        with st.sidebar:
                            st.markdown("---")  # Add a subtle separator
                            st.markdown(
                                f'<div style="font-size: 0.8em; color: #666;">'
                                f'<a href="{project_url}" target="_blank" title="View traces in Galileo">'
                                f'📊 View traces</a></div>',
                                unsafe_allow_html=True
                            )
                    else:
                        with st.sidebar:
                            st.markdown("---")
                            st.markdown(
                                '<div style="font-size: 0.8em; color: #666;">'
                                '📊 Traces logged</div>',
                                unsafe_allow_html=True
                            )
                
            except Exception as e:
                logger_debug.error(f"Error occurred: {str(e)}", exc_info=True)
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 
