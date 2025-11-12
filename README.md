# Galileo TradeDesk

A modern Streamlit-based financial trading assistant that uses Retrieval Augmented Generation (RAG) with LanceDB to provide intelligent stock analysis and trading insights. The application includes advanced AI models, Galileo observability, and comprehensive financial tooling.

## Key Features

- **Intelligent Chat**: Natural conversation with financial analysis capabilities
- **Enhanced Tool Integration**: Automatic tool execution for stock data and trading
- **RAG-powered Analysis**: Uses LanceDB for semantic search over financial data
- **Multi-Model Support**: OpenAI GPT models with intelligent capabilities
- **Stock Trading Simulation**: Purchase/sell functionality with realistic market data
- **Galileo Observability**: Comprehensive logging and monitoring
- **Galileo Protect**: AI safety and content filtering
- **Modern UI**: Clean, intuitive Streamlit interface with real-time chat

## Project Structure

```
.
├── app.py                    # Main Streamlit application
├── galileo_api_helper.py     # Helper functions for Galileo API
├── tools/                    # Financial trading tools
│   ├── purchase_stocks.py    # Stock purchase simulation
│   ├── sell_stocks.py        # Stock sale simulation
│   ├── get_stock_price.py   # Real-time stock price lookup
│   ├── get_ticker_symbol.py # Company to ticker symbol lookup
│   └── tool_definitions.py  # Tool definitions and configurations
├── data/                     # Data management
│   ├── __init__.py          # Data package initialization
│   └── vectordb_setup.py    # LanceDB setup and population functions
├── chat_lib/                 # Shared utilities
│   └── galileo_logger.py    # Galileo logging utilities
└── log_hallucination.py     # Hallucination logging utility
```

## Installation

1. **Clone the repository**
2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Set up environment variables** in `.streamlit/secrets.toml`:

```toml
# OpenAI Configuration
openai_api_key = "your_openai_api_key"

# Galileo Configuration
galileo_api_key = "your_galileo_api_key"
galileo_project = "your_galileo_project"
galileo_log_stream = "your_galileo_log_stream"
galileo_console_url = "your_galileo_console_url"
galileo_stage_agent_id = "your_stage_agent_id"

# Alpha Vantage API (for stock data)
alpha_vantage_api_key = "your_alpha_vantage_key"
```

4. **LanceDB will be automatically populated** with sample financial data on first run, or you can manually populate it:

```bash
cd finance-chat-streamlit
python -c "from data.vectordb_setup import check_and_populate_lancedb; check_and_populate_lancedb()"
```

## Running the Application

### Streamlit UI

```bash
streamlit run app.py
```

The application will be available at http://localhost:8501

## AI Models

The application supports multiple AI models:

### OpenAI Models
- **GPT-4o-mini**: Fast, cost-effective responses (default)
- **GPT-4o**: Advanced reasoning and analysis
- **GPT-4**: High-performance model
- **GPT-3.5-turbo**: Legacy model support

## Usage

The application provides:

1. **Interactive Chat Interface**: Real-time financial analysis and trading insights
2. **Stock Trading Simulation**: Purchase/sell stocks with realistic market data
3. **RAG-Powered Analysis**: Context-aware responses using LanceDB
4. **Galileo Observability**: Comprehensive monitoring and debugging
5. **AI Safety**: Galileo Protect integration for content filtering

## Tools

### Trading Tools
- **Stock Purchase Simulation**: Simulates buying stocks with specified ticker, quantity, and price
- **Stock Sale Simulation**: Simulates selling stocks with specified ticker, quantity, and price
- **Ticker Symbol Lookup**: Convert company names to stock tickers
- **Stock Price Lookup**: Get real-time stock prices using AlphaVantage API

### RAG System
- **LanceDB Integration**: Local vector database for semantic search
- **Financial Context**: Pre-loaded sample data for major stocks and market information
- **Smart Retrieval**: Context-aware responses based on relevant financial documents
- **Automatic Setup**: LanceDB is automatically configured and populated on startup

### AI Safety
- **Galileo Protect**: Content filtering and AI safety measures
- **Response Validation**: Ensures appropriate and safe AI responses

## Observability

The application uses Galileo for comprehensive monitoring:

- **Chat Interactions**: Log all user queries and AI responses
- **Tool Usage**: Track which trading tools are used and when
- **RAG Performance**: Monitor document retrieval and relevance
- **AI Model Performance**: Track response times and quality
- **Error Tracking**: Comprehensive error logging and debugging
- **Session Management**: Monitor user sessions and interactions

## LanceDB Setup

### What is LanceDB?
LanceDB is a local vector database that stores document embeddings for semantic search. It's faster and more cost-effective than cloud-based solutions.

### Automatic Configuration
The app automatically:
- Creates the data directory if it doesn't exist
- Sets up LanceDB with the correct schema
- Populates the database with sample financial data
- Handles schema validation and table recreation if needed

### Schema
The LanceDB table uses a structured schema with:
- **id**: Unique document identifier
- **text**: Financial document content
- **embedding**: 1536-dimensional vector representation
- **metadata**: Structured fields including:
  - company, ticker, sector, type
  - index, topic (for market-wide documents)

### Sample Data Included
The system automatically adds sample financial documents:
- **Apple (AAPL)**: Q4 earnings, revenue data
- **Microsoft (MSFT)**: Azure growth, AI initiatives
- **Tesla (TSLA)**: Vehicle deliveries, energy storage
- **NVIDIA (NVDA)**: AI chip market, data center growth
- **Amazon (AMZN)**: Holiday quarter results, AWS performance
- **Market Overview**: S&P 500 performance, Federal Reserve policy

### Adding Custom Data
To add your own financial documents programmatically:

```python
from data.vectordb_setup import add_document_to_lancedb

add_document_to_lancedb(
    text="Your financial document text here",
    metadata={"company": "Company Name", "type": "earnings"}
)
```

## Configuration Options

### Sidebar Controls
- **Galileo Log Stream**: Configure logging stream name
- **Ambiguous Tool Names**: Enable/disable tool name obfuscation for testing
- **Galileo Protect**: Enable/disable AI safety measures
- **Model Selection**: Choose from available OpenAI models

### RAG Settings
- **Top K**: Number of relevant documents to retrieve (default: 10)
- **Automatic Population**: Sample data is automatically added on startup

## Troubleshooting

### Common Issues
1. **LanceDB Connection Errors**: The app automatically creates the data directory and handles table setup
2. **Empty RAG Responses**: Sample data is automatically populated on startup
3. **Galileo Connection Issues**: Check API keys and project configuration

### Getting Help
- Check the Galileo console for detailed logs
- Review the debug configuration in the sidebar
- Ensure all environment variables are properly set
- The app includes comprehensive error logging and automatic fallback mechanisms

## Development

### Code Structure
- **Modular Design**: Separate modules for tools, data management, and utilities
- **Error Handling**: Comprehensive error handling with automatic fallbacks
- **Logging**: Detailed logging throughout the application
- **Type Hints**: Full type annotations for better code quality

### Adding New Tools
1. Create tool function in the `tools/` directory
2. Add tool definition to `tools/tool_definitions.py`
3. Import and register in `app.py`
4. Update this README with new tool documentation
