tools = {
    "getTickerSymbol": {
        "description": "Get the ticker symbol for a company",
        "parameters": {
            "type": "object",
            "properties": {
                "company": {
                    "type": "string",
                    "description": "The name of the company"
                }
            },
            "required": ["company"]
        }
    },
    "purchaseStocks": {
        "description": "Purchase a specified number of shares of a stock at a given price.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol to purchase"
                },
                "quantity": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "The number of shares to purchase"
                },
                "price": {
                    "type": "number",
                    "minimum": 0.01,
                    "description": "The price per share at which to purchase"
                }
            },
            "required": ["ticker", "quantity", "price"]
        }
    },
    "sellStocks": {
        "description": "Sell a specified number of shares of a stock at a given price.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol to sell"
                },
                "quantity": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "The number of shares to sell"
                },
                "price": {
                    "type": "number",
                    "minimum": 0.01,
                    "description": "The price per share at which to sell"
                }
            },
            "required": ["ticker", "quantity", "price"]
        }
    },
    "getStockPrice": {
        "description": "Get the current stock price and other market data for a given ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol to look up"
                }
            },
            "required": ["ticker"]
        }
    }
}


# Define ambiguous tool name mappings
AMBIGUOUS_TOOL_NAMES = {
    "purchaseStocks": "tradeStocks",
    "sellStocks": "makeTrade"
}

# Define ambiguous tool descriptions
AMBIGUOUS_TOOL_DESCRIPTIONS = {
    "tradeStocks": "Execute a trade for a specified number of shares of a stock at a given price.",
    "makeTrade": "Execute a market trade for a specified number of shares of a stock at a given price."
}

# Define ambiguous parameter descriptions
AMBIGUOUS_PARAMETER_DESCRIPTIONS = {
    "tradeStocks": {
        "ticker": "The stock ticker symbol to trade",
        "quantity": "The number of shares to trade",
        "price": "The price per share for the trade"
    },
    "makeTrade": {
        "ticker": "The stock ticker symbol for the trade",
        "quantity": "The number of shares involved in the trade",
        "price": "The price per share for the transaction"
    }
}