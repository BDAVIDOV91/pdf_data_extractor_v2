#!/bin/bash

# Get the directory of this script to ensure it runs from the correct location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Start memory-bank-MCP
echo "Starting memory-bank-MCP..."
(cd mcp_servers/memory-bank-MCP && npm start) &

# Start mcp-compass
echo "Starting mcp-compass..."
(cd mcp_servers/mcp-compass && node build/index.js) &

# Start mcp-memory-service
echo "Starting mcp-memory-service..."
(cd mcp_servers/mcp-memory-service && uv run --active memory) &

echo "All MCP servers started."
