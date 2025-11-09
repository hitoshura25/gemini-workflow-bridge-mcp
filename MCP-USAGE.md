# MCP Server Usage Guide for hitoshura25-gemini-workflow-bridge

## What is MCP?

The Model Context Protocol (MCP) allows AI agents like Claude to interact with external tools and services. This package implements an MCP server that exposes 5 tool(s) to AI agents.

## Configuration

### Claude Desktop

Add this to your Claude Desktop config file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "hitoshura25_gemini_workflow_bridge": {
      "command": "mcp-hitoshura25-gemini-workflow-bridge"
    }
  }
}
```

### Restart Claude Desktop

After adding the configuration, restart Claude Desktop to load the MCP server.

## Available Tools


### analyze_codebase_with_gemini

Analyze codebase using Gemini's 2M token context window

**Parameters:**

- `focus_description` (string, required): What to focus on in the analysis

- `directories` (array): Directories to analyze

- `file_patterns` (array): File patterns to include

- `exclude_patterns` (array): Patterns to exclude


**Example usage in Claude:**
```
Can you use analyze_codebase_with_gemini to...
```


### create_specification_with_gemini

Generate detailed technical specification using full codebase context

**Parameters:**

- `feature_description` (string, required): What feature to specify

- `context_id` (string): Optional context ID from previous analysis

- `spec_template` (string): Specification template to use

- `output_path` (string): Where to save the spec


**Example usage in Claude:**
```
Can you use create_specification_with_gemini to...
```


### review_code_with_gemini

Comprehensive code review using Gemini

**Parameters:**

- `files` (array): Files to review

- `review_focus` (array): Areas to focus on

- `spec_path` (string): Path to spec to review against

- `output_path` (string): Where to save review


**Example usage in Claude:**
```
Can you use review_code_with_gemini to...
```


### generate_documentation_with_gemini

Generate comprehensive documentation with full codebase context

**Parameters:**

- `documentation_type` (string, required): Type of documentation

- `scope` (string, required): What to document

- `output_path` (string): Where to save documentation

- `include_examples` (boolean): Include code examples


**Example usage in Claude:**
```
Can you use generate_documentation_with_gemini to...
```


### ask_gemini

General-purpose Gemini query with optional codebase context

**Parameters:**

- `prompt` (string, required): Question or task for Gemini

- `include_codebase_context` (boolean): Load full codebase context

- `context_id` (string): Reuse cached context

- `temperature` (number): Temperature for generation


**Example usage in Claude:**
```
Can you use ask_gemini to...
```



## Testing the Connection

Once configured, you can ask Claude:

```
What MCP servers are available?
```

You should see `hitoshura25_gemini_workflow_bridge` in the list.

## Troubleshooting

### Server not appearing

1. Check that the package is installed:
   ```bash
   which mcp-hitoshura25-gemini-workflow-bridge
   ```

2. Verify the config file syntax is valid JSON

3. Check Claude Desktop logs:
   - macOS: `~/Library/Logs/Claude/`
   - Windows: `%APPDATA%\Claude\logs\`

### Tool execution errors

If tools fail to execute:

1. Check that all required parameters are provided
2. Verify your environment has necessary permissions
3. Review the MCP server logs (stderr output)

## Development

### Running the MCP Server Manually

```bash
# Start the MCP server (stdio mode)
mcp-hitoshura25-gemini-workflow-bridge

# The server will wait for JSON-RPC requests on stdin
```

### Testing with Manual Requests

You can test the server by sending JSON-RPC requests:

```bash
echo '{"method": "tools/list", "id": 1}' | mcp-hitoshura25-gemini-workflow-bridge
```

## Protocol Details

This MCP server implements the Model Context Protocol using the official FastMCP SDK:
- **Protocol**: MCP (Model Context Protocol) via JSON-RPC over stdio
- **SDK**: Official MCP Python SDK (FastMCP)
- **Methods**: All standard MCP methods including:
  - `initialize` - Protocol initialization and capability negotiation
  - `initialized` - Initialization confirmation
  - `tools/list` - List available tools
  - `tools/call` - Execute a tool

The FastMCP SDK automatically handles:
- Protocol version negotiation
- Capability exchange
- Error handling and validation
- Message formatting

### Example Tool Call

When using this server through an MCP client (like Claude Desktop), the SDK handles all protocol details. Your tools are called with clean, validated arguments:


**Tool Definition:**
- Name: `analyze_codebase_with_gemini`
- Parameters: focus_description (string), directories (array), file_patterns (array), exclude_patterns (array)



**Result:**
The tool returns its result as a string, which the SDK formats into proper MCP response messages.

## Further Reading

- [MCP Documentation](https://modelcontextprotocol.io/)
- [Claude Desktop MCP Guide](https://modelcontextprotocol.io/docs/tools/claude-desktop)