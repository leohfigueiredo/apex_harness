"""
apex_harness.mcp_client
MCP (Model Context Protocol) Manager for Apex Harness.
Supports discovering, loading, and safely executing MCP tools from configured servers.
"""
import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

DEFAULT_CONFIG_PATHS = [
    Path("mcp_config.json"),
    Path.home() / ".gemini" / "config" / "mcp_config.json",
    Path.home() / ".config" / "apex" / "mcp.json"
]

class MCPManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = self._resolve_config_path(config_path)
        self.servers: Dict[str, Dict[str, Any]] = {}
        self.discovered_tools: Dict[str, Any] = {}
        self.tool_to_server: Dict[str, Tuple[str, str]] = {}  # tool_alias -> (server_name, original_name)
        self.load_config()

    def _resolve_config_path(self, custom_path: Optional[str]) -> Optional[Path]:
        if custom_path:
            p = Path(custom_path).expanduser().resolve()
            if p.exists():
                return p
        env_path = os.environ.get("APEX_MCP_CONFIG")
        if env_path:
            p = Path(env_path).expanduser().resolve()
            if p.exists():
                return p
        for p in DEFAULT_CONFIG_PATHS:
            if p.exists():
                return p
        return None

    def load_config(self) -> Dict[str, Dict[str, Any]]:
        """Load configured MCP servers from JSON config."""
        self.servers = {}
        if not self.config_path or not self.config_path.exists():
            return self.servers

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.servers = data.get("mcpServers", {})
        except Exception as e:
            print(f"[MCP] Erro ao ler {self.config_path}: {e}")
            self.servers = {}
        return self.servers

    def get_server_params(self, server_name: str) -> Optional[StdioServerParameters]:
        """Convert server configuration to StdioServerParameters."""
        conf = self.servers.get(server_name)
        if not conf:
            return None

        command = conf.get("command")
        args = conf.get("args", [])
        env = conf.get("env")

        merged_env = os.environ.copy()
        if env:
            merged_env.update({str(k): str(v) for k, v in env.items()})

        return StdioServerParameters(command=command, args=args, env=merged_env)

    async def _discover_tools_async(self, server_name: str, timeout: float = 12.0) -> List[Any]:
        params = self.get_server_params(server_name)
        if not params:
            return []

        try:
            async with asyncio.timeout(timeout):
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools_result = await session.list_tools()
                        return tools_result.tools
        except Exception:
            return []

    def discover_tools_for_server(self, server_name: str, timeout: float = 12.0) -> List[Dict[str, Any]]:
        """Synchronously discover tools for a single server."""
        try:
            raw_tools = asyncio.run(self._discover_tools_async(server_name, timeout=timeout))
        except Exception:
            raw_tools = []

        formatted = []
        for t in raw_tools:
            tool_alias = f"mcp_{server_name}_{t.name}"
            self.tool_to_server[tool_alias] = (server_name, t.name)
            schema = t.inputSchema if hasattr(t, "inputSchema") and t.inputSchema else {"type": "object", "properties": {}}
            item = {
                "type": "function",
                "function": {
                    "name": tool_alias,
                    "description": f"[MCP {server_name}] {t.description or t.name}",
                    "parameters": schema
                }
            }
            self.discovered_tools[tool_alias] = item
            formatted.append(item)
        return formatted

    def discover_all_tools(self, timeout_per_server: float = 8.0) -> List[Dict[str, Any]]:
        """Discover tools from all configured MCP servers."""
        all_tools = []
        for s_name in self.servers.keys():
            tools = self.discover_tools_for_server(s_name, timeout=timeout_per_server)
            all_tools.extend(tools)
        return all_tools

    async def _call_tool_async(self, server_name: str, tool_name: str, arguments: dict, timeout: float = 60.0) -> str:
        params = self.get_server_params(server_name)
        if not params:
            return f"Error: MCP server '{server_name}' configuration not found."

        try:
            async with asyncio.timeout(timeout):
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        res = await session.call_tool(tool_name, arguments=arguments)
                        if hasattr(res, "content") and res.content:
                            parts = []
                            for c in res.content:
                                if hasattr(c, "text"):
                                    parts.append(c.text)
                                else:
                                    parts.append(str(c))
                            return "\n".join(parts)
                        return "Tool executed successfully (empty output)."
        except asyncio.TimeoutError:
            return f"Error: MCP tool '{tool_name}' on server '{server_name}' timed out after {timeout}s."
        except Exception as e:
            return f"Error calling MCP tool '{tool_name}' on '{server_name}': {str(e)}"

    def execute_mcp_tool(self, tool_alias: str, arguments: dict) -> str:
        """Execute an MCP tool by its alias."""
        if tool_alias not in self.tool_to_server:
            return f"Error: Unrecognized MCP tool alias '{tool_alias}'."
        server_name, original_name = self.tool_to_server[tool_alias]
        try:
            return asyncio.run(self._call_tool_async(server_name, original_name, arguments))
        except Exception as e:
            return f"Error running MCP tool '{tool_alias}': {str(e)}"

# Global singleton instance
_global_mcp_manager: Optional[MCPManager] = None

def get_mcp_manager() -> MCPManager:
    global _global_mcp_manager
    if _global_mcp_manager is None:
        _global_mcp_manager = MCPManager()
    return _global_mcp_manager
