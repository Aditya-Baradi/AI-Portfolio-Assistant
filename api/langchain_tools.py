"""Deprecated compatibility surface.

Evergreen no longer uses LangChain tools.  Model-visible functions live in
``api.agent_tools`` and receive authentication through a server-owned
``AgentContext``.  This module intentionally exposes no session-id-based
portfolio loaders, optimizer, trade generator, or plaintext FinRL file path.
"""

from api.agent_tools import TOOL_SCHEMAS, AgentContext, dispatch_tool

__all__ = ["TOOL_SCHEMAS", "AgentContext", "dispatch_tool"]
