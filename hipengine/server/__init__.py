"""Optional OpenAI-compatible server surface.

Install with ``pip install hipengine[server]`` to use the FastAPI app factory or
``python -m hipengine.server`` CLI.
"""

from hipengine.server.api import ServerConfig, create_app, render_chat_prompt

__all__ = ["ServerConfig", "create_app", "render_chat_prompt"]
