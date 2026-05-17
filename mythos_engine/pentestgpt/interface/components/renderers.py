"""Tool-specific renderers for beautiful output formatting."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from rich.markup import escape as rich_escape


class BaseToolRenderer(ABC):
    """Base class for tool-specific renderers."""

    tool_name: ClassVar[str] = ""
    border_color: ClassVar[str] = "#525252"

    @classmethod
    def escape_markup(cls, text: str) -> str:
        """Escape Rich markup characters."""
        return str(rich_escape(text))

    @classmethod
    @abstractmethod
    def render(cls, tool_data: dict[str, Any]) -> str:
        """Render tool execution data to Rich markup string."""
        pass

    @classmethod
    def get_status_icon(cls, status: str) -> str:
        """Get status icon for tool execution."""
        icons = {
            "running": "[#f59e0b]●[/] Running",
            "completed": "[#10b981]✓[/] Success",
            "failed": "[#ef4444]✗[/] Failed",
            "error": "[#ef4444]✗[/] Error",
        }
        return icons.get(status, "[dim]○[/]")


class TerminalRenderer(BaseToolRenderer):
    """Renderer for terminal/command execution."""

    tool_name: ClassVar[str] = "terminal_execute"
    border_color: ClassVar[str] = "#10b981"  # Green

    @classmethod
    def render(cls, tool_data: dict[str, Any]) -> str:
        """Render terminal command execution."""
        args = tool_data.get("args", {})
        status = tool_data.get("status", "unknown")
        result = tool_data.get("result", "")

        command = args.get("command", "")
        if not command:
            return ""

        # Truncate long commands
        display_command = cls.escape_markup(command)
        if len(display_command) > 120:
            display_command = display_command[:117] + "..."

        lines = []
        lines.append(f"[#10b981]$ {display_command}[/]")

        # Add result if available
        if status in ("completed", "failed") and result:
            result_str = str(result)
            if len(result_str) > 500:
                result_str = result_str[:497] + "..."

            # Format output
            if result_str:
                lines.append("[dim]Output:[/]")
                for line in result_str.split("\n")[:20]:  # Max 20 lines
                    lines.append(f"  {cls.escape_markup(line)}")

        return "\n".join(lines)


class ThinkingRenderer(BaseToolRenderer):
    """Renderer for AI thinking/planning blocks."""

    tool_name: ClassVar[str] = "thinking"
    border_color: ClassVar[str] = "#8b5cf6"  # Purple

    @classmethod
    def render(cls, tool_data: dict[str, Any]) -> str:
        """Render thinking content."""
        args = tool_data.get("args", {})
        content = args.get("content", "")

        if not content:
            return "[#8b5cf6][italic]Analyzing...[/][/]"

        # Truncate long content
        display_content = cls.escape_markup(content)
        if len(display_content) > 300:
            display_content = display_content[:297] + "..."

        return f"[#8b5cf6]💭 {display_content}[/]"


class ResultRenderer(BaseToolRenderer):
    """Renderer for generic results."""

    tool_name: ClassVar[str] = "result"
    border_color: ClassVar[str] = "#6366f1"  # Indigo

    @classmethod
    def render(cls, tool_data: dict[str, Any]) -> str:
        """Render result data."""
        result = tool_data.get("result", "")
        title = tool_data.get("title", "Result")

        if not result:
            return ""

        result_str = str(result)
        if len(result_str) > 500:
            result_str = result_str[:497] + "..."

        lines = []
        lines.append(f"[bold]{cls.escape_markup(title)}[/]")
        lines.append(cls.escape_markup(result_str))

        return "\n".join(lines)


# Registry of renderers
RENDERER_REGISTRY: dict[str, type[BaseToolRenderer]] = {
    "terminal_execute": TerminalRenderer,
    "thinking": ThinkingRenderer,
    "result": ResultRenderer,
}


def get_renderer(tool_name: str) -> type[BaseToolRenderer] | None:
    """Get renderer for a specific tool name."""
    return RENDERER_REGISTRY.get(tool_name)


def render_tool(tool_data: dict[str, Any]) -> str:
    """Render tool data using appropriate renderer."""
    tool_name = tool_data.get("tool_name", "")
    renderer = get_renderer(tool_name)

    if renderer:
        return renderer.render(tool_data)

    # Fallback: generic rendering
    return f"Tool: {tool_name}"
