"""Activity feed component for displaying real-time agent updates."""

from collections.abc import Iterator
from datetime import datetime
from typing import Any

from rich.markup import escape as rich_escape
from textual.containers import VerticalScroll
from textual.widgets import Static


def escape_markup(text: str) -> str:
    """Escape Rich markup characters."""
    return str(rich_escape(text))


class ActivityFeed(VerticalScroll):
    """Scrollable activity feed showing agent actions in real-time."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize activity feed."""
        super().__init__(*args, **kwargs)
        self._content_widget: Static | None = None
        self._activities: list[dict[str, Any]] = []

    def compose(self) -> Iterator[Static]:
        """Create the feed content area."""
        content = Static("", id="activity_content")
        self._content_widget = content
        yield content

    def add_message(
        self, message: str, message_type: str = "info", timestamp: datetime | None = None
    ) -> None:
        """Add a simple text message to the feed."""
        if timestamp is None:
            timestamp = datetime.now()

        activity = {
            "type": "message",
            "message": message,
            "message_type": message_type,
            "timestamp": timestamp,
        }
        self._activities.append(activity)
        self._render_activities()

    def add_tool_execution(
        self,
        tool_name: str,
        args: dict[str, Any],
        status: str = "running",
        result: Any = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Add a tool execution block to the feed."""
        if timestamp is None:
            timestamp = datetime.now()

        activity = {
            "type": "tool",
            "tool_name": tool_name,
            "args": args,
            "status": status,
            "result": result,
            "timestamp": timestamp,
        }
        self._activities.append(activity)
        self._render_activities()

    def update_last_tool_status(self, status: str, result: Any = None) -> None:
        """Update the status of the most recent tool execution."""
        # Find the last tool activity
        for activity in reversed(self._activities):
            if activity["type"] == "tool":
                activity["status"] = status
                if result is not None:
                    activity["result"] = result
                break

        self._render_activities()

    def clear(self) -> None:
        """Clear all activities from the feed."""
        self._activities.clear()
        if self._content_widget:
            self._content_widget.update("")

    def _render_activities(self) -> None:
        """Render all activities to the content widget."""
        if not self._content_widget:
            return

        if not self._activities:
            placeholder = (
                "\n\n[dim italic]Waiting for agent to start...[/]\n\n"
                "[dim]The activity feed will show real-time updates here.[/]"
            )
            self._content_widget.update(placeholder)
            self._content_widget.set_classes("placeholder")
            return

        lines = []
        for activity in self._activities:
            if activity["type"] == "message":
                lines.append(self._render_message(activity))
            elif activity["type"] == "tool":
                lines.append(self._render_tool(activity))

        content = "\n\n".join(lines)
        self._content_widget.update(content)
        self._content_widget.remove_class("placeholder")

        # Auto-scroll to bottom
        self.call_later(self.scroll_end, animate=False)

    def _render_message(self, activity: dict[str, Any]) -> str:
        """Render a simple message activity."""
        timestamp = activity["timestamp"].strftime("%H:%M:%S")
        message = escape_markup(activity["message"])
        message_type = activity["message_type"]

        # Type-specific styling
        if message_type == "success":
            icon = "[#10b981]✓[/]"
            style = "activity-status-success"
        elif message_type == "error":
            icon = "[#ef4444]✗[/]"
            style = "activity-status-error"
        elif message_type == "warning":
            icon = "[#f59e0b]⚠[/]"
            style = ""
        else:
            icon = "[#6366f1]●[/]"
            style = "activity-status-active"

        return f"[dim]{timestamp}[/] {icon} [{style}]{message}[/]" if style else f"{message}"

    def _render_tool(self, activity: dict[str, Any]) -> str:
        """Render a tool execution block."""
        timestamp = activity["timestamp"].strftime("%H:%M:%S")
        tool_name = activity["tool_name"]
        args = activity["args"]
        status = activity["status"]
        result = activity["result"]

        # Determine tool type for styling
        tool_class = self._get_tool_class(tool_name)

        # Status indicator
        if status == "running":
            status_icon = "[#f59e0b]●[/] In progress..."
        elif status == "completed":
            status_icon = "[#10b981]✓[/] Done"
        elif status == "failed":
            status_icon = "[#ef4444]✗[/] Failed"
        else:
            status_icon = "[dim]○[/] Unknown"

        # Build the tool block
        lines = []
        lines.append(f"[dim]{timestamp}[/]")

        # Tool header
        header_class = "tool-header" if tool_class else ""
        lines.append(f"[{header_class}]▍ {escape_markup(tool_name)}[/] {status_icon}")

        # Arguments
        if args:
            for key, value in list(args.items())[:3]:  # Show first 3 args
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:97] + "..."

                # Special formatting for command
                if key == "command" and tool_name == "terminal_execute":
                    lines.append(f"  [#10b981]$ {escape_markup(value_str)}[/]")
                else:
                    lines.append(f"  [dim]{key}:[/] {escape_markup(value_str)}")

        # Result (if completed)
        if status in ("completed", "failed") and result:
            result_str = str(result)
            if len(result_str) > 200:
                result_str = result_str[:197] + "..."

            lines.append(f"  [dim]→[/] {escape_markup(result_str)}")

        content = "\n".join(lines)

        # Wrap in styled container
        return f"[{tool_class}]{content}[/]"

    def _get_tool_class(self, tool_name: str) -> str:
        """Get the CSS class for a tool type."""
        tool_classes = {
            "terminal_execute": "tool-terminal",
            "thinking": "tool-thinking",
            "result": "tool-result",
            "error": "tool-error",
        }
        return tool_classes.get(tool_name, "tool-block")
