import json
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage


class GroundingWithGoogleSearchTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        print(
            f"GroundingWithGoogleSearchTool - {json.dumps(tool_parameters, indent=2, ensure_ascii=False)}"
        )
        yield self.create_json_message({"result": "Hello, world!"})
