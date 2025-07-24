from collections.abc import Generator
from contextlib import suppress
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import json


class ToolPayload(BaseModel):
    query: str
    model_name: str | None = Field(default="gemini-2.5-flash")
    thinking_mode: bool | None = Field(default=None)
    parse_grounding_chunks: bool | None = Field(default=False)


class Ref(BaseModel):
    link: str
    title: str
    snippet: str | None = Field(default="")
    date: str | None = Field(default="")


class GroundingWithGoogleSearchTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        print(f"Grounding - {json.dumps(tool_parameters, indent=2, ensure_ascii=False)}")

        tp = ToolPayload(**tool_parameters)

        if not tp.thinking_mode:
            thinking_budget = 0
        elif tp.thinking_mode:
            thinking_budget = -1
        else:
            thinking_budget = None

        api_key = self.runtime.credentials.get("api_key")
        client = genai.Client(api_key=api_key)

        contents = [types.Content(role="user", parts=[types.Part.from_text(text=tp.query)])]

        tools = [types.Tool(google_search=types.GoogleSearch())]

        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            tools=tools,
            response_mime_type="text/plain",
        )

        response = client.models.generate_content(
            model=tp.model_name, contents=contents, config=generate_content_config
        )

        metadata = {"refs": []}
        with suppress(Exception):
            for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
                web = chunk.web
                metadata["refs"].append(Ref(link=web["uri"], title=web["title"]))

        yield self.create_text_message(response.text)
        yield self.create_json_message(metadata)
