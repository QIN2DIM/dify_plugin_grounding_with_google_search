from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import json
from loguru import logger


class ToolPayload(BaseModel):
    query: str
    model_name: str | None = Field(default="gemini-2.5-flash")
    thinking_mode: bool | None = Field(default=None)
    parse_grounding_chunks: bool | None = Field(default=False)


class Ref(BaseModel):
    link: str
    title: str
    # snippet: str | None = Field(default="")
    # date: str | None = Field(default="")


class GroundingWithGoogleSearchTool(Tool):

    @staticmethod
    def format_grounded_response_with_footnotes(
        response_data: types.GenerateContentResponse,
    ) -> str:
        """
        解析 Gemini with Google Search 的响应，将信源作为脚注添加到文本中。

        Args:
            response_data: 从 Gemini API 返回的 JSON 响应，已解析为 Python 字典。

        Returns:
            一个字符串，其中包含带有引用标记的原始文本和文末的信源脚注列表。
        """
        try:
            # 1. 提取核心信息
            candidate = response_data.candidates[0]
            full_text = candidate.content.parts[0].text
            grounding_metadata = candidate.grounding_metadata

            source_chunks = grounding_metadata.grounding_chunks
            grounding_supports = grounding_metadata.grounding_supports

            # 如果没有信源信息，直接返回原始文本
            if not source_chunks or not grounding_supports:
                return full_text

            # 2. 构建信源映射表
            # source_map: { "uri": {"number": 1, "title": "..."} }
            source_map = {}
            footnote_counter = 1
            # chunk_to_footnote_map: { chunk_index: footnote_number }
            chunk_to_footnote_map = {}

            for i, chunk in enumerate(source_chunks):
                web_info = chunk.web
                if not (uri := web_info.uri):
                    continue

                # 如果是新的 URI，分配一个新的脚注编号
                if uri not in source_map:
                    source_map[uri] = {
                        "number": footnote_counter,
                        "title": web_info.title or "Unknown source",
                    }
                    footnote_counter += 1

                # 记录每个 chunk 索引对应的脚注编号
                chunk_to_footnote_map[i] = source_map[uri]["number"]

            # 3. 注入引用标记
            # 必须按 end_index 倒序处理，以防插入新字符后导致后续索引错位
            sorted_supports = sorted(
                grounding_supports, key=lambda x: x.segment.end_index, reverse=True
            )

            # 将字符串转为列表，方便在指定索引处插入内容
            modified_text_list = list(full_text)

            for support in sorted_supports:
                segment = support.segment
                chunk_indices = support.grounding_chunk_indices
                end_index = segment.end_index

                # 获取当前 support 关联的所有不重复的脚注编号
                footnote_numbers = sorted(
                    list(
                        set(
                            chunk_to_footnote_map[i]
                            for i in chunk_indices
                            if i in chunk_to_footnote_map
                        )
                    )
                )

                if not footnote_numbers:
                    continue

                # 创建引用标记，例如 "[1]" 或 "[1,2]"
                footnote_marker = f" [{','.join(map(str, footnote_numbers))}]"

                # 在分段的末尾插入标记
                modified_text_list.insert(end_index, footnote_marker)

            # 将列表重新组合成字符串
            final_text = "".join(modified_text_list)

            # 4. 生成脚注列表
            footnote_lines = ["\n\n---", "**References:**"]

            # 按脚注编号排序信源
            sorted_sources = sorted(source_map.items(), key=lambda item: item[1]["number"])

            for uri, data in sorted_sources:
                line = f"[{data['number']}] {data['title']}: {uri}"
                footnote_lines.append(line)

            footnote_section = "\n".join(footnote_lines)

            # 5. 返回最终结果
            return final_text + footnote_section

        except (KeyError, IndexError) as e:
            return f"解析响应时出错: {e}. 请检查输入的数据结构是否正确。"

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        logger.debug(f"Grounding - {json.dumps(tool_parameters, indent=2, ensure_ascii=False)}")

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

        answer_with_supports = self.format_grounded_response_with_footnotes(response)

        yield self.create_text_message(answer_with_supports)
        logger.debug(f"Answer:\n{answer_with_supports}")
