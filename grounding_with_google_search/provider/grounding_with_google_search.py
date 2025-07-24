from typing import Any

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError
from google import genai


class GroundingWithGoogleSearchProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            api_key = credentials.get("api_key")
            client = genai.Client(api_key=api_key)
            client.models.count_tokens(
                model="gemini-2.0-flash-001", contents="why is the sky blue?"
            )
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
