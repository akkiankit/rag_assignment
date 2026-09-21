import logging
import os

from dotenv import load_dotenv
from openai import AzureOpenAI

from src.observability.tracing import trace_step
from src.providers.llm.base import LLMProvider
from src.schemas import RAGResponse


load_dotenv()

logger = logging.getLogger(__name__)


class AzureOpenAILLMProvider(LLMProvider):
    """
    Azure OpenAI implementation of the LLM provider contract.

    Required environment variables:
        AZURE_OPENAI_ENDPOINT
        AZURE_OPENAI_API_KEY
        AZURE_OPENAI_API_VERSION
        AZURE_OPENAI_CHAT_DEPLOYMENT

    Security:
        Credentials are loaded from environment variables and never written
        to logs.
    """

    def __init__(self) -> None:
        """
        Initialize the Azure OpenAI client and validate configuration.

        Raises:
            ValueError:
                If required Azure OpenAI configuration is missing.
        """

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_ENDPOINT": endpoint,
                "AZURE_OPENAI_API_KEY": api_key,
                "AZURE_OPENAI_API_VERSION": api_version,
                "AZURE_OPENAI_CHAT_DEPLOYMENT": deployment,
            }.items()
            if not value
        ]

        if missing:
            raise ValueError(
                "Missing Azure OpenAI LLM configuration: "
                + ", ".join(missing)
            )

        self.deployment = deployment

        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )

        logger.info(
            "Azure OpenAI LLM provider initialized | deployment=%s",
            self.deployment,
        )

    def generate_structured_response(self, system_prompt: str, user_prompt: str,) -> RAGResponse:
        """
        Generate a structured and validated RAG answer.

        Args:
            system_prompt:
                Instructions that define grounding and response rules.

            user_prompt:
                Retrieved context and user question.

        Returns:
            Validated RAGResponse object.

        Raises:
            ValueError:
                If either prompt is empty.

            RuntimeError:
                If the Azure OpenAI request fails or no structured response
                can be parsed.
        """

        if not system_prompt.strip():
            raise ValueError(
                "system_prompt must not be empty"
            )

        if not user_prompt.strip():
            raise ValueError(
                "user_prompt must not be empty"
            )

        logger.info(
            "Generating structured RAG response"
        )

        try:
            with trace_step("azure_llm_structured_generation"):
                completion = (
                    self.client.beta.chat.completions.parse(
                        model=self.deployment,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": user_prompt,
                            },
                        ],
                        response_format=RAGResponse,
                        # temperature=0,
                    )
                )

        except Exception as exc:
            logger.exception(
                "Azure OpenAI structured generation failed"
            )

            raise RuntimeError(
                "Failed to generate structured RAG response"
            ) from exc

        message = completion.choices[0].message

        if message.parsed is None:
            raise RuntimeError(
                "Azure OpenAI returned no parsed structured response"
            )

        logger.info(
            "Structured RAG response generated | "
            "insufficient_context=%s | confidence=%s",
            message.parsed.insufficient_context,
            message.parsed.confidence,
        )

        return message.parsed