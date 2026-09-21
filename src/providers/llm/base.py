from abc import ABC, abstractmethod

from src.schemas import RAGResponse


class LLMProvider(ABC):
    """
    Contract for language-model providers used by the RAG pipeline.
    """

    @abstractmethod
    def generate_structured_response(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> RAGResponse:
        """
        Generate a validated structured RAG response.

        Args:
            system_prompt:
                System-level instructions controlling model behavior.

            user_prompt:
                User question plus retrieved context.

        Returns:
            Validated RAGResponse instance.
        """
        raise NotImplementedError