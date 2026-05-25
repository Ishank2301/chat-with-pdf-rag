"""Query and response generation engine."""

import logging
from typing import Any, Dict, List

import tenacity
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM

from .config import settings

logger = logging.getLogger(__name__)


class RAGEngine:
    """Runs prompt construction and LLM response generation."""

    _instance = None
    _llm = None

    def __new__(cls):
        """Create a shared LLM connection."""
        if cls._instance is None:
            cls._instance = super(RAGEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model: str = settings.llm_model,
        provider: str = settings.llm_provider,
    ):
        """
        Initialize RAG engine.

        Args:
            model: LLM model name.
            provider: LLM provider name.
        """
        if self._initialized:
            return

        self._provider = provider.lower()
        self._model_name = model
        self._llm = self._build_llm()
        self._initialized = True
        logger.info("RAG engine initialized with %s model: %s", self._provider, model)

    def _build_llm(self):
        """Create the configured LLM client."""
        if self._provider == "gemini":
            if not settings.google_api_key:
                raise ValueError("GOOGLE_API_KEY is required when LLM_PROVIDER=gemini")
            return ChatGoogleGenerativeAI(
                model=self._model_name,
                google_api_key=settings.google_api_key,
            )

        if self._provider == "ollama":
            return OllamaLLM(
                model=self._model_name,
                base_url=settings.ollama_base_url,
            )

        raise ValueError(f"Unsupported LLM provider: {self._provider}")

    def _invoke(self, prompt: str) -> str:
        """Invoke the configured LLM and return plain text."""
        response = self._llm.invoke(prompt)
        if hasattr(response, "content"):
            return str(response.content).strip()
        return str(response).strip()

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
    )
    def generate_response(
        self,
        question: str,
        relevant_chunks: List[str],
        max_length: int = 3,
    ) -> str:
        """
        Generate response using relevant document chunks.

        Args:
            question: User query.
            relevant_chunks: List of relevant document chunks.
            max_length: Max sentences in response.

        Returns:
            Generated answer.
        """
        if not relevant_chunks:
            logger.warning("No relevant chunks provided")
            return "I do not have relevant information to answer this question."

        context = "\n\n".join(relevant_chunks)

        prompt = f"""You are an intelligent assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question accurately.
If you do not know the answer based on the context, say that you do not know.
Keep your answer concise (maximum {max_length} sentences).

Context:
{context}

Question: {question}

Answer:"""

        try:
            logger.info("Generating response for query: %s", question[:50])
            answer = self._invoke(prompt)
            logger.info("Response generated successfully")
            return answer

        except Exception as e:
            logger.error("Response generation failed: %s", e)
            raise

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
    )
    def summarize_documents(
        self,
        documents: List[str],
        summary_length: str = "short",
    ) -> str:
        """
        Generate summary of documents.

        Args:
            documents: List of document texts.
            summary_length: "short" or "detailed".

        Returns:
            Generated summary.
        """
        if not documents:
            logger.warning("No documents provided for summarization")
            return "No documents to summarize."

        combined_text = "\n\n".join(documents)

        if summary_length == "detailed":
            length_instruction = "Provide a detailed summary (1-2 paragraphs)."
        else:
            length_instruction = "Provide a brief summary (1-2 sentences)."

        prompt = f"""Summarize the following document(s):

{combined_text}

{length_instruction} Focus on key insights and main ideas.

Summary:"""

        try:
            logger.info("Summarizing %s document(s)", len(documents))
            summary = self._invoke(prompt)
            logger.info("Summary generated successfully")
            return summary

        except Exception as e:
            logger.error("Summarization failed: %s", e)
            raise

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
    )
    def analyze_resume_ats_score(self, resume_text: str) -> Dict[str, Any]:
        """
        Analyze resume for ATS compatibility and score.

        Args:
            resume_text: Resume content.

        Returns:
            Dictionary with ATS score and suggestions.
        """
        prompt = f"""Analyze this resume for ATS (Applicant Tracking System) compatibility.

Resume:
{resume_text}

Provide your analysis in this exact format:

ATS_SCORE: [0-100]

STRENGTHS:
- [Strength 1]
- [Strength 2]
- [Strength 3]

WEAKNESSES:
- [Weakness 1]
- [Weakness 2]
- [Weakness 3]

SUGGESTIONS_TO_IMPROVE:
- [Suggestion 1]
- [Suggestion 2]
- [Suggestion 3]

KEY_MISSING_ELEMENTS:
- [Missing 1]
- [Missing 2]"""

        try:
            logger.info("Analyzing resume for ATS score")
            analysis = self._invoke(prompt)
            logger.info("ATS analysis completed")

            return self._parse_ats_analysis(analysis)

        except Exception as e:
            logger.error("ATS analysis failed: %s", e)
            raise

    def _parse_ats_analysis(self, analysis: str) -> Dict[str, Any]:
        """Parse ATS analysis response into structured format."""
        result = {
            "raw_analysis": analysis,
            "ats_score": 0,
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
            "missing_elements": [],
        }

        lines = analysis.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()

            if line.startswith("ATS_SCORE:"):
                try:
                    score_text = line.replace("ATS_SCORE:", "").strip()
                    result["ats_score"] = int("".join(filter(str.isdigit, score_text)))
                except ValueError:
                    pass

            elif line.startswith("STRENGTHS:"):
                current_section = "strengths"
            elif line.startswith("WEAKNESSES:"):
                current_section = "weaknesses"
            elif line.startswith("SUGGESTIONS_TO_IMPROVE:"):
                current_section = "suggestions"
            elif line.startswith("KEY_MISSING_ELEMENTS:"):
                current_section = "missing_elements"

            elif line.startswith("- ") and current_section:
                item = line[2:].strip()
                if item:
                    result[current_section].append(item)

        return result

    def get_instance() -> "RAGEngine":
        """Get singleton instance."""
        return RAGEngine()
