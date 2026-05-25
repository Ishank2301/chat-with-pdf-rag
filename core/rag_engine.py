"""
Query and response generation engine for 2026 RAG Agent.
Supports Q&A, document summarization, and intelligent routing.
"""
import logging
from typing import List, Dict, Optional
from langchain_ollama import OllamaLLM
import tenacity

logger = logging.getLogger(__name__)


class RAGEngine:
    """Main RAG engine for query processing and response generation."""
    
    _instance = None
    _llm = None
    
    def __new__(cls):
        """Singleton pattern for LLM connection."""
        if cls._instance is None:
            cls._instance = super(RAGEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model: str = "llama3"):
        """
        Initialize RAG engine.
        
        Args:
            model: Ollama model name (default: llama3)
        """
        if self._initialized:
            return
        
        self._llm = OllamaLLM(model=model)
        self._model_name = model
        self._initialized = True
        logger.info(f"✅ RAGEngine initialized with model: {model}")
    
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
        stop=tenacity.stop_after_attempt(3),
        reraise=True
    )
    def generate_response(
        self,
        question: str,
        relevant_chunks: List[str],
        max_length: int = 3
    ) -> str:
        """
        Generate response using relevant document chunks.
        
        Args:
            question: User query
            relevant_chunks: List of relevant document chunks
            max_length: Max sentences in response
            
        Returns:
            Generated answer
        """
        if not relevant_chunks:
            logger.warning("⚠️ No relevant chunks provided")
            return "I don't have relevant information to answer this question."
        
        context = "\n\n".join(relevant_chunks)
        
        prompt = f"""You are an intelligent assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question accurately.
If you don't know the answer based on the context, say that you don't know.
Keep your answer concise (maximum {max_length} sentences).

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            logger.info(f"🤖 Generating response for: {question[:50]}...")
            answer = self._llm.invoke(prompt)
            logger.info(f"✅ Response generated successfully")
            return answer.strip()
        
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            raise
    
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
        stop=tenacity.stop_after_attempt(3),
        reraise=True
    )
    def summarize_documents(
        self,
        documents: List[str],
        summary_length: str = "short"
    ) -> str:
        """
        Generate summary of documents.
        
        Args:
            documents: List of document texts
            summary_length: "short" (1-2 sentences) or "detailed" (1-2 paragraphs)
            
        Returns:
            Generated summary
        """
        if not documents:
            logger.warning("⚠️ No documents provided for summarization")
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
            logger.info(f"📝 Summarizing {len(documents)} document(s)...")
            summary = self._llm.invoke(prompt)
            logger.info(f"✅ Summary generated successfully")
            return summary.strip()
        
        except Exception as e:
            logger.error(f"❌ Summarization failed: {e}")
            raise
    
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
        stop=tenacity.stop_after_attempt(3),
        reraise=True
    )
    def analyze_resume_ats_score(self, resume_text: str) -> Dict[str, any]:
        """
        Analyze resume for ATS compatibility and score.
        
        Args:
            resume_text: Resume content
            
        Returns:
            Dictionary with ATS score and suggestions
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
            logger.info("📊 Analyzing resume for ATS score...")
            analysis = self._llm.invoke(prompt)
            logger.info("✅ ATS analysis completed")
            
            return self._parse_ats_analysis(analysis)
        
        except Exception as e:
            logger.error(f"❌ ATS analysis failed: {e}")
            raise
    
    def _parse_ats_analysis(self, analysis: str) -> Dict[str, any]:
        """Parse ATS analysis response into structured format."""
        result = {
            "raw_analysis": analysis,
            "ats_score": 0,
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
            "missing_elements": []
        }
        
        lines = analysis.split("\n")
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("ATS_SCORE:"):
                try:
                    score_text = line.replace("ATS_SCORE:", "").strip()
                    result["ats_score"] = int(''.join(filter(str.isdigit, score_text)))
                except:
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
