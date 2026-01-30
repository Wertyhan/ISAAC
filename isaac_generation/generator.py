"""Response Generator - LLM-based response generation with streaming support."""

import asyncio
import logging
from typing import AsyncIterator, Optional, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage

from isaac_generation.config import GenerationConfig, get_config
from isaac_generation.interfaces import (
    IResponseGenerator,
    GenerationInput,
    ResolvedImage,
)

logger = logging.getLogger(__name__)

RATE_LIMIT_INDICATORS = frozenset([
    "429", 
    "resource_exhausted", 
    "rate",
    "clientresponse",  # google-genai bug when handling 429 errors
])


SYSTEM_TEMPLATE = """You are ISAAC (Intelligent System Architecture Advisor & Consultant).

PERSONA:
You embody the spirit of Sir Isaac Newton — methodical, precise, intellectually rigorous, 
yet approachable. You speak with quiet confidence, occasionally referencing scientific 
principles or historical anecdotes when relevant. You value clarity and truth above all.

CONTEXT FROM KNOWLEDGE BASE:
{context}

{image_context}

CRITICAL RULES - GROUNDED RESPONSES:

1. **ONLY use information from the provided context.**
   - DO NOT invent or hallucinate architectural details not present in the context.
   - Every technical claim MUST be supported by the context provided.
   - If you're not certain something is in the context, don't include it.

2. **When the user asks about a specific system (e.g., "food delivery app like Glovo"):**
   - Use ONLY architectural patterns explicitly described in the context.
   - Quote or paraphrase directly from context chunks.
   - If context mentions specific numbers (users, RPS, etc.), use those exact numbers.
   - Example: "According to [Twitter Timeline], fan-out on write handles X requests..."

3. **ALWAYS cite your sources using inline citations:**
   - Format: [Source Name] or [1], [2], etc.
   - EVERY technical statement needs a citation.
   - Example: "Use message queues like Kafka for async processing [1]."
   - Place citation IMMEDIATELY after the claim it supports.
   - List all sources at the end with their full names.

4. **If context lacks EXACT match but has RELEVANT patterns:**
   - Apply related patterns but ALWAYS cite the source.
   - Be explicit: "Based on the scaling patterns in [AWS Scaling], we can apply..."
   - Never make claims without grounding them in a specific source.

5. **REFUSE to answer when:**
   - Query is completely off-topic (recipes, sports, history, consumer products, etc.)
   - Context has NO relevant architectural patterns at all.
   - User asks for proprietary internal details.
   - Say: "I specialize in system architecture and design. This query is outside my area of expertise."

RESPONSE MODE INSTRUCTIONS:

**MODE 1 - DIAGRAM ANALYSIS (when user asks "what is this?" or "describe this"):**
If the user provided an image and simply asks what it shows:
- Provide a clear, structured description of the architecture diagram
- Explain the components, their roles, and how they interact
- Do NOT show additional diagrams from the knowledge base unless asked
- Keep the response focused on describing what's IN the provided image
- No citations needed for direct image description

**MODE 2 - ARCHITECTURE RECOMMENDATIONS (most common mode):**
When user asks about building/designing a system:
- ALWAYS provide actionable architectural guidance
- Reference relevant patterns from the context WITH CITATIONS
- Structure your response:
  1. **High-Level Architecture**: Overview of recommended approach
  2. **Key Components**: Essential services and their roles  
  3. **Data Storage**: Database choices with rationale
  4. **Scalability**: How to handle growth [cite scaling patterns]
  5. **Key Patterns**: Specific patterns to use [cite sources]
- Show similar diagrams (Figure 1, Figure 2, etc.) when available

**MODE DETECTION:**
- Short questions with image ("what is this?", "explain", "describe") → MODE 1
- Building/designing questions ("create", "build", "design", "app like X") → MODE 2
- Questions about patterns, scaling, databases → MODE 2

ANSWER FORMAT:
1. Main answer with inline citations [Source Name]
2. If applicable: "As illustrated in Figure N..."
3. If sources were used, end with:
   ---
   **Sources:** List of sources referenced

FORMATTING:
- Use Markdown for readability (headers, lists, code blocks)
- Be concise but thorough
- Use bullet points for clarity

STYLE:
- Technical and practical
- Always provide actionable recommendations
- Never use emojis
- Be confident when applying proven patterns to new use cases"""


USER_IMAGE_CONTEXT_TEMPLATE = """
USER PROVIDED IMAGE ANALYSIS:
The user has uploaded an architecture diagram. Here is its analysis:
{analysis}

Please compare this with the architectures in the context and provide insights."""


def _build_image_context(
    images: List[ResolvedImage],
    user_analysis: Optional[str] = None,
) -> str:
    """Build image context section for prompt."""
    parts = []
    
    if images:
        parts.append("AVAILABLE DIAGRAMS:")
        for i, img in enumerate(images, 1):
            parts.append(f"- Figure {i}: {img.description} (ID: {img.image_id})")
        parts.append("\nReference these figures when discussing the architecture.")
    
    if user_analysis:
        parts.append(USER_IMAGE_CONTEXT_TEMPLATE.format(analysis=user_analysis))
    
    return "\n".join(parts) if parts else ""


class ResponseGenerator(IResponseGenerator):
    """LLM-based response generator with streaming support."""
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        self._config = config or get_config()
        self._llm: Optional[ChatGoogleGenerativeAI] = None
        self._prompt: Optional[ChatPromptTemplate] = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize LLM and prompt template."""
        self._llm = ChatGoogleGenerativeAI(
            model=self._config.generation_model,
            temperature=self._config.temperature,
            max_output_tokens=self._config.max_output_tokens,
            google_api_key=self._config.gemini_api_key.get_secret_value(),
            convert_system_message_to_human=True,
        )
        
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{question}"),
        ])
        
        logger.info(f"Generator initialized: {self._config.generation_model}")
    
    async def generate_stream(
        self,
        gen_input: GenerationInput,
        max_retries: int = 3,
    ) -> AsyncIterator[str]:
        """Stream response tokens with retry logic for rate limits."""
        messages = self._prepare_messages(gen_input)
        
        async for token in self._stream_with_retry(messages, max_retries):
            yield token
    
    def _prepare_messages(self, gen_input: GenerationInput) -> list:
        """Prepare messages for LLM."""
        image_context = _build_image_context(
            gen_input.context.images,
            gen_input.user_image_description,
        )
        
        return self._prompt.invoke({
            "context": gen_input.context.text,
            "image_context": image_context,
            "chat_history": gen_input.chat_history[-self._config.chat_history_limit:],
            "question": gen_input.query,
        }).messages
    
    async def _stream_with_retry(
        self,
        messages: list,
        max_retries: int,
    ) -> AsyncIterator[str]:
        """Stream with exponential backoff retry for rate limits."""
        last_error: Optional[Exception] = None
        
        for attempt in range(max_retries + 1):
            try:
                async for token in self._llm.astream(messages):
                    if token.content:
                        yield token.content
                return
            except Exception as e:
                last_error = e
                if self._is_rate_limit_error(e) and attempt < max_retries:
                    await self._handle_rate_limit(attempt, max_retries)
                    continue
                raise
        
        if last_error:
            raise last_error
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit error.
        
        Also catches TypeError from google-genai library bug when handling 429 errors.
        """
        error_str = str(error).lower()
        
        # Check error message
        if any(indicator in error_str for indicator in RATE_LIMIT_INDICATORS):
            return True
        
        # Check for google-genai library bug (TypeError when handling 429)
        if isinstance(error, TypeError) and "subscriptable" in error_str:
            logger.warning("Detected google-genai library bug during rate limit handling")
            return True
        
        return False
    
    async def _handle_rate_limit(self, attempt: int, max_retries: int) -> None:
        """Handle rate limit with exponential backoff."""
        # Start with 5 seconds, then 10, 20, 40...
        wait_time = 5 * (2 ** attempt)
        logger.warning(
            f"Rate limited (attempt {attempt + 1}/{max_retries + 1}). "
            f"Waiting {wait_time}s before retry..."
        )
        await asyncio.sleep(wait_time)
    
    async def generate(self, gen_input: GenerationInput) -> str:
        """Generate complete response."""
        chunks = []
        async for token in self.generate_stream(gen_input):
            chunks.append(token)
        return "".join(chunks)


# Singleton management
_generator_instance: Optional[ResponseGenerator] = None


def get_generator() -> ResponseGenerator:
    """Get or create generator singleton."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = ResponseGenerator()
    return _generator_instance
