"""ISAAC - Intelligent System Architecture Advisor & Consultant."""

import logging
from pathlib import Path
from typing import Optional, List

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from isaac_generation import GenerationService
from isaac_generation.service import get_generation_service, StreamingResponse
from isaac_generation.interfaces import ResolvedImage

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

ANALYSIS_TRIGGERS = frozenset([
    "what is this", "what architecture", "analyze this", "what system",
    "similar to", "like this", "compare", "identify", "describe this",
    "explain this diagram", "що це", "яка архітектура", "проаналізуй", "схожі",
])

WELCOME_MESSAGE = (
    "## ISAAC System Initialized\n\n"
    "*\"Truth is ever to be found in simplicity, "
    "and not in the multiplicity and confusion of things.\"*\n\n"
    "---\n\n"
    "I am prepared to assist with your architectural inquiries.\n\n"
    "**Capabilities:**\n"
    "- Explain system architectures (Twitter, scaling, caching, etc.)\n"
    "- Show relevant architecture diagrams\n"
    "- Analyze architecture images you provide\n"
    "- Find similar architectures to your designs\n\n"
    "What matter shall we examine today?"
)


def _extract_user_image(message: cl.Message) -> Optional[Path]:
    """Extract image from message attachments."""
    if not message.elements:
        return None
    
    for element in message.elements:
        if isinstance(element, cl.Image) and hasattr(element, 'path') and element.path:
            return Path(element.path)
    
    return None


def _is_image_analysis_request(content: str) -> bool:
    """Determine if user is asking to analyze their uploaded image."""
    content_lower = content.lower()
    return any(trigger in content_lower for trigger in ANALYSIS_TRIGGERS)


async def _stream_response(
    response_msg: cl.Message,
    streaming_response: StreamingResponse,
) -> str:
    """Stream response tokens and return full text."""
    full_response = ""
    
    async for token in streaming_response.token_stream:
        full_response += token
        await response_msg.stream_token(token)
    
    return full_response


async def _attach_images(
    response_msg: cl.Message,
    images: List[ResolvedImage],
) -> None:
    """Attach resolved images to response message."""
    elements = []
    
    for idx, img in enumerate(images, 1):
        if not img.exists:
            logger.warning(f"Image not found: {img.path}")
            continue
        
        # Create Chainlit image element
        element = cl.Image(
            name=f"figure_{idx}",
            display="inline",
            path=str(img.path),
            size="large",
        )
        elements.append(element)
        
        # Add figure reference to response
        await response_msg.stream_token(
            f"\n\n**Figure {idx}:** *{img.description}*"
        )
        
        logger.debug(f"Attached image: {img.path.name}")
    
    response_msg.elements = elements


async def _add_citations(
    response_msg: cl.Message,
    sources: List[str],
    full_response: str,
) -> str:
    """Add source citations to response."""
    if not sources:
        return full_response
    
    unique_sources = list(dict.fromkeys(sources))
    citations = "\n\n---\n**References:** " + " | ".join(unique_sources)
    
    await response_msg.stream_token(citations)
    return full_response + citations


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session."""
    service = get_generation_service()
    
    cl.user_session.set("service", service)
    cl.user_session.set("chat_history", [])
    
    await cl.Message(content=WELCOME_MESSAGE, author="ISAAC").send()
    logger.info("ISAAC session started")


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages with optional image attachments."""
    service: Optional[GenerationService] = cl.user_session.get("service")
    chat_history: List[BaseMessage] = cl.user_session.get("chat_history", [])
    
    if not service:
        await cl.Message(content="Session error. Please refresh the page.", author="ISAAC").send()
        return
    
    response_msg = cl.Message(content="", author="ISAAC")
    await response_msg.send()
    
    try:
        user_image_path = _extract_user_image(message)
        
        if user_image_path and _is_image_analysis_request(message.content):
            streaming_response = await service.process_image_similarity(
                image_path=user_image_path,
                chat_history=chat_history,
            )
        else:
            streaming_response = await service.process_query(
                query=message.content,
                chat_history=chat_history,
                user_image_path=user_image_path,
            )
        
        full_response = await _stream_response(response_msg, streaming_response)
        await _attach_images(response_msg, streaming_response.context.images)
        full_response = await _add_citations(response_msg, streaming_response.context.sources, full_response)
        
        await response_msg.update()
        
        chat_history.append(HumanMessage(content=message.content))
        chat_history.append(AIMessage(content=full_response))
        cl.user_session.set("chat_history", chat_history)
        
        logger.info("Response generated successfully")
        
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        error_message = _build_error_message(e)
        await response_msg.stream_token(error_message)
        await response_msg.update()


def _build_error_message(error: Exception) -> str:
    """Build user-friendly error message."""
    error_str = str(error).lower()
    
    if "429" in str(error) or "resource_exhausted" in error_str or "rate" in error_str:
        return (
            "\n\nI find myself momentarily overtaxed — the celestial machinery "
            "requires a brief respite. Please wait a moment and try again.\n\n"
            "*Technical note: API rate limit reached. Wait 30-60 seconds before retrying.*"
        )
    
    if "api" in error_str or "key" in error_str:
        return (
            "\n\nAn issue with my connection to the ethereal realm has occurred. "
            "Please verify the API configuration."
        )
    
    return (
        f"\n\nI must confess an error has occurred in my calculations: "
        f"`{type(error).__name__}`\n\n"
        "Perhaps we might attempt this inquiry anew?"
    )


@cl.on_chat_end
async def on_chat_end():
    """Cleanup when chat session ends."""
    logger.info("ISAAC session ended")