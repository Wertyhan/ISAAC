import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from isaac_api import __version__
from isaac_api.core.config import settings
from isaac_api.core.database import get_vector_store, close_vector_store
from isaac_api.models.schemas import (
    QueryRequest,
    RetrievalResponse,
    HealthResponse,
    ErrorResponse,
)
from isaac_api.services.retriever import RetrieverService, get_retriever_service

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    #Startup
    logger.info("Starting ISAAC API...")
    try:
        _ = get_vector_store()
        logger.info("Vector store connection established")
    except Exception as e:
        logger.error(f"Failed to connect to vector store on startup: {e}")

    yield

    #Shutdown
    logger.info("Shutting down ISAAC API...")
    close_vector_store()
    logger.info("Cleanup complete")


# Create FastAPI application
app = FastAPI(
    title="ISAAC Retrieval API",
    description=(
        "Semantic search API for the ISAAC multimodal RAG system. "
        "Retrieves contextual chunks from the vector store with image reference extraction."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ConnectionError)
async def connection_error_handler(request, exc: ConnectionError) -> JSONResponse:
    logger.error(f"Connection error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error="ServiceUnavailable",
            message="Unable to connect to the vector store",
            detail=str(exc),
        ).model_dump(),
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc: RuntimeError) -> JSONResponse:
    logger.error(f"Runtime error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An error occurred during search",
            detail=str(exc),
        ).model_dump(),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint",
)
async def health_check() -> HealthResponse:
    vector_store_connected = False
    try:
        _ = get_vector_store()
        vector_store_connected = True
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy" if vector_store_connected else "degraded",
        version=__version__,
        vector_store_connected=vector_store_connected,
    )


@app.post(
    "/search",
    response_model=RetrievalResponse,
    tags=["Search"],
    summary="Semantic search",
    description=(
        "Perform semantic search on the ISAAC vector store. "
        "Returns relevant document chunks and extracts image references."
    ),
    responses={
        200: {
            "description": "Successful search",
            "model": RetrievalResponse,
        },
        400: {
            "description": "Invalid request",
            "model": ErrorResponse,
        },
        503: {
            "description": "Vector store unavailable",
            "model": ErrorResponse,
        },
    },
)
async def search(
    request: QueryRequest,
    retriever: RetrieverService = Depends(get_retriever_service),
) -> RetrievalResponse:
    logger.info(f"Search request: query='{request.query[:50]}...', k={request.k}")
    logger.debug(f"filter_metadata type={type(request.filter_metadata)}, value={request.filter_metadata!r}")
    
    try:
        response = retriever.search(
            query=request.query,
            k=request.k,
            filter_metadata=request.filter_metadata,
        )
        return response
        
    except ConnectionError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector store connection failed: {e}",
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search operation failed: {e}",
        )


@app.get(
    "/",
    tags=["System"],
    summary="API root",
)
async def root() -> dict:
    return {
        "name": "ISAAC Retrieval API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "isaac_api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
