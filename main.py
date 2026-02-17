import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.model import load_model, model_predict

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger.info("Loading model artifacts...")
    model, tokenizer, label_mapping = load_model()
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.label_mapping = label_mapping
    logger.info("Model loaded. label_mapping=%s", label_mapping is not None)
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Support Ticket Classification", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class Ticket(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)


class TicketResponse(BaseModel):
    ticket_category: int
    ticket_category_label: str | None = None


class ErrorResponse(BaseModel):
    detail: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        "method=%s path=%s status=%d latency_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/ticket_support_classification",
    response_model=TicketResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def classify_ticket(ticket: Ticket, request: Request) -> TicketResponse:
    try:
        category_id = int(
            model_predict(
                request.app.state.model,
                request.app.state.tokenizer,
                [ticket.message],
            )
        )
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

    label = None
    if request.app.state.label_mapping:
        label = request.app.state.label_mapping.get(category_id)

    return TicketResponse(ticket_category=category_id, ticket_category_label=label)
