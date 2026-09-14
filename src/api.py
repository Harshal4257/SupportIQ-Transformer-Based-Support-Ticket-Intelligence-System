'''FastAPI service for support-ticket intent classification.'''

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Annotated, AsyncIterator

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, StringConstraints

from src.config import settings
from src.inference import TicketClassifier


TicketText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=10_000),
]


class PredictionRequest(BaseModel):
    text: TicketText


class PredictionResponse(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    low_confidence: bool


def model_path() -> Path:
    return Path(os.getenv('SUPPORT_MODEL_PATH', str(settings.transformer_output_dir)))


def confidence_threshold() -> float:
    value = float(
        os.getenv('SUPPORT_CONFIDENCE_THRESHOLD', str(settings.confidence_threshold))
    )
    if not 0.0 <= value <= 1.0:
        raise ValueError('SUPPORT_CONFIDENCE_THRESHOLD must be between 0 and 1.')
    return value


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.classifier = TicketClassifier(
        model_path=model_path(),
        confidence_threshold=confidence_threshold(),
    )
    yield


app = FastAPI(
    title='Support Ticket Intelligence API',
    description='Classify customer-support tickets into routing intents.',
    version='1.0.0',
    lifespan=lifespan,
)


@app.get('/')
def service_info() -> dict[str, str]:
    return {'name': app.title, 'docs': '/docs', 'health': '/health'}


@app.get('/health')
def health(request: Request) -> dict[str, object]:
    classifier = request.app.state.classifier
    return {
        'status': 'ready',
        'model_loaded': True,
        'model_path': str(model_path()),
        'device': str(classifier.device),
    }


@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest, http_request: Request) -> PredictionResponse:
    result = http_request.app.state.classifier.predict(request.text)
    return PredictionResponse(**result)

