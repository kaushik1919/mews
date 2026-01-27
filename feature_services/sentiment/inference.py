"""
FinBERT inference module.

Provides sentiment inference for financial text using FinBERT.
Supports both real FinBERT inference and deterministic mock mode.

CRITICAL DESIGN DECISIONS:
1. Model is FROZEN - no fine-tuning, no online learning
2. Inference is DETERMINISTIC - same text → same output
3. Mock mode is clearly marked and uses deterministic hashing
4. Interface is stable for future swap-in of real FinBERT

From features.yaml:
- model: "finbert_frozen"
- source_fields: [headline, body]
"""

import hashlib
from dataclasses import dataclass

from .mapping import SentimentLabel, label_to_score


@dataclass
class SentimentResult:
    """
    Result of sentiment inference on a single article.

    Attributes:
        label: FinBERT classification (positive, neutral, negative)
        score: Numeric score derived from label
        confidence: Model confidence (0-1), or None for mock
        is_mock: Whether this result is from mock inference
    """

    label: SentimentLabel
    score: float
    confidence: float | None
    is_mock: bool

    def __post_init__(self):
        """Validate result."""
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be in [-1, 1], got {self.score}")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


class FinBERTInference:
    """
    FinBERT inference wrapper.

    Supports two modes:
    1. Real mode: Uses HuggingFace transformers with FinBERT weights
    2. Mock mode: Deterministic hash-based inference for testing

    The mock mode is designed to:
    - Be deterministic (same text → same output)
    - Have realistic distribution (mix of labels)
    - Be clearly marked in output

    Interface is stable for swapping mock → real without API changes.
    """

    def __init__(self, use_mock: bool = True):
        """
        Initialize FinBERT inference.

        Args:
            use_mock: If True, use deterministic mock inference.
                      If False, attempt to load real FinBERT model.
        """
        self._use_mock = use_mock
        self._model = None
        self._tokenizer = None

        if not use_mock:
            self._load_model()

    def _load_model(self) -> None:
        """
        Load real FinBERT model from HuggingFace.

        Raises:
            ImportError: If transformers is not installed
            RuntimeError: If model cannot be loaded
        """
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_name = "ProsusAI/finbert"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model.eval()  # Freeze for inference
        except ImportError as e:
            raise ImportError(
                "transformers package required for real FinBERT inference. "
                "Install with: pip install transformers torch"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load FinBERT model: {e}") from e

    def infer(self, text: str) -> SentimentResult:
        """
        Infer sentiment for a single text.

        Args:
            text: Text to analyze (headline + body combined)

        Returns:
            SentimentResult with label, score, confidence

        Note: For articles with body, caller should combine headline + body.
        """
        if self._use_mock:
            return self._mock_infer(text)
        return self._real_infer(text)

    def _real_infer(self, text: str) -> SentimentResult:
        """
        Real FinBERT inference.

        Uses HuggingFace transformers for inference.
        Model is frozen (eval mode, no gradients).
        """
        import torch

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Inference (no gradients)
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        # FinBERT labels: positive, negative, neutral (in that order)
        # Map to our standard order
        label_map = {0: "positive", 1: "negative", 2: "neutral"}
        predicted_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, predicted_idx].item()

        label = label_map[predicted_idx]
        score = label_to_score(label)

        return SentimentResult(
            label=label,
            score=score,
            confidence=confidence,
            is_mock=False,
        )

    def _mock_infer(self, text: str) -> SentimentResult:
        """
        Deterministic mock inference for testing.

        Uses MD5 hash of text to produce deterministic output.
        Distribution is calibrated to be realistic:
        - ~40% neutral
        - ~35% positive
        - ~25% negative

        This matches typical financial news distribution.
        """
        # Hash text for deterministic output
        text_hash = hashlib.md5(text.encode()).hexdigest()
        hash_int = int(text_hash[:8], 16)

        # Map to label based on hash value
        # Distribution: neutral 40%, positive 35%, negative 25%
        normalized = hash_int / 0xFFFFFFFF

        if normalized < 0.40:
            label: SentimentLabel = "neutral"
        elif normalized < 0.75:
            label = "positive"
        else:
            label = "negative"

        score = label_to_score(label)

        # Mock confidence based on hash (makes testing predictable)
        mock_confidence = 0.6 + (normalized * 0.35)  # Range: 0.6 - 0.95

        return SentimentResult(
            label=label,
            score=score,
            confidence=mock_confidence,
            is_mock=True,
        )

    def infer_batch(self, texts: list[str]) -> list[SentimentResult]:
        """
        Infer sentiment for multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentResult objects

        Note: In mock mode, this is equivalent to calling infer() repeatedly.
        In real mode, batch processing would be more efficient.
        """
        # For now, simple sequential processing
        # Real FinBERT could batch for efficiency
        return [self.infer(text) for text in texts]

    @property
    def is_mock(self) -> bool:
        """Check if using mock inference."""
        return self._use_mock
