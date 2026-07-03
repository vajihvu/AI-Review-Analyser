# Mock torch to prevent Hugging Face transformers from raising NameError when PyTorch is not installed
import builtins
class DummyTorch:
    pass
builtins.torch = DummyTorch()

from .preprocess import preprocess
from .sentiment import analyze_sentiment
from .topics import topic_modeling
from .summarize import summarize_reviews
