"""Neural detectors for the Phase 4 roster (4.3.1).

Importing this package imports torch. The benchmark constructs these lazily
inside its model factories so the harness core stays importable without torch.
"""

from scripts.models.generative.autoencoder import AutoencoderDetector
from scripts.models.generative.lstm_vae import LSTMVAEDetector
from scripts.models.generative.recurrent import RecurrentPredictorDetector
from scripts.models.generative.vae import VAEDetector

__all__ = ["AutoencoderDetector", "LSTMVAEDetector", "RecurrentPredictorDetector", "VAEDetector"]
