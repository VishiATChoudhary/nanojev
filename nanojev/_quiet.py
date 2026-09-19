"""Quieten third-party startup noise for user-facing entry points.

Imported by the CLI and the examples, never by the library itself. A library
that silences warnings on import is a menace; a command-line tool that prints
two progress bars and a telemetry notice before its actual output is merely
annoying, and that part is worth fixing.
"""

from __future__ import annotations

import logging
import os
import warnings


def quiet() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    for name in ("transformers", "huggingface_hub", "torch"):
        logging.getLogger(name).setLevel(logging.ERROR)

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except Exception:
        # Never let cosmetics break the program.
        pass
