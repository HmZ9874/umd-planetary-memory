"""Small ONNX cross-encoder used as UMD 3.18 two-body interaction energy."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


class OnnxCrossEncoder:
    def __init__(
        self,
        cache_dir: str | Path,
        model_id: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        model_file: str = "onnx/model_quantized.onnx",
        max_length: int = 256,
        batch_size: int = 64,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.model_id = model_id
        self.model_file = model_file
        self.max_length = max_length
        self.batch_size = batch_size
        self._session = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._session is not None:
            return
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model = hf_hub_download(
            self.model_id, self.model_file, cache_dir=str(self.cache_dir),
        )
        tokenizer_path = hf_hub_download(
            self.model_id, "tokenizer.json", cache_dir=str(self.cache_dir),
        )
        tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer.enable_truncation(max_length=self.max_length)
        self._tokenizer = tokenizer
        self._session = ort.InferenceSession(
            model, providers=["CPUExecutionProvider"],
            sess_options=self._session_options(ort),
        )

    @staticmethod
    def _session_options(ort):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        return options

    def predict(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        self._load()
        assert self._tokenizer is not None and self._session is not None
        outputs: list[float] = []
        input_names = {value.name for value in self._session.get_inputs()}
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start:start + self.batch_size]
            encoded = [self._tokenizer.encode(query, passage) for passage in batch]
            width = max(len(value.ids) for value in encoded)
            ids = np.zeros((len(encoded), width), dtype=np.int64)
            masks = np.zeros_like(ids)
            types = np.zeros_like(ids)
            for row, value in enumerate(encoded):
                length = len(value.ids)
                ids[row, :length] = value.ids
                masks[row, :length] = value.attention_mask
                types[row, :length] = value.type_ids
            feed = {"input_ids": ids, "attention_mask": masks}
            if "token_type_ids" in input_names:
                feed["token_type_ids"] = types
            logits = self._session.run(None, feed)[0]
            outputs.extend(float(value) for value in np.asarray(logits).reshape(-1))
        return outputs

    def predict_multiwindow(
        self, query: str, passages: Sequence[str], *,
        window_chars: int = 900, max_windows: int = 3,
    ) -> list[float]:
        """Max-pool bounded head/middle/tail interactions per source.

        Character windows are deliberately query-independent. They prevent a
        single tokenizer head truncation from assigning all interaction mass
        to the beginning of a long memory while keeping inference bounded.
        """
        if not passages:
            return []
        expanded: list[str] = []
        owners: list[int] = []
        width = max(256, int(window_chars))
        for owner, passage in enumerate(passages):
            if len(passage) <= width:
                windows = [passage]
            else:
                windows = [passage[:width], passage[-width:]]
                if max_windows >= 3 and len(passage) > 2 * width:
                    middle = max(0, (len(passage) - width) // 2)
                    windows.insert(1, passage[middle:middle + width])
                windows = windows[:max(1, max_windows)]
            expanded.extend(windows)
            owners.extend([owner] * len(windows))
        values = self.predict(query, expanded)
        pooled = [float("-inf")] * len(passages)
        for owner, value in zip(owners, values):
            pooled[owner] = max(pooled[owner], float(value))
        return [value if value != float("-inf") else 0.0 for value in pooled]
