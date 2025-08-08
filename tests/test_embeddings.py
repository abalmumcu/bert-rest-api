import os
import sys
from unittest.mock import patch

import pytest
import torch

# Ensure the application package is importable when tests are executed from the
# "tests" directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.model import BERTModel


class DummyTokenizer:
    def batch_encode_plus(
        self,
        texts,
        add_special_tokens=True,
        padding='longest',
        truncation=True,
        return_tensors='pt',
    ):
        token_lists = [t.split() for t in texts]
        seq_lengths = [len(tokens) + 2 for tokens in token_lists]
        max_len = max(seq_lengths)

        input_ids = []
        attention_masks = []
        for seq_len in seq_lengths:
            ids = list(range(seq_len)) + [0] * (max_len - seq_len)
            mask = [1] * seq_len + [0] * (max_len - seq_len)
            input_ids.append(ids)
            attention_masks.append(mask)

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_masks),
        }

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        return list(range(len(tokens)))


class DummyModel:
    def eval(self):
        pass

    def __call__(self, input_ids, attention_mask=None):
        batch, seq_len = input_ids.shape
        hidden = 8
        return type('obj', (), {'last_hidden_state': torch.zeros(batch, seq_len, hidden)})


@pytest.fixture
def model():
    # Patch the expensive task-specific pipelines and base model/tokenizer so
    # tests do not require external downloads.
    with patch('app.model.pipeline') as mock_pipeline:
        mock_pipeline.return_value = lambda x: x

        def dummy_ensure(self):
            if self.model is None or self.tokenizer is None:
                self.tokenizer = DummyTokenizer()
                self.model = DummyModel()

        with patch.object(BERTModel, '_ensure_base_model', dummy_ensure):
            yield BERTModel()


def test_get_embeddings_single_input_shape(model):
    embeddings = model.get_embeddings(["Hello world"])
    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    seq_len = len(embeddings[0])
    hidden_size = len(embeddings[0][0])
    assert seq_len > 0
    assert hidden_size > 0


def test_get_embeddings_multi_input_shape(model):
    embeddings = model.get_embeddings(["Hello world", "Another sentence"])
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    seq_len0 = len(embeddings[0])
    seq_len1 = len(embeddings[1])
    hidden_size0 = len(embeddings[0][0])
    hidden_size1 = len(embeddings[1][0])
    assert seq_len0 == seq_len1
    assert hidden_size0 == hidden_size1
    assert seq_len0 > 0
    assert hidden_size0 > 0
