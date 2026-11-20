"""Analytical operation-count comparison, not a quality or latency benchmark."""

from __future__ import annotations

import time

def estimate_work(
    sequence_length: int = 64,
    hidden_size: int = 128,
    layers: int = 1,
    input_size: int | None = None,
) -> dict[str, int]:
    input_size = hidden_size if input_size is None else input_size
    values = (sequence_length, hidden_size, layers, input_size)
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in values):
        raise ValueError("sequence_length, hidden_size, layers and input_size must be positive integers")
    # MACs count three GRU gates and a Transformer block with a 2x feed-forward layer.
    rnn_macs = layers * sequence_length * 3 * (input_size * hidden_size + hidden_size * hidden_size)
    transformer_macs = layers * (
        sequence_length * 8 * hidden_size * hidden_size
        + 2 * sequence_length * sequence_length * hidden_size
    )
    return {
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "layers": layers,
        "rnn_macs": rnn_macs,
        "transformer_macs": transformer_macs,
        "rnn_sequential_steps": sequence_length * layers,
        "transformer_attention_pairs": sequence_length * sequence_length * layers,
    }


def run(
    seed: int = 42,
    sequence_lengths: tuple[int, ...] = (16, 32, 64, 128),
    hidden_size: int = 16,
    budget_macs: int = 24_000_000,
) -> dict:
    if not isinstance(budget_macs, int) or isinstance(budget_macs, bool) or budget_macs < 1:
        raise ValueError("budget_macs must be a positive integer")
    estimates = [estimate_work(length, hidden_size) for length in sequence_lengths]
    sequence_length = 16
    vocab_size = 8
    batch_size = 16
    if hidden_size % 4:
        raise ValueError("hidden_size must be divisible by the Transformer head count (4)")
    cost = estimate_work(sequence_length, hidden_size, input_size=vocab_size)
    rnn_step_macs = cost["rnn_macs"] * batch_size
    transformer_step_macs = cost["transformer_macs"] * batch_size
    rnn_steps = budget_macs // rnn_step_macs
    transformer_steps = budget_macs // transformer_step_macs
    if min(rnn_steps, transformer_steps) < 1:
        raise ValueError("compute budget is too small for one batch of both models")

    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("article 06 training requires PyTorch; install prototypes/requirements.txt") from exc

    old_threads = torch.get_num_threads()
    old_rng_state = torch.random.get_rng_state()
    torch.set_num_threads(1)
    try:
        data_gen = torch.Generator().manual_seed(seed)
        train_x = torch.randint(0, vocab_size, (512, sequence_length), generator=data_gen)
        test_x = torch.randint(0, vocab_size, (256, sequence_length), generator=data_gen)
        train_y = (train_x.eq(1).sum(dim=1) >= 2).long()
        test_y = (test_x.eq(1).sum(dim=1) >= 2).long()

        class RnnClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.GRU(vocab_size, hidden_size, batch_first=True)
                self.head = nn.Linear(hidden_size, 2)

            def forward(self, tokens):
                one_hot = F.one_hot(tokens, num_classes=vocab_size).float()
                _, state = self.rnn(one_hot)
                return self.head(state[-1])

        class TransformerClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.token = nn.Embedding(vocab_size, hidden_size)
                self.position = nn.Parameter(torch.zeros(1, sequence_length, hidden_size))
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size, nhead=4, dim_feedforward=hidden_size * 2,
                    dropout=0.0, batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
                self.head = nn.Linear(hidden_size, 2)

            def forward(self, tokens):
                encoded = self.encoder(self.token(tokens) + self.position)
                return self.head(encoded.mean(dim=1))

        def fit_and_score(model_type, steps: int, model_seed: int, estimated_step_macs: int) -> dict:
            torch.manual_seed(model_seed)
            model = model_type()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            sample_gen = torch.Generator().manual_seed(model_seed + 1000)
            started = time.perf_counter()
            last_loss = 0.0
            for _ in range(steps):
                indices = torch.randint(len(train_x), (batch_size,), generator=sample_gen)
                logits = model(train_x[indices])
                loss = F.cross_entropy(logits, train_y[indices])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach())
            seconds = time.perf_counter() - started
            model.eval()
            with torch.no_grad():
                predictions = model(test_x).argmax(dim=1)
                accuracy = float((predictions == test_y).float().mean())
                test_loss = float(F.cross_entropy(model(test_x), test_y))
            return {
                "steps": steps,
                "batch_size": batch_size,
                "estimated_macs_per_step": estimated_step_macs,
                "estimated_macs_used": estimated_step_macs * steps,
                "training_seconds": round(seconds, 6),
                "last_train_loss": round(last_loss, 6),
                "test_loss": round(test_loss, 6),
                "accuracy": round(accuracy, 6),
            }

        training = [
            {"model": "GRU", **fit_and_score(RnnClassifier, rnn_steps, seed + 1, rnn_step_macs)},
            {"model": "TransformerEncoder", **fit_and_score(TransformerClassifier, transformer_steps, seed + 2, transformer_step_macs)},
        ]
    finally:
        torch.random.set_rng_state(old_rng_state)
        torch.set_num_threads(old_threads)

    return {
        "evidence_type": "synthetic",
        "seed": seed,
        "compute_budget_macs": budget_macs,
        "sequence_task": {
            "sequence_length": sequence_length,
            "vocabulary_size": vocab_size,
            "train_sequences": len(train_x),
            "test_sequences": len(test_x),
            "label_rule": "positive when token 1 occurs at least twice",
        },
        "architecture_assumptions": "GRU with 3 gates and input_size=vocabulary_size; Transformer with 4 heads, 2x feed-forward width, dense attention",
        "estimates": estimates,
        "training": training,
        "torch_version": torch.__version__,
        "limitations": [
            "This is a synthetic binary sequence-classification task, not language modeling.",
            "MACs are architecture-level estimates; embeddings, heads, optimizer work, memory traffic, and backprop are omitted.",
            "The equal MAC budget is approximate and results from one seed do not establish which architecture is generally better.",
        ],
    }
