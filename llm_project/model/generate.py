"""
model/generate.py
=================
Text generation strategies for the GPT model.

All strategies share the same autoregressive loop:
  1. Feed the current context to the model → get logits for next token
  2. Apply a sampling strategy to the logits → pick the next token
  3. Append it to the context and repeat

Strategies implemented:
  greedy_decode    — always pick the highest-probability token (deterministic)
  temperature_sample — scale logits before softmax (controls randomness)
  top_k_sample     — keep only the top-k tokens, sample from them
  top_p_sample     — nucleus sampling: keep smallest set summing to probability p
  beam_search      — keep B candidate sequences, pick the highest-scoring one

Usage:
    from model.generate import GenerationConfig, generate
    cfg = GenerationConfig(max_new_tokens=100, temperature=0.8, top_k=40)
    output_ids = generate(model, prompt_ids, cfg, device="cpu")
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

@dataclass
class GenerationConfig:
    """
    All generation hyperparameters in one place.

    temperature:
      Controls the randomness of predictions.
      < 1.0 → more focused / repetitive (sharp distribution)
      > 1.0 → more diverse / creative (flat distribution)
      = 1.0 → use the model's raw probabilities unchanged

    top_k:
      Keep only the top-k most probable tokens; set the rest to −∞.
      Typical values: 10–50. Set to 0 to disable.

    top_p (nucleus sampling):
      Keep the smallest set of tokens whose cumulative probability ≥ top_p.
      Typical values: 0.9–0.95. Set to 1.0 to disable.

    Recommended combos:
      Creative text  : temperature=0.9, top_k=50, top_p=0.95
      Focused/factual: temperature=0.3, top_k=10,  top_p=1.0
      Greedy         : temperature=1.0, top_k=1,   top_p=1.0
    """
    max_new_tokens: int   = 100
    temperature:    float = 1.0
    top_k:          int   = 0      # 0 = disabled
    top_p:          float = 1.0    # 1.0 = disabled
    beam_width:     int   = 1      # 1 = no beam search
    repetition_penalty: float = 1.0  # > 1.0 penalises repeated tokens


# ─────────────────────────────────────────────
# LOGIT PROCESSORS
# ─────────────────────────────────────────────

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Divide logits by temperature before softmax.

    High temperature flattens the distribution (more random).
    Low temperature sharpens it (more greedy).
    temperature=1.0 leaves logits unchanged.
    """
    if temperature == 1.0:
        return logits
    return logits / max(temperature, 1e-8)


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Zero out all logits except the top-k.

    After this filter the model can only sample from the k most likely tokens,
    which prevents very low-probability (often nonsensical) tokens from appearing.
    """
    if k <= 0:
        return logits
    k = min(k, logits.size(-1))
    top_values, _ = torch.topk(logits, k)
    threshold     = top_values[:, -1:].expand_as(logits)
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling: keep the smallest vocabulary subset whose total
    probability mass is at least p; set everything else to −∞.

    This is more adaptive than top-k: on confident steps the nucleus
    is small (maybe 5 tokens); on uncertain steps it grows automatically.
    """
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens once their cumulative prob exceeds p
    # (shift by one so we keep the token that pushes over the threshold)
    remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[remove_mask] = float("-inf")

    # Scatter back to original ordering
    logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)
    return logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    """
    Divide the logit of any token that has already appeared by `penalty`.

    penalty > 1.0 discourages repetition.
    penalty = 1.0 has no effect.
    """
    if penalty == 1.0 or not generated_ids:
        return logits
    for token_id in set(generated_ids):
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty
    return logits


# ─────────────────────────────────────────────
# MAIN GENERATE FUNCTION
# ─────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,           # (1, T) long tensor
    cfg: GenerationConfig,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Autoregressive token generation.

    Applies temperature → top_k → top_p → repetition penalty → sample.

    Parameters
    ----------
    model      : trained GPT instance
    prompt_ids : (1, T) tensor of prompt token IDs
    cfg        : GenerationConfig with sampling parameters
    device     : 'cpu' or 'cuda'

    Returns
    -------
    (1, T + max_new_tokens) tensor of token IDs (includes the prompt)
    """
    model.eval()
    idx         = prompt_ids.to(device)
    context_len = model.config.context_len
    generated   = []

    for _ in range(cfg.max_new_tokens):
        # Crop to context window
        idx_cond = idx[:, -context_len:]

        # Forward pass — only need the last position's logits
        logits, _, _ = model(idx_cond)
        logits = logits[:, -1, :]    # (1, vocab_size)

        # Apply processors in order
        logits = apply_temperature(logits, cfg.temperature)
        logits = apply_repetition_penalty(logits, generated, cfg.repetition_penalty)
        logits = apply_top_k(logits, cfg.top_k)
        logits = apply_top_p(logits, cfg.top_p)

        # Sample or argmax
        probs    = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)   # (1, 1)

        generated.append(next_tok.item())
        idx = torch.cat([idx, next_tok], dim=1)

    return idx


# ─────────────────────────────────────────────
# BEAM SEARCH
# ─────────────────────────────────────────────

@torch.no_grad()
def beam_search(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,   # (1, T)
    max_new_tokens: int = 50,
    beam_width: int = 4,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Beam search: maintain beam_width candidate sequences in parallel.
    At each step, expand every candidate with its top-beam_width continuations
    and keep only the beam_width sequences with the highest cumulative log-prob.

    More expensive than sampling (beam_width × forward passes per step)
    but finds higher-probability sequences.

    Returns the single best sequence as a (1, T + max_new_tokens) tensor.
    """
    model.eval()
    context_len = model.config.context_len

    # Each beam: [token_ids_list, cumulative_log_prob]
    beams = [(prompt_ids[0].tolist(), 0.0)]

    for _ in range(max_new_tokens):
        candidates = []

        for seq, score in beams:
            # Build input tensor
            inp = torch.tensor([seq[-context_len:]], dtype=torch.long, device=device)
            logits, _, _ = model(inp)
            logits = logits[:, -1, :]                          # (1, V)
            log_probs = F.log_softmax(logits, dim=-1)[0]       # (V,)

            # Top-k extensions
            top_log_probs, top_ids = torch.topk(log_probs, beam_width)
            for log_p, tok_id in zip(top_log_probs.tolist(), top_ids.tolist()):
                candidates.append((seq + [tok_id], score + log_p))

        # Keep the best beam_width candidates
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    best_seq = beams[0][0]
    return torch.tensor([best_seq], dtype=torch.long)


# ─────────────────────────────────────────────
# TOKEN PROBABILITY HELPER  (for dashboard)
# ─────────────────────────────────────────────

@torch.no_grad()
def get_next_token_probs(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,   # (1, T)
    top_k: int = 10,
    temperature: float = 1.0,
    device: str = "cpu",
) -> list[tuple[int, float]]:
    """
    Return the top-k (token_id, probability) pairs for the next token.

    Used by the Dash dashboard to render the token probability bar chart.

    Returns
    -------
    List of (token_id, probability) tuples, sorted descending by probability.
    """
    model.eval()
    idx  = prompt_ids.to(device)[:, -model.config.context_len:]
    logits, _, _ = model(idx)
    logits = logits[:, -1, :] / max(temperature, 1e-8)   # (1, V)
    probs  = F.softmax(logits, dim=-1)[0]                 # (V,)

    top_probs, top_ids = torch.topk(probs, min(top_k, probs.size(0)))
    return [(tid.item(), prob.item()) for tid, prob in zip(top_ids, top_probs)]

