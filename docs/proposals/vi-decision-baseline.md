# Proposal: Vietnamese decision model (reranker / general / content moderation)

Status: proposed, 2026-10-15. Product track, separate from the architecture research (`hyp/` branches): it applies an
established recipe to ship a model while the research continues.

## Baseline

- **Train:** [OneJev1.5-4B](https://github.com/OmniJev/OneJev) with its own recipe: full fine-tune, CE + Brier over the
  answer-letter probabilities, lr 5e-6, 1 epoch, vision tower frozen. Code, configs and data are public (Apache-2.0); the
  readout is the LM head's letter tokens, so `train/sft.py` runs unchanged on our data.
- **Reference only:** [d3-mini](https://huggingface.co/vllm-sr/d3-mini) (Apache-2.0 weights, recipe and data not
  released, custom 255-way readout). Run zero-shot on our test sets as the bar to beat.
- **Compute:** 2×H200. Their 4B config assumes 4 GPUs: run `torchrun --nproc-per-node 2` with FSDP and double gradient
  accumulation. [INFERENCE] full AdamW state for 4B (~64 GB) fits when sharded over 2×141 GB.

## Task format

Everything is a System One question, scored by one softmax over the options:

| Product | Question |
|---|---|
| Reranker | choice: "which passage answers the query?", 1 positive + k hard negatives (listwise) |
| Moderation | noul "Vi phạm?" + choice sạch / xúc phạm / thù ghét + score (severity) |
| General | choice / noul from NLI and reading-comprehension data |

## Public data

| Domain | Dataset | Size | License |
|---|---|---|---|
| Moderation | [ViHSD](https://huggingface.co/datasets/uitnlp/vihsd) | 33k comments, clean/offensive/hate | UIT, verify commercial use |
| Moderation | [UIT-ViCTSD](https://huggingface.co/datasets/tarudesu/ViCTSD) | 10k, toxicity + constructiveness | verify |
| Moderation | [ViHOS](https://huggingface.co/datasets/phusroyal/ViHOS) | 11k, with spans | verify |
| Reranker | [mMARCO-vi](https://huggingface.co/datasets/minhnguyent546/mmarco-vietnamese-split) | 39.8M triples; sample ~50k queries | **MS MARCO: non-commercial** |
| Reranker (legal) | [ViNLI-Zalo-supervised](https://huggingface.co/datasets/anti-ai/ViNLI-Zalo-supervised) | 33k query / positive / hard negative | verify |
| Reranker (legal) | [VLSP-Legal-Reranker-Train](https://huggingface.co/datasets/hoailebads/VLSP-Legal-Reranker-Train) | 10k, 1 positive + ≤23 hard negatives | Apache-2.0 |
| Reranker eval | [Zalo legal retrieval](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn) | 61k-document corpus | eval only |
| General | [OneJev-Data](https://huggingface.co/datasets/OmniJev/OneJev-Data), 30–50k subset | keeps general decision skill | check card |
| General (vi) | ViNLI, ViMMRC, UIT-VSFC | small | **research-only / non-commercial** |

Gap: no public Vietnamese LLM-safety (harmful prompt/response) data;
[PolyGuardMix](https://huggingface.co/datasets/ToxicityPrompts/PolyGuardMix) covers 17 languages, not Vietnamese.
Fill by LLM-translating a subset with human spot checks, or LLM-labelling against the company policy.

## Risks

- **Licensing is the main risk.** Most Vietnamese academic sets are research-only and MS MARCO is non-commercial. Default:
  restricted data for R&D and evaluation only; the shipped model trains on Apache, company and synthetic data. Otherwise get
  author permission or legal sign-off per source.
- Company moderation logs and policy will match production better than public social-media comments.

## Plan

1. **Week 1:** convert the sets to OneJev's format with a held-out test split per set; zero-shot d3-mini and
   OneJev1.5-4B on every product test set.
2. **Weeks 2–3:** fine-tune OneJev1.5-4B on the mix; ablate with vs without the restricted data to price the legal cost.
3. **Release gate:** beat d3-mini zero-shot on every product test set, with at most a ~2-point loss on the public
   Decision Index (no general regression).

## Open inputs

- Company moderation data and written policy: availability, size, categories.
- Reranking domain priority: legal, e-commerce or general.
