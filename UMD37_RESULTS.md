# UMD 3.7 Validation Results

- 24/24 planetary adversarial integration checks passed.
- LoCoMo official raw-data evidence retrieval: MRR 0.4994, Any R@5 0.6736,
  Full R@5 0.5898, Any R@10 0.7381.
- LongMemEval official 500-question evidence retrieval: MRR 0.92131,
  Any R@5 0.9723, Full R@5 0.8532, Any R@10 0.9894.
- No answer text or evidence labels were used for ranking.
- Key rotation covered memory records, state, event chains, and planetary
  sidecars; the old master key was rejected afterward.
- Automatic grounded reflection accepted supported claims and rejected an
  adversarial unsupported claim.
- Four-hop typed entity reasoning returned all four source memories.
- ABAC, shared agent blocks, procedural memory, model-specific memory,
  retention deletion/redaction, incremental node writes, quorum placement, and
  restart recovery passed.

The public scores are retrieval metrics, not end-to-end LLM answer accuracy.

