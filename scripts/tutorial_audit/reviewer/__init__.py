"""Reviewer-driven validators for the tutorial audit pipeline.

This package gathers the rules that resist full automation and require
human or LLM judgment: narrative quality (E2.11), motivating question
(E4.31), tone (E4.35), hedging (E5.46), and Diataxis purity (E6.47). The
reviewer files ``reviewer_score.json`` into the per-tutorial dossier and
the CI gate requires every reviewer-only rule scored >= 3.

The actual rubric_judge module is filled in by a separate agent; this file
exists only so the package is importable from day one.
"""
