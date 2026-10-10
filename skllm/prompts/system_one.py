"""Question wording for System One models, which take typed questions, not prompts.

The text travels as the question's ``state``; these instructions name what to
decide about it, and any data they need rides alongside, referred to by key in
backticks as the provider recommends.
"""

from __future__ import annotations

__all__ = ["CATEGORY_QUESTION", "MEMBERSHIP_QUESTION", "RUBRIC_QUESTION"]

CATEGORY_QUESTION = "Which category does the text belong to?"

MEMBERSHIP_QUESTION = "Does the text belong to `category`?"

RUBRIC_QUESTION = "Which score does the text earn under `rubric`?"
