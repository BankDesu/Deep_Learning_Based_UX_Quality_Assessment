"""
Rule definitions for Layout Analysis + Rule-based UX Evaluation.

Each rule produces a score in [0, 1]:
    1.0 = fully satisfies the rule (good UX)
    0.0 = complete violation (bad UX)

Rules removed (require hierarchy input not available from screenshot alone):
    touch_target      — needs interactive_mask from accessibility tree
    heading_hierarchy — needs heading_levels from accessibility tree
"""

# Total number of rules
N_RULES = 7

# Human-readable rule names (index matches rule_scores vector)
RULE_NAMES: list[str] = [
    "contrast",        # 0 — color contrast between elements (WCAG AA proxy)
    "whitespace",      # 1 — adequate empty space (anti-clutter)
    "visual_balance",  # 2 — element centroid near horizontal center  ★ heatmap-enhanced
    "density",         # 3 — number of elements per area
    "alignment",       # 4 — edges align to common grid lines
    "cta_prominence",  # 5 — CTA/button is larger than avg element    ★ heatmap-enhanced
    "reading_flow",    # 6 — top-left-heavy layout (F/Z pattern)      ★ heatmap-enhanced
]

# Prior weights for aggregation (higher = more important for UX)
# Used as initialization; overridden by the learned RuleViolationHead weights.
RULE_WEIGHTS: list[float] = [
    1.5,   # contrast         — accessibility-critical
    1.0,   # whitespace
    0.8,   # visual_balance
    1.0,   # density
    1.2,   # alignment
    1.0,   # cta_prominence
    0.8,   # reading_flow
]

# WCAG AA contrast ratio threshold for normal text
WCAG_AA_CONTRAST: float = 4.5

# Ideal whitespace ratio range [lo, hi] → scores peak at midpoint
WHITESPACE_LO: float = 0.25
WHITESPACE_HI: float = 0.60

# Rule index constants for programmatic access
RULE_CONTRAST       = 0
RULE_WHITESPACE     = 1
RULE_VISUAL_BALANCE = 2
RULE_DENSITY        = 3
RULE_ALIGNMENT      = 4
RULE_CTA_PROMINENCE = 5
RULE_READING_FLOW   = 6
