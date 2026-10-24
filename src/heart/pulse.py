"""Old name of heart.observe (A6). Kept on purpose: every plexus commit
before 2026-10-23 imports heart.pulse, and the replay set (plexus
evals/replay.json) re-runs those commits against today's heart."""
import sys

from heart import observe

sys.modules[__name__] = observe
