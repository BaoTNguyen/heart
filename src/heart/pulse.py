"""Old name of heart.observe (A6). plexus still imports it; drop once it
imports heart.observe."""
import sys

from heart import observe

sys.modules[__name__] = observe
