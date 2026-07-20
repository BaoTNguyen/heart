#!/usr/bin/env bash
# stack-smoke: one goal through plexus -> heart -> arteries, then assert the
# same episode_id in four places: plexus ledger, heart runs dir, spine events,
# arteries reward ingest. STACK_READINESS §1.3.
#
# Deterministic by default — a scripted agent plans and fixes, zero tokens.
# The capillaries gate only fires for hook-instrumented CLI agents, so here
# it's reported, not asserted.
#
# Usage: bash contrib/stack-smoke.sh
# Exit 0 = the factory's plumbing is connected end to end.
set -euo pipefail

HEART_SRC="$(cd "$(dirname "$0")/.." && pwd)/src"
PLEXUS_SRC="$HOME/Coding/Projects/plexus/src"
export PYTHONPATH="$PLEXUS_SRC:$HEART_SRC:${PYTHONPATH:-}"
export HEART_INGEST=off            # we ingest explicitly, at the end

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
export HEART_SPOOL_DIR="$WORK/spool"   # isolated spine for clean assertions
REPO="$WORK/toyrepo"
mkdir -p "$REPO"; cd "$REPO"

git init -q
git -c user.name=smoke -c user.email=s@s commit -q --allow-empty -m root
printf 'def add(a, b):\n    return a - b\n' > calc.py
cat > test_calc.py <<'EOF'
import unittest
from calc import add

class T(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

if __name__ == "__main__":
    unittest.main()
EOF

# scripted agent: emits a canned plan when asked to plan, applies the fix
# when asked to implement — the whole control plane runs, no model does
cat > agent.sh <<'EOF'
#!/usr/bin/env bash
case "$HEART_PROMPT" in
  *"JSON array"*)
    echo '[{"id": "fix-add", "title": "fix add", "spec": "make add return a+b",
            "acceptance": "python3 -m unittest -q test_calc"}]' ;;
  *) sed -i 's/a - b/a + b/' calc.py ;;
esac
EOF
chmod +x agent.sh

cat > plexus.toml <<'EOF'
[goal]
id = "stack-smoke"
text = "add() must return the sum of its arguments"
context = "single-file toy repo; tests in test_calc.py"

[ground_truth]
suite = "python3 -m unittest -q test_calc"

[agent]
cmd = "bash agent.sh"
timeout = 60
EOF
git add -A
git -c user.name=smoke -c user.email=s@s commit -qm "buggy base"

echo "== plexus plan/approve/run =="
python3 -m plexus.cli plan --root .
python3 -m plexus.cli approve --root .
python3 -m plexus.cli run --root .

echo "== four-place episode_id check =="
EP=$(python3 - <<'EOF'
import json, pathlib
recs = [json.loads(l) for l in pathlib.Path(".plexus/ledger.jsonl").read_text().splitlines()]
ids = [r["episode_id"] for r in recs if r.get("episode_id")]
assert ids, f"no episode_ids in plexus ledger: {recs}"
print(ids[-1])
EOF
)
echo "episode: $EP  (1/4: plexus ledger)"
test -f "runs/$EP/episode.json"                     && echo "2/4: heart runs dir"
grep -rq "$EP" "$HEART_SPOOL_DIR"                   && echo "3/4: spine events"
if command -v art >/dev/null; then
  # writes one tiny row to the real ledger (task stack-smoke-fix-add-a1) —
  # deliberate: 4/4 proves arteries is actually reachable, not stubbed
  HEART_INGEST= art ingest runs | tail -1           && echo "4/4: arteries ingest"
else
  echo "4/4 SKIPPED: art CLI not on PATH"
fi
grep -rq "prompt.gate" "$HEART_SPOOL_DIR" \
  && echo "capillaries gate: fired" \
  || echo "capillaries gate: absent (expected for the scripted agent)"

echo "== goal state =="
python3 -m plexus.cli status --root .
echo "stack-smoke: OK"
