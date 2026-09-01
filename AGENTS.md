<!-- arteries:start -->
## Arteries Memory

At session start, run `PLUGIN_DATA=1 node /home/bao-tn/Coding/Projects/heart/.arteries/hooks/arteries-activate.cjs` and include the output as session context.
On each user prompt, pipe the prompt JSON to `PLUGIN_DATA=1 node /home/bao-tn/Coding/Projects/heart/.arteries/hooks/arteries-observe.cjs` and use any returned `additionalContext` to guide your response.
When compacting, run `ARTERIES_CLI=codex bash /home/bao-tn/Coding/Projects/heart/.arteries/hooks/hook-compact-packet.sh codex-compact` and preserve the returned continuity packet.
When an assistant response is available from a hook or transcript event, pipe that event to `ARTERIES_CLI=codex bash /home/bao-tn/Coding/Projects/heart/.arteries/hooks/hook-assistant-observe.sh codex-assistant` so Arteries can extract assistant-discovered project memory.

Arteries observes turns and assistant responses, builds memory, may surface retrieved prompts, and produces compact continuity packets as additional context.
<!-- arteries:end -->

## Rate card

Every dollar figure in this stack comes from `~/.config/heart/models.json`.

**If you see an unpriced model — from `heart models check`, or a plexus cost
panel reporting `gaps.unpriced` — follow `PRICING.md` before trusting any cost
figure that includes it.** Short version: read the rate off the vendor's own
pricing page for the exact model id, then

```bash
heart models set-price <model-id> --input N --output N --source <vendor-url>
```

`--source` is required. If the vendor page does not list the model, record
nothing and say so — a guessed rate that looks official is worse than a gap
that is labelled.
