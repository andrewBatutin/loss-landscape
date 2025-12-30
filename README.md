# Digital Garden

AI research knowledge graph + prototypes.

## Structure

```
vault/                  # ← Obsidian opens this
  concepts/             # Atomic ideas
  papers/               # Paper breakdowns  
  projects/             # Mini research
  inbox/                # Raw idea dump
  published/            # Post archive

prototypes/             # ← VS Code / Claude Code
  templates/            # Reusable scaffolds
  experiments/          # Code → HF Spaces
```

## Setup

1. Clone repo
2. Obsidian → "Open folder as vault" → select `vault/`
3. Code editing in `prototypes/` with VS Code

## Workflow

1. **Capture** → dump to `vault/inbox.md` (phone/Obsidian). Can be done with AI automation
2. **Paper** -> capture essence of paper . Can be done with AI automation.
3. **Process** → create concept node, link to paper. Important to do it by hand
4. **Prototype** → build code in `prototypes/experiments/`. Important to do it by hand
5. **Publish** → ship to HF Spaces, post to LinkedIn, archive in `vault/published/`. Can be done with AI automation.

## Linking Convention

From vault to code:
```markdown
Code: [app.py](../prototypes/experiments/lora-viz/app.py)
```

Between vault files:
```markdown
Related: [[concepts/lora]]
```
