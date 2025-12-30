# Digital Garden

AI research knowledge graph + prototypes.

## Structure

```
vault/                  # ← Obsidian opens this
  concepts/             # Atomic ideas
  papers/               # Paper breakdowns  
  projects/             # Mini research
  inbox.md              # Raw idea dump
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

1. **Capture** → dump to `vault/inbox.md` (phone/Obsidian)
2. **Process** → create concept node, link to paper
3. **Prototype** → build code in `prototypes/experiments/`
4. **Publish** → ship to HF Spaces, post to LinkedIn, archive in `vault/published/`

## Linking Convention

From vault to code:
```markdown
Code: [app.py](../prototypes/experiments/lora-viz/app.py)
```

Between vault files:
```markdown
Related: [[concepts/lora]]
```
