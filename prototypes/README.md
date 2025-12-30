# Prototypes

Code artifacts. Edit with VS Code / Claude Code.

## Structure

```
templates/          # Reusable scaffolds
experiments/        # Each folder → HF Space
  lora-layer-selection/
    app.py
    requirements.txt
    README.md       # HF Spaces metadata
```

## New Experiment

1. Copy template: `cp -r templates/gradio experiments/your-experiment`
2. Edit `app.py`
3. Deploy to HF Spaces

## Deploy

```bash
# One-time: create space
huggingface-cli repo create your-experiment --type space --sdk gradio

# Push
cd experiments/your-experiment
git init
git remote add space https://huggingface.co/spaces/YOUR_USER/your-experiment
git add . && git commit -m "init"
git push space main
```

Or use GitHub Actions for auto-deploy.
