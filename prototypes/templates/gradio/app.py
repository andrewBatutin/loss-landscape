"""
Minimal Gradio template for HF Spaces.

Usage:
1. Copy folder to experiments/your-experiment/
2. Edit TITLE, DESCRIPTION, core_function
3. Deploy to HF Spaces
"""

import gradio as gr

# === EDIT THESE ===

TITLE = "Experiment Name"

DESCRIPTION = """
**Hypothesis**: What we're testing

**Insight**: The one thing this demonstrates
"""

def core_function(input_text: str) -> str:
    """
    The essence. Keep this minimal.
    Everything else is scaffolding.
    """
    # YOUR CODE HERE
    result = input_text.upper()  # placeholder
    return result


# === TEMPLATE (usually don't edit) ===

demo = gr.Interface(
    fn=core_function,
    inputs=gr.Textbox(label="Input", lines=3),
    outputs=gr.Textbox(label="Output", lines=3),
    title=TITLE,
    description=DESCRIPTION,
    examples=[
        ["example input 1"],
        ["example input 2"],
    ],
)

if __name__ == "__main__":
    demo.launch()
