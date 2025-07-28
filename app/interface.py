import gradio as gr
from PIL import Image


# 🌸 Flower Classification Function
def classify_flower(image: Image.Image) -> str:
    return "🌸 Hmm… that might be a daisy, but don't quote me."


# 🌈 Custom CSS for Gradio Interface
custom_css = """
body {
    background: #fff0f5;
}
.gradio-container {
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, .title, .description {
    text-align: center;
    color: #C71585;
}
"""

# 🌼 Gradio Interface Setup
demo = gr.Interface(
    fn=classify_flower,
    inputs=gr.Image(type="pil", label="🌸 Upload a Flower"),
    outputs=gr.Textbox(label="🌼 Prediction"),
    title="🌸 Deep Bloom: Flower Classifier",
    description="Upload a flower image and let the model guess its species.",
    examples=["app/sample_flower.jpg"],
    theme="soft",
    css=custom_css,
    live=True,
)

# 🚀 Launch the Gradio App
if __name__ == "__main__":
    demo.launch()
