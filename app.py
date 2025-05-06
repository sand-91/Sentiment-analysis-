import gradio as gr
from transformers import pipeline

# Load a sentiment analysis model
classifier = pipeline("sentiment-analysis")

# Define prediction function
def analyze_text(text):
    result = classifier(text)
    return f"Label: {result[0]['label']}, Confidence: {result[0]['score']:.4f}"

# Create Gradio interface
demo = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(label="Enter text"),
    outputs=gr.Text(label="Sentiment Result"),
    title="Sentiment Analysis Demo"
)

# Launch the app
demo.launch()
