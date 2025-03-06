import gradio as gr
from transformers import pipeline

# Load Hugging Face sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

# Function to analyze sentiment
def analyze_sentiment(text):
    result = sentiment_pipeline(text)
    label = result[0]["label"]
    confidence = result[0]["score"]
    return f"Sentiment: {label}\nConfidence: {confidence:.2f}"

# Create Gradio interface
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter text here..."),
    outputs="text",
    title="Sentiment Analysis",
    description="Enter a sentence to analyze its sentiment (Positive/Negative)."
)

# Launch Gradio app
if __name__ == "__main__":
    iface.launch()
