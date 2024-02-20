"""Main file for the app."""

# Import dependencies
import gradio as gr
import joblib
import numpy as np


# Predict function
def predict(
    region1: int,
    region2: int,
    region3: int,
    region4: int,
    region5: int,
    region6: int,
) -> str:
    """Predict if the traffic is normal or not."""
    model = joblib.load("./localoutlierfactor.joblib")
    yhat = model.predict(
        np.array([[region1, region2, region3, region4, region5, region6]]),
    )
    return "Weird Traffic" if yhat == -1 else "Normal Traffic"


# Building the Interface
app = gr.Interface(
    title="Anomaly Detection in Web Traffic",
    fn=predict,
    inputs=[
        gr.Slider(0, 800),
        gr.Slider(0, 800),
        gr.Slider(0, 800),
        gr.Slider(0, 800),
        gr.Slider(0, 800),
        gr.Slider(0, 800),
    ],
    outputs="text",
)

# Launch the app
app.launch()
