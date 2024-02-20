"""Main file for the app."""

# Import dependencies
import streamlit as st
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

# Streamlit app
def main():
    st.title("Anomaly Detection in Web Traffic")

    # Inputs
    region1 = st.slider("Region 1", 0, 800)
    region2 = st.slider("Region 2", 0, 800)
    region3 = st.slider("Region 3", 0, 800)
    region4 = st.slider("Region 4", 0, 800)
    region5 = st.slider("Region 5", 0, 800)
    region6 = st.slider("Region 6", 0, 800)

    # Prediction button
    if st.button("Predict"):
        prediction = predict(region1, region2, region3, region4, region5, region6)
        st.write("### Prediction:")
        st.write(prediction)

    # Clear button
    if st.button("Clear"):
        st.rerun()

    # Add a button to reload the model
    if st.button("Reload Model"):
        model = joblib.load("./localoutlierfactor.joblib")

if __name__ == "__main__":
    main()
