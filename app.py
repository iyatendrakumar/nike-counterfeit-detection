import streamlit as st
import cv2
from inference import predict

st.set_page_config(
    page_title="Counterfeit Product Detection",
    layout="centered"
)

st.title("🛡️ Counterfeit Product Detection System")
st.subheader("Nike Shoes Label Verification")

st.markdown("---")

# Since we are only working with shoes now
domain = "shoes"

uploaded = st.file_uploader(
    "Upload Nike shoe label image",
    type=["jpg", "png", "jpeg"]
)

if uploaded is not None:
    # Save uploaded image
    with open("temp.jpg", "wb") as f:
        f.write(uploaded.getbuffer())

    st.image(
        "temp.jpg",
        caption="Uploaded Image",
        width=350
    )

    # Explicit verification button
    if st.button("🔍 Verify Authenticity"):
        with st.spinner("Analyzing image..."):
            label, confidence, roi = predict("temp.jpg", domain)

        st.markdown("### 🔎 Verification Result")

        st.image(
            roi,
            caption="Extracted Region of Interest (ROI)",
            width=300
        )

        if label == "Genuine":
            st.success(f"✅ **{label}**")
        else:
            st.error(f"❌ **{label}**")

        st.metric(
            label="Confidence Score",
            value=f"{confidence}%"
        )

        st.progress(confidence / 100)
