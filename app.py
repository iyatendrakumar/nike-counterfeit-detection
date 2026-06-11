import streamlit as st
from inference import predict

st.set_page_config(
    page_title="Counterfeit Product Detection",
    layout="centered"
)

st.title("🛡️ Counterfeit Product Detection System")
st.subheader("Nike Shoes Label Verification")

st.markdown("---")

domain = "shoes"

uploaded = st.file_uploader(
    "Upload Nike shoe label image",
    type=["jpg", "png", "jpeg"]
)

if uploaded is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded.getbuffer())

    st.image(
        "temp.jpg",
        caption="Uploaded Image",
        width=350
    )

    if st.button("🔍 Verify Authenticity"):
        with st.spinner("Analyzing image..."):
            label, confidence, roi = predict("temp.jpg", domain)

        st.markdown("### 🔎 Verification Result")

        # ── Handle Invalid Image ──────────────────────────────────────────
        if "Invalid" in label:
            st.warning(f"⚠️ **{label}**")
            st.info("Please upload a clear image of a **Nike shoe label or shoe**. Human photos, random objects, or blurry images cannot be verified.")

            if roi is not None:
                st.image(
                    roi,
                    caption="Extracted Region of Interest (ROI)",
                    width=300
                )

        # ── Genuine ───────────────────────────────────────────────────────
        elif label == "Genuine":
            st.image(
                roi,
                caption="Extracted Region of Interest (ROI)",
                width=300
            )
            st.success(f"✅ **{label} Nike Product**")
            st.metric(
                label="Confidence Score",
                value=f"{confidence}%"
            )
            st.progress(confidence / 100)

        # ── Counterfeit ───────────────────────────────────────────────────
        else:
            st.image(
                roi,
                caption="Extracted Region of Interest (ROI)",
                width=300
            )
            st.error(f"❌ **{label}**")
            st.metric(
                label="Confidence Score",
                value=f"{confidence}%"
            )
            st.progress(confidence / 100)