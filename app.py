import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="🛡️",
    layout="centered"
)

MODEL_PATH = Path("models/spam_detector.pkl")
METRICS_PATH = Path("reports/metrics.json")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return None

model = load_model()
metrics = load_metrics()

st.title("🛡️ AI Spam Message Detector")
st.caption("NLP + TF-IDF + Machine Learning")

st.write(
    "Paste an SMS or email below. The model analyzes the text and predicts "
    "whether it is **spam** or **legitimate (ham)**."
)

examples = {
    "Choose an example": "",
    "Normal message": "Hey, are we still meeting at 3pm?",
    "Possible spam": "Congratulations! You have won a cash prize. Claim now!",
    "Delivery message": "Your package has arrived. Please collect it today."
}

choice = st.selectbox("Try an example", list(examples))
default_text = examples[choice]

message = st.text_area(
    "Message",
    value=default_text,
    height=170,
    placeholder="Type or paste a message here..."
)

if st.button("🔍 Analyze Message", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([message])[0]
        st.write("### Result")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([message])[0]
            classes = list(model.classes_)
            prob_map = dict(zip(classes, probs))
            confidence = prob_map[prediction]

            chart = pd.DataFrame({
                "Class": ["Not Spam", "Spam"],
                "Probability": [
                    prob_map.get("ham", 0),
                    prob_map.get("spam", 0)
                ]
            }).set_index("Class")
            st.bar_chart(chart, y="Probability")
        else:
            confidence = None

        if prediction == "spam":
            st.error(
                f"🚨 **SPAM**"
                + (f" — model confidence: **{confidence:.1%}**" if confidence is not None else "")
            )
            st.write("The message resembles patterns learned from spam examples.")
        else:
            st.success(
                f"✅ **NOT SPAM**"
                + (f" — model confidence: **{confidence:.1%}**" if confidence is not None else "")
            )
            st.write("The message resembles legitimate messages in the training data.")

st.divider()

if metrics:
    st.subheader("📊 Model performance")
    best = metrics["best_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{best['accuracy']:.1%}")
    c2.metric("Precision", f"{best['precision']:.1%}")
    c3.metric("Recall", f"{best['recall']:.1%}")
    c4.metric("F1", f"{best['f1']:.1%}")

    st.caption(
        f"Best model: {metrics['best_model']} | "
        f"Dataset: {metrics['dataset_rows']:,} messages"
    )

with st.expander("ℹ️ About this project"):
    st.write(
        "The project uses the UCI SMS Spam Collection, a public dataset of "
        "5,574 labeled SMS messages. Text is transformed with TF-IDF and "
        "classified using multiple machine-learning algorithms."
    )
    st.write("For educational purposes; predictions are not guaranteed to be correct.")

with st.expander("📚 Dataset source"):
    st.write("UCI SMS Spam Collection — CC BY 4.0.")
    st.write("https://archive.ics.uci.edu/dataset/228/sms+spam+collection")
