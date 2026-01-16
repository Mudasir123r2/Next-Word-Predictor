import os
import warnings

# Suppress TensorFlow warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

# Page configuration
st.set_page_config(
    page_title="Next Word Predictor | AI-Powered Text Completion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hero Section */
    .hero-container {
        background: white;
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.8);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.8rem;
        line-height: 1.2;
        display: inline-block;
        width: 100%;
    }
    
    .hero-subtitle {
        font-size: 1.4rem;
        color: #2d3748;
        text-align: center;
        font-weight: 600;
        margin-bottom: 1.2rem;
    }
    
    .hero-description {
        font-size: 1.05rem;
        color: #4a5568;
        text-align: center;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.7;
    }
    
    /* Card Styles */
    .card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border: 1px solid rgba(226, 232, 240, 0.8);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
    }
    
    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
        margin: 2rem 0;
    }
    
    .prediction-text {
        font-size: 2rem;
        font-weight: 600;
        text-align: center;
        line-height: 1.5;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-word {
        color: #ffd700;
        font-weight: 700;
        font-style: italic;
    }
    
    .confidence-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
    }
    
    /* Top Predictions List */
    .prediction-item {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .prediction-item:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border-left-color: #764ba2;
    }
    
    .prediction-rank {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
        min-width: 40px;
    }
    
    .prediction-word-item {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2d3748;
        flex-grow: 1;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .prediction-confidence {
        font-size: 1.1rem;
        font-weight: 600;
        color: #667eea;
        background: rgba(102, 126, 234, 0.1);
        padding: 0.3rem 1rem;
        border-radius: 50px;
    }
    
    /* Input Styles */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 1rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar Styles */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
        padding-top: 0 !important;
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 3rem;
    }
    
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 3rem !important;
    }
    
    .sidebar-content {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background-color: #fef3c7;
        border: 1px solid #fbbf24;
        border-radius: 12px;
        padding: 1rem;
        color: #92400e;
    }
    
    .stSuccess > div {
        color: #92400e;
    }
    
    .stError, .stWarning, .stInfo {
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: white;
        font-size: 0.9rem;
        margin-top: 3rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    </style>
""", unsafe_allow_html=True)

# Load model and tokenizer
@st.cache_resource
def load_resources():
    """Load the trained model and tokenizer"""
    import os
    import tensorflow as tf
    from tensorflow import keras
    
    model_path = 'next_word_model.h5'
    tokenizer_path = 'tokenizer.pickle'
    
    try:
        # Custom object scope to handle LSTM compatibility
        import h5py
        with h5py.File(model_path, 'r') as f:
            # Load model structure
            model_config = f.attrs.get('model_config')
            if model_config is None:
                raise ValueError("No model config found in the model file.")
        
        # Load model with compile=False to avoid loading optimizer state
        # This bypasses the time_major issue
        with keras.utils.custom_object_scope({}):
            model = keras.models.load_model(
                model_path, 
                compile=False,
                safe_mode=False  # Disable safe mode for legacy models
            )
        
        # Recompile the model with current TensorFlow version
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except FileNotFoundError as e:
        st.error(f"""
        ⚠️ **Model files not found in deployment.**
        
        This usually happens when Git LFS files aren't properly downloaded on Streamlit Cloud.
        
        **To fix this:**
        1. Verify Git LFS is installed: `git lfs install`
        2. Check tracked files: `git lfs ls-files`
        3. Force push LFS files: `git lfs push --all origin main`
        
        File not found: {str(e)}
        """)
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("""
        **Compatibility Issue Detected**
        
        The model was trained with an older TensorFlow version. 
        Attempting to rebuild the model architecture...
        """)
        return None, None

def predict_next_word(model, tokenizer, text, max_sequence_len):
    """Predict the next word based on input text"""
    try:
        tokenlist = tokenizer.texts_to_sequences([text])[0]
        tokenlist = pad_sequences([tokenlist], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(tokenlist, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word
        return None
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def get_top_predictions(model, tokenizer, text, max_sequence_len, top_k=5):
    """Get top k predictions with probabilities"""
    try:
        tokenlist = tokenizer.texts_to_sequences([text])[0]
        tokenlist = pad_sequences([tokenlist], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(tokenlist, verbose=0)[0]
        
        # Get top k indices
        top_indices = np.argsort(predicted)[-top_k:][::-1]
        
        # Create reverse word index
        reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}
        
        predictions = []
        for idx in top_indices:
            word = reverse_word_index.get(idx, None)
            if word:
                predictions.append((word, predicted[idx] * 100))
        
        return predictions
    except Exception as e:
        st.error(f"Error getting predictions: {str(e)}")
        return []

# Main app
def main():
    # Hero Section
    st.markdown("""
        <div class="hero-container fade-in">
            <h1 class="hero-title">🧠 Next Word Predictor</h1>
            <p class="hero-subtitle">AI-Powered Text Completion System</p>
            <p class="hero-description">
                Leveraging advanced LSTM neural networks trained on Shakespeare's Hamlet to predict 
                the next word in your sequence with remarkable accuracy. Experience the power of deep learning 
                in natural language processing.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load resources
    with st.spinner("🔄 Initializing AI model..."):
        model, tokenizer = load_resources()
    
    if model is None or tokenizer is None:
        st.error("❌ Failed to load model. Please ensure 'next_word_model.h5' and 'tokenizer.pickle' are in the same directory.")
        return
    
    # Calculate max sequence length
    max_sequence_len = model.input_shape[1] + 1
    
    # Sidebar information
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">📊 Model Statistics</div>', unsafe_allow_html=True)
        
        # Metrics in styled cards
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(tokenizer.word_index):,}</div>
                <div class="metric-label">Vocabulary Size</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{max_sequence_len}</div>
                <div class="metric-label">Max Sequence Length</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # About Section
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">ℹ️ About</div>', unsafe_allow_html=True)
        st.markdown("""
            <p style='color: #4a5568; line-height: 1.6; font-size: 0.95rem;'>
            This application uses a deep LSTM neural network architecture trained on Shakespeare's 
            Hamlet to predict the next word in a given sequence.
            </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # How to Use
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">📖 How to Use</div>', unsafe_allow_html=True)
        st.markdown("""
            <ol style='color: #4a5568; line-height: 1.8; font-size: 0.95rem;'>
                <li>Enter your text in the input field</li>
                <li>Click the "Predict Next Word" button</li>
                <li>View top 5 predictions with confidence scores</li>
                <li>Analyze the probability distribution</li>
            </ol>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tech Stack
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🛠️ Tech Stack</div>', unsafe_allow_html=True)
        st.markdown("""
            <ul style='color: #4a5568; line-height: 1.8; font-size: 0.95rem; list-style: none; padding-left: 0;'>
                <li>🔹 TensorFlow / Keras</li>
                <li>🔹 LSTM Architecture</li>
                <li>🔹 Streamlit UI</li>
                <li>🔹 NumPy & NLTK</li>
            </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area with two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input section in a card
        st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
        st.markdown("### 💬 Enter Your Text")
        input_text = st.text_input(
            "Type a phrase or sentence",
            value="To be, or not to be, that is",
            placeholder="Start typing...",
            help="Enter a phrase and the model will predict the next word",
            label_visibility="collapsed"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🚀 Predict Next Word", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Quick tips card
        st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
            <ul style='color: #4a5568; font-size: 0.9rem; line-height: 1.6;'>
                <li>Use complete phrases</li>
                <li>Try Shakespearean quotes</li>
                <li>Experiment with context</li>
            </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Predictions Section
    if predict_button and input_text:
        with st.spinner("🤖 Analyzing text and generating predictions..."):
            time.sleep(0.5)  # Brief pause for effect
            # Get top predictions
            predictions = get_top_predictions(model, tokenizer, input_text, max_sequence_len, top_k=5)
            
            if predictions:
                # Main prediction card
                st.markdown(f"""
                    <div class="prediction-card fade-in">
                        <div class="prediction-text">
                            "{input_text} <span class="prediction-word">{predictions[0][0]}</span>"
                        </div>
                        <div style="text-align: center;">
                            <span class="confidence-badge">Confidence: {predictions[0][1]:.2f}%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Progress bar for top prediction
                st.progress(predictions[0][1] / 100)
                
                # All predictions in styled cards
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
                st.markdown("### 📊 Top 5 Predictions")
                st.markdown("<br>", unsafe_allow_html=True)
                
                for i, (word, prob) in enumerate(predictions, 1):
                    # Color gradient for rank
                    rank_colors = ["#667eea", "#7c8eec", "#929eee", "#a8aef0", "#bebef2"]
                    st.markdown(f"""
                        <div class="prediction-item" style="border-left-color: {rank_colors[i-1]};">
                            <span class="prediction-rank">#{i}</span>
                            <span class="prediction-word-item">{word}</span>
                            <span class="prediction-confidence">{prob:.2f}%</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Additional insights
                st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
                st.markdown("### 📈 Prediction Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Top Confidence", f"{predictions[0][1]:.2f}%", 
                             delta=f"{predictions[0][1] - predictions[1][1]:.2f}% vs 2nd")
                
                with col2:
                    avg_confidence = sum(p[1] for p in predictions) / len(predictions)
                    st.metric("Avg Confidence", f"{avg_confidence:.2f}%")
                
                with col3:
                    st.metric("Total Predictions", len(predictions))
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Could not generate predictions. Please try different text.")
    
    elif predict_button:
        st.warning("⚠️ Please enter some text first.")
    
    # Footer
    st.markdown("""
        <div class="footer">
            <p style='font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;'>
                Built with ❤️ using Streamlit & TensorFlow
            </p>
            <p style='opacity: 0.8;'>
                Deep Learning | Natural Language Processing | LSTM Neural Networks
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
