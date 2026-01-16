# 📝 Next Word Prediction - LSTM Neural Network

A deep learning application that predicts the next word in a sequence using an LSTM neural network trained on Shakespeare's Hamlet.

## 🌟 Features

- **LSTM Neural Network** with 150 and 100 units
- **Dropout Layers** for regularization (0.2)
- **Early Stopping** to prevent overfitting
- **Interactive Web Interface** built with Streamlit
- **Top-K Predictions** with confidence scores
- **Real-time Predictions**

## 🏗️ Model Architecture

```
Input Layer (Embedding) → LSTM(150) → Dropout(0.2) → LSTM(100) → Dense(Softmax)
```

## 📊 Dataset

- **Source:** Shakespeare's Hamlet from NLTK corpus
- **Preprocessing:** Tokenization, n-gram sequence generation, padding
- **Train/Test Split:** 80/20

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📁 Project Structure

```
Next Word Prediction/
├── app.py                    # Streamlit application
├── expermiments.ipynb        # Model training notebook
├── next_word_model.h5        # Trained model
├── tokenizer.pickle          # Tokenizer object
├── hamlet.txt                # Training data
├── requirements.txt          # Dependencies
├── DEPLOYMENT_GUIDE.md       # Deployment instructions
└── README.md                 # This file
```

## 🎯 How It Works

1. **Input:** User enters a text phrase
2. **Tokenization:** Text is converted to sequences
3. **Prediction:** LSTM model predicts probability distribution
4. **Output:** Top 5 most likely next words with confidence scores

## 💻 Usage Example

**Input:** `"To be, or not to be, that is"`

**Output:**
1. **the** - 45.23%
2. **a** - 23.45%
3. **question** - 12.34%
4. **not** - 8.76%
5. **all** - 5.43%

## 🛠️ Technologies Used

- **TensorFlow/Keras** - Deep learning framework
- **Streamlit** - Web application framework
- **NumPy** - Numerical computing
- **NLTK** - Natural language processing
- **Scikit-learn** - Train/test splitting

## 📈 Model Performance

- **Training:** 100 epochs (with early stopping)
- **Validation Monitoring:** val_loss
- **Patience:** 10 epochs
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy

## 🌐 Deployment

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed deployment instructions on:
- Streamlit Community Cloud (FREE)
- Heroku
- AWS EC2
- Docker

## 🔮 Future Enhancements

- [ ] Add multiple model options (GPT-2, BERT)
- [ ] Support for longer context windows
- [ ] Multi-word prediction
- [ ] Fine-tuning on custom datasets
- [ ] API endpoint for integrations
- [ ] Mobile-responsive design improvements

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

Created as a demonstration of LSTM-based next word prediction.

## 🙏 Acknowledgments

- Shakespeare's Hamlet dataset from NLTK
- Streamlit for the amazing framework
- TensorFlow team for the deep learning tools

---

**Happy Predicting! 🎉**
