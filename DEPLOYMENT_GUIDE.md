# 🚀 Next Word Prediction - Streamlit Deployment Guide

## 📋 Prerequisites

Before deploying, ensure you have:
- ✅ Trained model saved as `next_word_model.h5`
- ✅ Tokenizer saved as `tokenizer.pickle`
- ✅ `app.py` file created
- ✅ `requirements.txt` updated

## 🖥️ Local Deployment

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Streamlit App
```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### Step 3: Test the Application
1. Enter a phrase in the text input
2. Click "Predict Next Word"
3. View the top predictions with confidence scores

---

## ☁️ Cloud Deployment Options

### Option 1: Streamlit Community Cloud (Recommended - FREE)

#### Prerequisites:
- GitHub account
- Your code in a GitHub repository

#### Steps:
1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Next Word Prediction"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Important Files Structure:**
   ```
   your-repo/
   ├── app.py
   ├── requirements.txt
   ├── next_word_model.h5
   ├── tokenizer.pickle
   └── README.md
   ```

#### ⚠️ Important Notes:
- GitHub has a file size limit of 100MB
- If your model is larger than 100MB, use Git LFS:
  ```bash
  git lfs install
  git lfs track "*.h5"
  git add .gitattributes
  ```

---

### Option 2: Heroku Deployment

#### Steps:
1. **Install Heroku CLI:**
   - Download from [heroku.com](https://devcenter.heroku.com/articles/heroku-cli)

2. **Create additional files:**

   **Procfile:**
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

   **setup.sh:**
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   ```

3. **Deploy:**
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   heroku open
   ```

---

### Option 3: AWS EC2 Deployment

#### Steps:
1. **Launch EC2 Instance:**
   - Choose Ubuntu Server
   - Select t2.micro or larger
   - Configure security group to allow port 8501

2. **Connect and Setup:**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   
   # Install Python and dependencies
   sudo apt update
   sudo apt install python3-pip
   pip3 install -r requirements.txt
   
   # Run Streamlit
   streamlit run app.py --server.port=8501 --server.address=0.0.0.0
   ```

3. **Keep it running with tmux:**
   ```bash
   tmux new -s streamlit
   streamlit run app.py --server.port=8501 --server.address=0.0.0.0
   # Press Ctrl+B then D to detach
   ```

---

### Option 4: Docker Deployment

#### Create Dockerfile:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Build and Run:
```bash
docker build -t next-word-predictor .
docker run -p 8501:8501 next-word-predictor
```

---

## 🔧 Troubleshooting

### Issue: Model file too large for GitHub
**Solution:** Use Git LFS or host the model on cloud storage (AWS S3, Google Drive) and download it in the app:
```python
import gdown
url = 'https://drive.google.com/uc?id=YOUR_FILE_ID'
gdown.download(url, 'next_word_model.h5', quiet=False)
```

### Issue: Memory error on Streamlit Cloud
**Solution:** 
- Reduce model size by quantization
- Use a smaller vocabulary
- Request more resources from Streamlit support

### Issue: Slow predictions
**Solution:**
- Use `@st.cache_resource` for model loading (already implemented)
- Consider model optimization (TensorFlow Lite)

---

## 📊 Performance Tips

1. **Model Caching:** Already implemented with `@st.cache_resource`
2. **Reduce Model Size:** Use model quantization
3. **Optimize Predictions:** Batch predictions if needed
4. **Use CDN:** For static assets

---

## 🔐 Security Best Practices

1. **Environment Variables:** Store sensitive data in `.env` file
2. **Rate Limiting:** Implement rate limiting for API calls
3. **Input Validation:** Sanitize user inputs (already implemented)
4. **HTTPS:** Always use HTTPS in production

---

## 📱 Features Implemented

✅ Real-time next word prediction  
✅ Top 5 predictions with confidence scores  
✅ Interactive UI with Streamlit  
✅ Model caching for performance  
✅ Error handling  
✅ Responsive design  
✅ Progress indicators  

---

## 🎯 Next Steps

1. ✅ Run locally to test
2. ✅ Choose deployment platform
3. ✅ Push to GitHub
4. ✅ Deploy to cloud
5. ✅ Share your app URL!

---

## 📞 Support

If you encounter issues:
- Check Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)
- Streamlit community forum: [discuss.streamlit.io](https://discuss.streamlit.io)

---

**Good luck with your deployment! 🚀**
