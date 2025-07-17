{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "60e14fb4-51e5-420b-9c6e-f5c02b0e91de",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-07-17 14:21:00.871 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-17 14:21:00.872 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-17 14:21:00.872 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-17 14:21:00.873 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-17 14:21:00.874 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-17 14:21:00.875 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-17 14:21:00.875 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-17 14:21:00.876 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-17 14:21:00.877 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-17 14:21:00.878 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-17 14:21:00.879 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-17 14:21:00.879 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-17 14:21:00.880 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import re\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "import joblib\n",
    "\n",
    "# Load saved model & vectorizer\n",
    "model = joblib.load(\"nb_model.pkl\")\n",
    "vectorizer = joblib.load(\"vectorizer.pkl\")\n",
    "\n",
    "# Text cleaning function\n",
    "def clean_text(text):\n",
    "    text = re.sub(r\"<.*?>\", \"\", text)\n",
    "    text = re.sub(r\"[^a-zA-Z]\", \" \", text)\n",
    "    return text.lower()\n",
    "\n",
    "# Streamlit UI\n",
    "st.title(\"🎬 IMDb Sentiment Analyzer\")\n",
    "review = st.text_area(\"Enter a movie review:\")\n",
    "\n",
    "if st.button(\"Predict Sentiment\"):\n",
    "    if review:\n",
    "        cleaned = clean_text(review)\n",
    "        vector = vectorizer.transform([cleaned])\n",
    "        pred = model.predict(vector)[0]\n",
    "        sentiment = \"😊 Positive\" if pred == 1 else \"😡 Negative\"\n",
    "        st.success(f\"Predicted Sentiment: **{sentiment}**\")\n",
    "    else:\n",
    "        st.warning(\"Please enter a review to analyze.\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
