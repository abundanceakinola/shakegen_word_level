import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import gdown

# Load and preprocess data
def load_and_preprocess_data(file_path, max_words=29897):  # Update max_words to match your vocabulary size
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts([raw_text])
    return raw_text, tokenizer

# Function to download files
def download_file_from_google_drive(file_id, output_file):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_file, quiet=False)

# Generate text function
def generate_text(model, start_text, tokenizer, seq_length, length=50, temperature=0.5):
    generated_text = start_text.split()
    for _ in range(length):
        encoded = tokenizer.texts_to_sequences([' '.join(generated_text[-seq_length:])])[0]
        encoded = pad_sequences([encoded], maxlen=seq_length, padding='pre')
        
        preds = model.predict(encoded, verbose=0)[0]
        next_index = sample_with_temperature(preds, temperature)
        next_word = ""
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                next_word = word
                break
        generated_text.append(next_word)
    return ' '.join(generated_text)

# Temperature sampling function
def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Download model and Shakespeare text if not already downloaded
model_file = "best_word_lstm_model.keras"
shakespeare_file = "shakespeare.txt"
model_file_id = "1KNVCkEOr5yRfwPHZAsiAMSOHo3nRoUbY"  # New model file ID
shakespeare_file_id = "1DIMeFhb40tE03Lay2gOXN40ytz1f3ptP"  # Unchanged

if not os.path.exists(model_file):
    with st.spinner('Downloading the model from Google Drive...'):
        download_file_from_google_drive(model_file_id, model_file)

if not os.path.exists(shakespeare_file):
    with st.spinner('Downloading Shakespeare text from Google Drive...'):
        download_file_from_google_drive(shakespeare_file_id, shakespeare_file)

# Download logo from Google Drive
logo_file = "ShakeGen_logo_no_bg.png"
logo_file_id = "1nscQAGhY6STIbV3lyB1oh8RhNxONP7Dw"  # Unchanged

if not os.path.exists(logo_file):
    with st.spinner('Downloading the logo from Google Drive...'):
        download_file_from_google_drive(logo_file_id, logo_file)

# Load the model
@st.cache_resource
def load_model_from_file():
    return load_model(model_file)

model = load_model_from_file()

# Load and preprocess Shakespeare text
raw_text, tokenizer = load_and_preprocess_data(shakespeare_file)

# Initialize chat history if not already done
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "user", "content": "User's seed text goes here"},
                                 {"role": "assistant", "content": "Generated text with seed text goes here"}]

# Streamlit UI title
st.title('ShakeGen: AI Sonnet Generator')
st.info('AI-powered Shakespearean Sonnet Generator using Word-Level LSTM')

# Add logo to sidebar
logo = "ShakeGen_logo_no_bg.png"
st.sidebar.image(logo, use_column_width=True)

# Sidebar content
st.sidebar.header("ShakeGen Information")
st.sidebar.write("""
ShakeGen is an AI-powered text generator designed to create poetry in the style of Shakespeare using a Word-Level LSTM model. 
You can experiment with seed text and adjust the temperature slider for creative variability.
- Temperature: Controls randomness. Lower values (e.g. 0.5) generate more predictable text, while higher values (e.g. 1.5) increase creativity.
- Example Seed Text: "Shall I compare thee to a summer's day"
""")

# Add temperature slider
temperature = st.sidebar.slider('Select Temperature', 0.1, 2.0, value=0.5)

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter a seed text"):
    # Display user message
    st.chat_message("user").markdown(prompt)
    
    # Generate text based on user input
    seq_length = 40  # This should match the sequence length used during training
    response = generate_text(model, prompt, tokenizer, seq_length, length=50, temperature=temperature)
    
    # Display generated text
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add both user and assistant responses to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})
