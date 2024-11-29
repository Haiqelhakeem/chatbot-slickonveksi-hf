import streamlit as st  # Import the Streamlit library
from transformers import AutoModelForCausalLM, AutoTokenizer  # Import Hugging Face tools
import time  # Import the time library

# Load Hugging Face model and tokenizer
def load_huggingface_model():
    model_name = "openai-community/gpt2"  # Replace with your desired Hugging Face model
    if 'hf_model' not in st.session_state:
        st.session_state.hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    if 'hf_tokenizer' not in st.session_state:
        st.session_state.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to generate a response using the Hugging Face model
def generate_response_with_hf(user_input):
    tokenizer = st.session_state.hf_tokenizer
    model = st.session_state.hf_model

    inputs = tokenizer.encode(user_input, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Load the model and tokenizer when the app starts
load_huggingface_model()

# Streamlit UI
st.title("Slickonveksi Chatbot")

# Initialize the chat history in the session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi there! How can I assist you today?"}]

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = generate_response_with_hf(prompt)  # Get response from Hugging Face model

        # Simulate "typing" effect
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")

        # Display full response and save to session state
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
