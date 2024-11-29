import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer  # Import Hugging Face tools

# Function to load and initialize the Hugging Face model
def initialize_hf_model():
    model_name = "openai-community/gpt2"  # Hugging Face model name
    if "hf_model" not in st.session_state:
        st.session_state.hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    if "hf_tokenizer" not in st.session_state:
        st.session_state.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to generate a response using the Hugging Face model
def generate_hf_response(user_input):
    tokenizer = st.session_state.hf_tokenizer
    model = st.session_state.hf_model

    inputs = tokenizer.encode(user_input, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 1. Function to set up the page's layout and design elements
def set_design():
    col1, col2, col3 = st.columns([1, 2, 1])  # Creating a 3-column layout for the Streamlit app
    
    with col2:  # Display logo in the middle column
        st.image("sample_logo.png", use_column_width=True)

    st.markdown(  # Add title to the app
        "<p style='text-align: center; font-size: 30px;'><b>[Sample Generative AI Chatbot]</b></p>",
        unsafe_allow_html=True
    )

# 2. Function to initialize variables in the session state
def initialize_session_state():
    # Chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi there, how can I assist you today?"}
        ]
    # Hugging Face model and tokenizer
    initialize_hf_model()

# 3. Function to initialize sidebar UI elements
def sidebar():
    st.sidebar.markdown(
        """
        <h1 style='color: black; font-size: 24px;'>Chatbot Configuration</h1>
        """,
        unsafe_allow_html=True
    )

# 4. Function to clear the conversation
def clear_button():
    if st.sidebar.button("Clear Conversation", key="clear"):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi there, how can I assist you today?"}
        ]

# 5. Function to prepare a downloadable conversation log
def download_convo():
    if "messages" in st.session_state and len(st.session_state["messages"]) > 1:
        full_conversation = "\n".join(
            [
                f"\n{'-'*20}\n"
                f"Role: {msg['role']}\n"
                f"{'-'*20}\n"
                f"{msg['content']}\n"
                for msg in st.session_state["messages"]
            ]
        )
        return full_conversation
    else:
        st.warning("Not enough messages to download. Start a conversation first.")
        return ""

# 6. Function to add a download button for the conversation
def download_button():
    full_conversation = download_convo()
    st.sidebar.download_button(
        label="Download Conversation",
        data=full_conversation,
        file_name="conversation.txt",
        mime="text/plain"
    )
