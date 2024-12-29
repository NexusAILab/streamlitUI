import streamlit as st
from datetime import datetime
import requests
import json
import re
import io
import pandas as pd
import docx2txt
import PyPDF2

# Replace OpenAI configs with base URL and API key constants
BASE_URL = "https://helixmind.online/v1/chat/completions"
API_KEY = "helix-fdgumg4STWpGELyU66c_Cki6emi3wzpZCzssEhmDtl0"

# Page config
st.set_page_config(page_title="Nexus ChatGPT", layout="wide")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# System prompt
SYSTEM_PROMPT = """You are a helpful AI assistant. You aim to give accurate, informative responses while being direct and concise. For mathematical or technical topics, you provide clear explanations with examples when helpful."""

# Sidebar
with st.sidebar:
    # Model selection
    model = st.selectbox(
        "Select Model",
        ["deepseek-chat", "gpt-4o"]
    )

    # System prompt input
    system_prompt = st.text_area(
        "System Prompt",
        value=SYSTEM_PROMPT,
        height=150
    )

    # New chat button
    if st.button("New Chat"):
        # Save current chat to history if it exists
        if st.session_state.messages:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Get the first user message for preview, defaulting to "New Chat" if none exists
            preview = "New Chat"
            for msg in st.session_state.messages:
                if msg['role'] == 'user':
                    preview = msg['content'][:30] + "..."
                    break
            
            st.session_state.chat_history.append({
                "timestamp": timestamp,
                "preview": preview,
                "messages": st.session_state.messages.copy()
            })
        # Clear current messages
        st.session_state.messages = []
        st.rerun()

    # Chat history
    st.write("### Chat History")
    for i, chat in enumerate(st.session_state.chat_history):
        if st.button(f"{chat['timestamp']}: {chat['preview']}", key=f"history_{i}"):
            st.session_state.messages = chat['messages'].copy()
            st.rerun()

# Main chat interface
st.title("Nexus ChatGPT")

def clean_math_expression(expr):
    # Replace [...] line breaks with actual line breaks
    expr = re.sub(r'\]\s*\[', r' \\\\ ', expr)
    
    # Remove unnecessary spaces around operators and clean up notation
    expr = re.sub(r'\s*,\s*', ' ', expr)  # Replace commas with spaces
    expr = re.sub(r'\[\s*', '', expr)     # Remove opening brackets
    expr = re.sub(r'\s*\]', '', expr)     # Remove closing brackets
    expr = re.sub(r'\(\s*', '(', expr)    # Clean up parentheses
    expr = re.sub(r'\s*\)', ')', expr)
    
    # Fix power notation (x^2 -> x^{2} for complex exponents)
    expr = re.sub(r'\^(\w+)', r'^{\1}', expr)
    
    # Add display math environment for better formatting
    if not expr.strip().startswith('\\begin'):
        expr = f'\\begin{{align*}}\n{expr}\n\\end{{align*}}'
        
    return expr

def process_content(content):
    # Split into sentences (roughly)
    sentences = re.split(r'(?<=[.!?])\s+', content)
    
    for sentence in sentences:
        # Look for math expressions in each sentence
        parts = re.split(r'(\$\$.*?\$\$|\$.*?\$|\[.*?\]|\(.*?\))', sentence)
        
        formatted_parts = []
        for part in parts:
            if part:
                if part.startswith('[') and part.endswith(']'):
                    # Display math
                    cleaned_math = clean_math_expression(part)
                    st.latex(cleaned_math)
                elif part.startswith('(') and part.endswith(')'):
                    # Inline math
                    cleaned_math = clean_math_expression(part[1:-1])
                    st.latex(cleaned_math)
                elif part.strip():
                    # Regular text
                    st.write(part.strip())

def process_file_content(uploaded_file):
    """Process different types of uploaded files and return their content as text."""
    # Add file size check (10 MB limit)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB in bytes
    
    if uploaded_file.size > MAX_FILE_SIZE:
        return "File is too large. Please upload a file smaller than 10 MB."
    
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'txt':
            # Text files
            return uploaded_file.getvalue().decode('utf-8')
            
        elif file_type == 'pdf':
            # PDF files
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text() + '\n'
            return text
            
        elif file_type == 'docx':
            # Word documents
            return docx2txt.process(uploaded_file)
            
        elif file_type == 'csv':
            # CSV files
            df = pd.read_csv(uploaded_file)
            return df.to_string()
            
        else:
            return "Unsupported file type"
            
    except Exception as e:
        return f"Error processing file: {str(e)}"

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        process_content(message["content"])

# Add file uploader with size warning
uploaded_file = st.file_uploader("Upload a file (PDF, TXT, DOCX, CSV, etc.) - Max 10 MB", type=['txt', 'pdf', 'docx', 'csv'])

# Remove the separate text area and handle everything in chat input
if prompt := st.chat_input("What would you like to know?"):
    # If there's a file, combine file content with the prompt
    if uploaded_file is not None:
        file_content = process_file_content(uploaded_file)
        prompt = f"{prompt}\n\n{file_content}"
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            }
            
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                ],
                "temperature": 0.7,  # Fixed temperature value
                "stream": True
            }
            
            # Make streaming request
            response = requests.post(
                BASE_URL,
                headers=headers,
                json=data,
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    # Remove "data: " prefix and parse JSON
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        line = line[6:]  # Remove "data: " prefix
                    if line != "[DONE]":
                        try:
                            json_object = json.loads(line)
                            content = json_object['choices'][0]['delta'].get('content', '')
                            full_response += content
                            # Process math in streaming response
                            message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                        except json.JSONDecodeError:
                            continue
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

#pushkarsingh4343@gmail.com
#python -m streamlit run chatui.py
