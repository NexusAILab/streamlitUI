import streamlit as st
from datetime import datetime as dt
import requests
import json
import re
import io
import pandas as pd
import docx2txt
import PyPDF2
import streamlit.components.v1 as components
import time
import extra_streamlit_components as stx
import datetime
from streamlit_javascript import st_javascript

# Set page config first, before any other Streamlit commands
st.set_page_config(
    page_title="Nexus ChatGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "error_message" not in st.session_state:
    st.session_state.error_message = None

print("===============")
# Remove the @st.cache_resource decorator and modify the cookie management
cookie_manager = stx.CookieManager()
cookies = cookie_manager.get_all()
print("cookies", cookies)
session_id = cookie_manager.get(cookie="session_id")
print("session_id", session_id)
print("===============")

# Replace OpenAI configs with base URL and API key constants
BASE_URL = "https://api.nexusmind.tech/nexus/v3/chat/completions"
#BASE_URL = "https://helixmind.online/v1/chat/completions"
API_KEY = "xxx"

# Then add CSS customizations
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: transparent !important;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        margin: 0.2rem 0;
        background-color: transparent !important;
        border: 1px solid #1f77b4;
    }
    
    /* Chat history buttons */
    .stButton>button:not([data-baseweb="button"]) {
        background-color: transparent !important;
        border: 1px solid #1f77b4;
        text-align: left;
        padding: 0.5rem;
    }
    
    /* Button hover effect */
    .stButton>button:hover {
        background-color: rgba(31, 119, 180, 0.1) !important;
    }
    
    /* Title styling */
    h1 {
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    
    /* File uploader */
    .stFileUploader {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Add reCAPTCHA script
components.html(
    """
    <script src="https://www.google.com/recaptcha/api.js?render=6Len-6kqAAAAABOogjdRl_UTTtLJQa4BowBi4lup"></script>
    <script>
    function executeRecaptcha() {
        grecaptcha.ready(function() {
            grecaptcha.execute('6Len-6kqAAAAABOogjdRl_UTTtLJQa4BowBi4lup', {action: 'submit'})
            .then(function(token) {
                console.log('reCAPTCHA token obtained');
            });
        });
    }

    function getReCaptchaToken() {
        try {
            return grecaptcha.execute('6Len-6kqAAAAABOogjdRl_UTTtLJQa4BowBi4lup', {action: 'submit'});
        } catch (error) {
            console.error('reCAPTCHA error:', error);
            return Promise.resolve('');
        }
    }
    </script>
    """,
    height=0  # Reduced height since v3 is invisible
)

# System prompt
SYSTEM_PROMPT = """You are a helpful AI assistant. You aim to give accurate, informative responses while being direct and concise."""

# Sidebar
with st.sidebar:
    st.markdown("### Model Settings")
    # Add a container for better spacing
    with st.container():
        model = st.selectbox(
            "Select Model",
            ["llama-3.3-70b", "llama-3.2-90b", "llama-3.1-405B", "deepseek-chat", "gpt-4o"],
            help="Choose the AI model for your conversation"
        )

        system_prompt = st.text_area(
            "System Prompt",
            value=SYSTEM_PROMPT,
            height=150,
            help="Customize the AI's behavior with a system prompt"
        )
    
    st.markdown("---")  # Add a divider
    
    # Group chat management buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 New Chat", use_container_width=True):
            # Save current chat to history if it exists
            if st.session_state.messages:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("### 📚 Chat History")
    # Update chat history display
    for i, chat in enumerate(st.session_state.chat_history):
        with st.container():
            if st.button(
                f"🕒 {chat['timestamp']}\n{chat['preview']}",
                key=f"history_{i}",
                help="Click to load this chat"
            ):
                st.session_state.messages = chat['messages'].copy()
                st.rerun()

# Main chat interface
st.markdown("# 🤖 Nexus ChatGPT")

# Add error message display
if st.session_state.error_message:
    st.error(st.session_state.error_message)
    st.session_state.error_message = None  # Clear the error after displaying

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
    try:
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
    except Exception as e:
        st.session_state.error_message = f"Error processing content: {str(e)}"

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

# Update file uploader with better styling
uploaded_file = st.file_uploader(
    "📎 Upload a file (PDF, TXT, DOCX, CSV, etc.) - Max 10 MB",
    type=['txt', 'pdf', 'docx', 'csv'],
    help="Upload a file to analyze or discuss with the AI"
)

# Modify get_session_cookie function to be simpler
def get_session_cookie():
    try:
        return session_id or ''
    except Exception as e:
        st.warning(f"Session cookie error: {str(e)}")
        return ''

# Modify the chat input section (replace the existing if prompt block)
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
            
            def prepare_api_payload(messages, model, system_prompt):
                try:
                    recaptcha_token = st_javascript("""
                        await grecaptcha.execute('6Len-6kqAAAAABOogjdRl_UTTtLJQa4BowBi4lup', {action: 'submit'})
                    """)
                except Exception as e:
                    print(f"reCAPTCHA error: {str(e)}")
                    recaptcha_token = ''
                
                return {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        *messages
                    ],
                    "temperature": 0.7,
                    "stream": True,
                    "session": get_session_cookie(),
                    "token": recaptcha_token
                }

            # Update the API request section
            try:
                data = prepare_api_payload(
                    st.session_state.messages,
                    model,
                    system_prompt
                )
                
                response = requests.post(
                    BASE_URL,
                    headers=headers,
                    json=data,
                    stream=True,
                    timeout=30
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
                
            except requests.exceptions.Timeout:
                st.error("Request timed out. Please try again.")
                st.session_state.error_message = "Request timed out. Please try again."
                st.rerun()
            except requests.exceptions.RequestException as e:
                st.error(f"Network error: {str(e)}")
                st.session_state.error_message = f"Network error: {str(e)}"
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.error_message = f"Error: {str(e)}"
                st.rerun()

        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
            st.session_state.error_message = "Request timed out. Please try again."
            st.rerun()
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {str(e)}")
            st.session_state.error_message = f"Network error: {str(e)}"
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.error_message = f"Error: {str(e)}"
            st.rerun()

#python -m streamlit run chatui.py
