import streamlit as st
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to scrape website content from URL
def scrape_website_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        # Extract text from the HTML content
        text = " ".join([p.get_text() for p in soup.find_all('p')])
        return text
    except Exception as e:
        st.error(f"Failed to scrape {url}: {e}")
        return ""

# Function to split text into chunks for embedding
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Conversational chain with Google Generative AI
def get_conversational_chain():
    prompt_template = """
    You are an intelligent assistant designed to analyze and provide precise information derived from website data. Your goal is to respond clearly, concisely, and accurately, based solely on the information scraped from the provided URLs.

    Guidelines:
    - Deliver well-organized, structured answers.
    - Base your responses strictly on the content scraped from the website.
    - Be concise, clear, and objective while addressing complex patterns in the question.
    - If the requested information matches previously known patterns or answers, provide a consistent response.
    - If the requested information is unclear, missing, or unavailable, kindly inform the user.

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    raw_response = response['output_text']
    print(f"Raw response: {raw_response}")

    formatted_response = (
        raw_response
        .replace(",", ", ")  # Add space after commas
        .replace(".", ".\n")  # Add a newline after periods
        .replace(" ", " ")
    )

    # Store question and answer in session state for chat history
    st.session_state.chat_history.insert(0, (user_question, formatted_response))  # Insert at the beginning for upward scrolling

def main():
    st.set_page_config(page_title="Website Assistant Chatbot", page_icon="🌐", layout="wide")

    # Custom CSS for color theme and chat layout
    st.markdown("""
        <style>
        .stApp::before {
            content: '';
            background: #f7f7f7
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            top: 0;
            left: 0;
            bottom: 0;
            right: 0;
            position: absolute;
            z-index: -1;
        }
        .main-container {
            background-color: #1D3557; /* Navy Blue */
            padding: 5px;
            border-radius: 5px;
            z-index: 1; /* Ensure it stays above the background */
            position: relative;
            color: #F1FAEE; /* Off-White */
        }
        .user-msg {
            background-color: #00B4D8; /* Turquoise */
            color: #F1FAEE; /* Off-White */
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            text-align: right;
        }
        .bot-msg {
            background-color: #6D6875; /* Slate Gray */
            color: #F1FAEE; /* Off-White */
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            text-align: left;
            display: flex;
            align-items: center;
        }
        .bot-icon {
            width: 30px;
            height: 30px;
            margin-right: 10px;
        }
        .chat-container {
            background-color: #F1FAEE; /* Off-White */
            padding: 20px;
            border-radius: 15px;
            max-height: 400px;
            overflow-y: auto;
            display: flex;
            flex-direction: column-reverse; /* Chat history scrolling upwards */
        }
        
        """, unsafe_allow_html=True)

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize session state for URL boxes
    if "url_boxes" not in st.session_state:
        st.session_state.url_boxes = 1

    # Layout: Split between URL input and chat
    left_col, right_col = st.columns([1, 3])

    # URL Input Area (Left)
    with left_col:
        st.markdown("<h3 style='color: #00B4D8;'>Enter The URL</h3>", unsafe_allow_html=True)
        
        # URL container with a border
        st.markdown("<div class='url-container'>", unsafe_allow_html=True)
        
        # Display URL input boxes
        urls = []
        for i in range(st.session_state.url_boxes):
            url = st.text_input(f"URL {i + 1}", key=f"url_{i + 1}")
            if url:
                urls.append(url)
        
        # Add more URL boxes
        if st.button("Add"):
            if st.session_state.url_boxes < 3:
                st.session_state.url_boxes += 1
        
        # Submit URLs for scraping
        if st.button("Submit"):
            full_text = ""
            for url in urls:
                scraped_text = scrape_website_content(url)
                full_text += scraped_text

            if full_text:
                text_chunks = get_text_chunks(full_text)
                get_vector_store(text_chunks)
                st.success("URLs scraped and processed successfully!")
        
        st.markdown("</div>", unsafe_allow_html=True)  # End of the URL container

    # Chat Area (Right)
    with right_col:
        st.markdown("<h1 style='text-align: center; color: #00B4D8;'>Website Assistant Chatbot 🌐</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #6D6875;'>Ask any question about the scraped content from provided URLs.</p>", unsafe_allow_html=True)
        user_question = st.text_input("Ask a Question from the Scraped Website Content")

        if user_question:
            user_input(user_question)

        # Display chat history (upward scrolling)
        if st.session_state.chat_history:
            st.write("## Chat History")
            with st.container():
                for i, (question, answer) in enumerate(st.session_state.chat_history, 1):
                    st.markdown(f"<div class='user-msg'><strong>You:</strong> {question}</div>", unsafe_allow_html=True)
                    st.markdown(f"""
                        <div class='bot-msg'>
                            <img src='https://cdn-icons-png.freepik.com/512/13086/13086996.png' class='bot-icon'/>
                            <div><strong>Bot:</strong> {answer}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Add a colorful background to the main container
    with st.container():
        st.markdown("<div class='main-container'></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
