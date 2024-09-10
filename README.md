# Webchat
### Webchat: Talk with Websites for Insightful Conversations 
## Overview:

Webchat is a chatbot that interacts with websites using LLM and RAG (Retrieval-Augmented Generation) techniques to provide structured responses based on scraped content. It leverages BeautifulSoup for scraping, FAISS for similarity search, and Google Generative AI embeddings for data analysis. The chatbot is built using Streamlit for a user-friendly interface and integrates libraries like LangChain and requests. Webchat efficiently answers queries based on website content, making it ideal for content analysis and interactionreal-world reliability.

Webchat is an intelligent chatbot designed to interact with websites by scraping content and providing concise, structured responses based on the extracted information. Leveraging natural language processing (NLP) and advanced embeddings from Google’s Generative AI, Webchat processes the content from user-provided URLs and engages in meaningful conversations about that content.

## Key Features:

Website Scraping: Webchat uses BeautifulSoup to scrape the textual content from provided website URLs. It processes the relevant data for conversational interactions.

Text Splitting & Embeddings: The content is split into manageable chunks using RecursiveCharacterTextSplitter and transformed into embeddings using Google Generative AI models.

FAISS Vector Store: The chatbot uses FAISS for fast, efficient similarity search to retrieve relevant content from the scraped data for answering user queries.

Generative Conversational Chain: Powered by LangChain, Webchat utilizes the Google Generative AI model to provide accurate, context-based responses drawn from the website data.

Dynamic Chat UI: Streamlit provides a sleek, responsive interface with customizable themes and a structured chat history that scrolls upwards.

## Use Case: 
Webchat is ideal for users or organizations looking to analyze and interact with website content efficiently. Whether you’re scraping news sites, blogs, or any content-rich platform, Webchat helps derive valuable insights through conversation.

![WhatsApp Image 2024-09-10 at 13 19 36_021f7d16](https://github.com/user-attachments/assets/85ac0aed-15c0-4bf0-af4b-17ef5bf713da)

