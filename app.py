import streamlit as st
import pandas as pd
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import json
import re
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from datetime import datetime
import time

# Set page config
st.set_page_config(
    page_title="NitiAI - Government Scheme Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .profile-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2a5298;
        margin: 1rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: black; 
    }
    
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #0d47a1;
    }
    
    .bot-message {
        background: #f1f8e9;
        border-left: 4px solid #4caf50;
        color: #1b5e20;
    }
    
    .tool-info {
        background: #fff3e0;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        color: #e65100;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Import your classes and functions
from utils.user_profile import UserProfile
from utils.tools import *
from utils.agent_setup import setup_agent

# Initialize session state
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = UserProfile()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None

if 'vectorstore_loaded' not in st.session_state:
    st.session_state.vectorstore_loaded = False

@st.cache_resource
def load_vectorstore():
    """Load vectorstore with caching"""
    try:
        persist_dir = "chroma_store"
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )
        
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
        )
        
        return vectorstore, retriever
    except Exception as e:
        st.error(f"Error loading vectorstore: {e}")
        return None, None

@st.cache_resource
def initialize_agent():
    """Initialize agent with caching"""
    try:
        vectorstore, retriever = load_vectorstore()
        if vectorstore is None:
            return None
        
        st.session_state.vectorstore_loaded = True
            
        # Setup API key (you should use st.secrets for production)
        groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
        if not groq_api_key:
            st.error("GROQ_API_KEY not found. Please set it in secrets or environment variables.")
            return None
            
        os.environ["GROQ_API_KEY"] = groq_api_key
        
        agent_executor = setup_agent(vectorstore, retriever)
        return agent_executor
    except Exception as e:
        st.error(f"Error initializing agent: {e}")
        return None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ NitiAI </h1>
        <p>Your AI Assistant for Indian Government Schemes</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize agent if not done
    if st.session_state.agent_executor is None:
        with st.spinner("🔄 Loading AI model and database..."):
            st.session_state.agent_executor = initialize_agent()
    
    # Sidebar for profile setup
    with st.sidebar:
        st.header("👤 User Profile")
        
        # Profile setup form
        with st.form("profile_form"):
            st.subheader("Setup Your Profile")
            st.info("Complete profile helps find relevant schemes!")
            
            age = st.number_input("Age", min_value=1, max_value=100, 
                                value=int(st.session_state.user_profile.age) if st.session_state.user_profile.age else 25)
            
            income = st.selectbox("Annual Income", [
                "Below 1 Lakh", "1-2 Lakhs", "2-5 Lakhs", 
                "5-10 Lakhs", "10-20 Lakhs", "Above 20 Lakhs"
            ], index=1)
            
            location = st.text_input("State/City", 
                                   value=st.session_state.user_profile.location if st.session_state.user_profile.location else "")
            
            category = st.selectbox("Category", ["General", "SC", "ST", "OBC"], index=0)
            
            occupation = st.text_input("Occupation (Optional)", 
                                     value=st.session_state.user_profile.occupation if st.session_state.user_profile.occupation else "")
            
            education = st.text_input("Education (Optional)", 
                                    value=st.session_state.user_profile.education if st.session_state.user_profile.education else "")
            
            if st.form_submit_button("✅ Update Profile", type="primary"):
                st.session_state.user_profile.set_profile(
                    age=str(age),
                    income=income,
                    location=location,
                    category=category,
                    occupation=occupation,
                    education=education
                )
                st.success("Profile updated successfully!")
                st.rerun()
        
        # Display current profile
        if st.session_state.user_profile.is_complete:
            st.markdown(f"""
            <div class="profile-card">
                <h4>📊 Current Profile</h4>
                <p><strong>Age:</strong> {st.session_state.user_profile.age}</p>
                <p><strong>Income:</strong> {st.session_state.user_profile.income}</p>
                <p><strong>Location:</strong> {st.session_state.user_profile.location}</p>
                <p><strong>Category:</strong> {st.session_state.user_profile.category}</p>
                {f"<p><strong>Occupation:</strong> {st.session_state.user_profile.occupation}</p>" if st.session_state.user_profile.occupation else ""}
            </div>
            """, unsafe_allow_html=True)
        
        # Quick action buttons
        st.subheader("🚀 Quick Actions")
        
        quick_queries = [
            "What schemes can I apply for?",
            "Show me employment schemes",
            "Find education scholarships",
            "Business loan schemes",
            "Housing assistance schemes"
        ]
        
        for query in quick_queries:
            if st.button(f"💡 {query}", key=f"quick_{query}", use_container_width=True):
                if st.session_state.user_profile.is_complete and st.session_state.agent_executor:
                    process_query(query)
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Chat with NitiAI")
        
        # Check if profile is complete
        if not st.session_state.user_profile.is_complete:
            st.warning("⚠️ Please complete your profile in the sidebar to get personalized scheme recommendations!")
        
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            for i, (query, response, timestamp) in enumerate(st.session_state.chat_history):
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🧑 You ({timestamp}):</strong><br>
                    {query}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 NitiAI:</strong><br>
                    {response}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_input("Ask about government schemes:", 
                                 placeholder="e.g., What schemes can I apply for?",
                                 key="chat_input")
        
        col_send, col_clear = st.columns([1, 1])
        
        with col_send:
            if st.button("🚀 Send", type="primary", use_container_width=True):
                if user_input.strip() and st.session_state.user_profile.is_complete and st.session_state.agent_executor:
                    process_query(user_input.strip())
                elif not user_input.strip():
                    st.error("Please enter a question!")
                elif not st.session_state.user_profile.is_complete:
                    st.error("Please complete your profile first!")
                elif not st.session_state.agent_executor:
                    st.error("AI model not loaded. Please check your API key and try refreshing the page.")
        
        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
    
    with col2:
        st.subheader("ℹ️ Help & Tips")
        
        st.info("""
        **Sample Questions:**
        
        🔍 **Discovery:**
        - "What schemes can I apply for?"
        - "Show me schemes for students"
        - "Find schemes for women entrepreneurs"
        
        ✅ **Eligibility:**
        - "Am I eligible for PM Kisan?"
        - "Check eligibility for Mudra Loan"
        
        📋 **Details:**
        - "What documents do I need for..."
        - "How to apply for..."
        - "Explain this scheme in simple terms"
        
        💰 **Benefits:**
        - "What benefits does ... offer?"
        - "How much money will I get?"
        """)
        
        # System status
        st.subheader("🔧 System Status")
        
        status_items = [
            ("👤 Profile", "✅ Complete" if st.session_state.user_profile.is_complete else "❌ Incomplete"),
            ("🤖 AI Model", "✅ Loaded" if st.session_state.agent_executor else "❌ Not Loaded"),
            ("📊 Database", "✅ Connected" if st.session_state.vectorstore_loaded else "❌ Disconnected")
        ]
        
        for item, status in status_items:
            st.write(f"{item}: {status}")

def process_query(query):
    """Process user query and get AI response"""
    if not st.session_state.agent_executor:
        st.error("AI model not loaded!")
        return
    
    try:
        # Show thinking indicator
        with st.spinner("🤔 NitiAI is thinking..."):
            # Format input with profile context
            profile_context = f"User Profile: Age: {st.session_state.user_profile.age}, Income: {st.session_state.user_profile.income}, Location: {st.session_state.user_profile.location}, Category: {st.session_state.user_profile.category}"
            formatted_input = f"{profile_context}\n\nUser Question: {query}"
            
            # Get response from agent
            response = st.session_state.agent_executor.invoke({
                "input": formatted_input
            })
            
            bot_response = response.get("output", "I'm unable to find a clear answer. Please check the official site or provide more details.")
        
        # Add to chat history
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state.chat_history.append((query, bot_response, timestamp))
        
        # Clear input and refresh
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()