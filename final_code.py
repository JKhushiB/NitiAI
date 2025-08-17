#requirements.txt
# !pip install -U langchain-huggingface
# !pip install groq langchain-groq
# !pip install langchain_community
# !pip install keybert sentence-transformers
# !pip install chromadb langchain sentence-transformers


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
from os.environ import getenv


# ============================================================================
df = pd.read_csv("updated_dataset.csv")
persist_dir = "chroma_store"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

'''later for relaoding the data '''

vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embedding_model
)


# Configure retriever for better results
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.7
    }
)
# ============================================================================
# USER PROFILE CLASS
# ============================================================================
class UserProfile:
    def __init__(self):
        self.age = None
        self.income = None
        self.location = None
        self.category = None
        self.occupation = None
        self.education = None
        self.is_complete = False

    def set_profile(self, age, income, location, category, occupation="", education=""):
        self.age = age
        self.income = income
        self.location = location
        self.category = category
        self.occupation = occupation
        self.education = education
        self.is_complete = True

    def get_profile_string(self):
        if not self.is_complete:
            return ""

        profile = f"Age: {self.age}, Income: {self.income}, Location: {self.location}"
        if self.category:
            profile += f", Category: {self.category}"
        if self.occupation:
            profile += f", Occupation: {self.occupation}"
        if self.education:
            profile += f", Education: {self.education}"
        return profile

    def get_search_context(self):
        """Get formatted context for better search results"""
        if not self.is_complete:
            return ""

        context_parts = []
        if self.category and self.category.lower() != 'general':
            context_parts.append(self.category)
        if self.location:
            context_parts.append(self.location)
        if self.age:
            try:
                age_num = int(self.age)
                if age_num < 18:
                    context_parts.append("minor student")
                elif age_num < 25:
                    context_parts.append("young adult student")
                elif age_num > 60:
                    context_parts.append("senior citizen elderly")
            except:
                pass

        return " ".join(context_parts)

# ============================================================================
# GLOBAL INSTANCES
# ============================================================================
# Initialize global user profile and cache
user_profile = UserProfile()

# Your existing initialization code
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Assuming vectorstore is initialized elsewhere in your code
# vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
# retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7})

os.environ["GROQ_API_KEY"] = "groqenv"  # Replace with your actual key

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=1000,
    timeout=30,
    max_retries=2
)

# Your existing summarization model setup
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

def custom_summarize(scheme_name, scheme_details, max_tokens=1024):
    """Your existing summarization function"""
    prompt = f"""
You are a helpful assistant. Summarize the following scheme in simple terms for a college student.
Make sure to clearly explain:
- What the scheme is about
- Who it is meant for
- What benefits it offers

Scheme Title: {scheme_name}
Details: {scheme_details}
""".strip()

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids
    prompt_trimmed = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    summary = summarizer(prompt_trimmed, max_length=120, min_length=40, do_sample=False)[0]['summary_text']
    return f"🎯 Scheme: {scheme_name}\n📄 Summary: {summary}\n"

# ============================================================================
# ENHANCED TOOLS WITH CACHING
# ============================================================================
class SchemeSearchTool(BaseTool):
    name: str = "scheme_search"
    description: str = """Search for government schemes based on user query.
    Use this when users ask about schemes they can apply for or want to find schemes for specific purposes.
    Input should be the user's query."""

    def _run(self, query: str) -> str:
        try:

            # Add user context to search query
            search_context = user_profile.get_search_context()
            enhanced_query = f"{query} {search_context}".strip()

            print(f"🔍 Searching with enhanced query: {enhanced_query}")
            docs = retriever.invoke(enhanced_query)


            if not docs:
                return "No schemes found matching your criteria. Try different keywords."

            results = []
            eligible_schemes = []
            for i, doc in enumerate(docs[:5]):
                scheme_name = doc.metadata.get("scheme_name", "Unknown")
                scheme_info = {
                    "rank": i + 1,
                    "scheme_name": scheme_name,
                    "category": doc.metadata.get("category", "General"),
                    "level": doc.metadata.get("level", "Unknown"),
                    "eligibility": doc.metadata.get("eligibility", "")[:300] + "..." if len(doc.metadata.get("eligibility", "")) > 300 else doc.metadata.get("eligibility", ""),
                    "benefits": doc.metadata.get("benefits", "")[:200] + "..." if len(doc.metadata.get("benefits", "")) > 200 else doc.metadata.get("benefits", ""),
                    "url": doc.metadata.get("url", "")
                }
                results.append(scheme_info)

                # Quick eligibility check based on profile
                eligibility_text = doc.metadata.get("eligibility", "").lower()
                user_age = int(user_profile.age) if user_profile.age and user_profile.age.isdigit() else 0
                user_location = user_profile.location.lower() if user_profile.location else ""

                # Basic eligibility filtering
                is_potentially_eligible = True

                # Age check
                if "18" in eligibility_text and user_age < 18:
                    is_potentially_eligible = False
                elif "21" in eligibility_text and "24" in eligibility_text and not (21 <= user_age <= 24):
                    is_potentially_eligible = False

                # Location check (for state-specific schemes)
                if any(state in eligibility_text for state in ["tamil nadu", "himachal pradesh", "kerala", "maharashtra"]):
                    if not any(state in user_location for state in ["tamil nadu", "himachal pradesh", "kerala", "maharashtra"]):
                        # Only mark as ineligible if it's clearly a different state scheme
                        if "resident" in eligibility_text or "domicile" in eligibility_text:
                            is_potentially_eligible = False

                if is_potentially_eligible:
                    eligible_schemes.append(scheme_name)

            result_text = f"🎯 Found {len(results)} employment schemes. Based on your profile (Age: {user_profile.age}, Location: {user_profile.location}), here are the most relevant ones:\n\n"

            for result in results:
                eligibility_status = "✅ Potentially Eligible" if result['scheme_name'] in eligible_schemes else "⚠️ Check Eligibility"
                result_text += f"• **{result['scheme_name']}** ({result['category']}) {eligibility_status}\n"
                result_text += f"  📋 Eligibility: {result['eligibility'][:150]}...\n"
                result_text += f"  💰 Benefits: {result['benefits'][:100]}...\n"
                result_text += f"  🔗 URL: {result['url']}\n\n"

            result_text += f"\n💡 **Next Steps:** Ask me 'Check eligibility for [scheme name]' for detailed eligibility verification, or 'What documents do I need for [scheme name]' for application requirements."

            return result_text

        except Exception as e:
            return f"Error searching schemes: {str(e)}"

class EligibilityCheckTool(BaseTool):
    name: str = "eligibility_check"
    description: str = """Check detailed eligibility for a specific scheme.
    Use this only when a user asks if they're eligible for a particular scheme.
    Input should be the scheme name."""

    def _run(self, scheme_name: str) -> str:
        try:
            if not user_profile.is_complete:
                return "I need your complete profile to check eligibility. Please set up your profile first."


            # Search for specific scheme
            docs = vectorstore.similarity_search(f"scheme_name:{scheme_name}", k=3)
            if not docs:
                docs = vectorstore.similarity_search(scheme_name, k=3)

            if not docs:
                return f"Could not find detailed information for scheme: {scheme_name}"

            doc = docs[0]

            result = f"🏛️ Eligibility Check for: {doc.metadata.get('scheme_name', '')}\n\n"
            result += f"👤 Your Profile: {user_profile.get_profile_string()}\n\n"
            result += f"📋 Eligibility Requirements:\n{doc.metadata.get('eligibility', '')}\n\n"
            result += f"📄 Required Documents:\n{doc.metadata.get('documents', '')}\n\n"
            result += f"💰 Benefits:\n{doc.metadata.get('benefits', '')}\n\n"
            result += f"📝 Application Process:\n{doc.metadata.get('application', '')}\n\n"
            result += f"🔗 Official URL: {doc.metadata.get('url', '')}"

            return result

        except Exception as e:
            return f"Error checking eligibility: {str(e)}"

class BenefitsSearchTool(BaseTool):
    name: str = "benefits_search"
    description: str = """Search for specific benefits provided by government schemes.
    Use this when users ask about what benefits they can get.
    Input should be the benefit type or general query about benefits."""

    def _run(self, query: str) -> str:
        try:

            # Search with user context
            search_context = user_profile.get_search_context()
            search_query = f"benefits {query} {search_context}".strip()
            docs = vectorstore.similarity_search(search_query, k=4)

            if not docs:
                return "No specific benefits found for your query. Try different keywords."

            result_text = f"💰 Benefits available for '{query}' based on your profile:\n\n"
            for i, doc in enumerate(docs[:3]):
                scheme_name = doc.metadata.get("scheme_name", "Unknown Scheme")
                benefits = doc.metadata.get("benefits", "")[:250]
                category = doc.metadata.get("category", "General")

                result_text += f"{i+1}. {scheme_name} ({category})\n"
                result_text += f"   💰 Benefits: {benefits}...\n"
                result_text += f"   🔗 URL: {doc.metadata.get('url', '')}\n\n"

            return result_text

        except Exception as e:
            return f"Error searching benefits: {str(e)}"

class DocumentRequirementTool(BaseTool):
    name: str = "document_requirements"
    description: str = """Get required documents for scheme application.
    Use this when users ask about what documents they need to apply for schemes."""

    def _run(self, scheme_name: str) -> str:
        try:
            # Try cache first

            docs = vectorstore.similarity_search(f"documents {scheme_name}", k=2)
            if not docs:
                return f"Could not find document requirements for: {scheme_name}"

            doc = docs[0]

            result = f"📄 Documents needed for {doc.metadata.get('scheme_name', scheme_name)}:\n\n"
            result += f"📋 Required Documents:\n{doc.metadata.get('documents', '')}\n\n"
            result += f"📝 Application Process:\n{doc.metadata.get('application', '')}\n\n"
            result += f"🔗 Official URL: {doc.metadata.get('url', '')}"

            return result

        except Exception as e:
            return f"Error getting document requirements: {str(e)}"

class SchemeSummaryTool(BaseTool):
    name: str = "scheme_summarizer"
    description: str = """Summarize one or more government schemes in simple language.
    Use this when users want a plain-language overview of schemes."""

    def _run(self, query: str) -> str:
        try:

            # docs = search_schemes_with_langchain(query, vectorstore)
            docs = vectorstore.similarity_search(query, k=3)

            if not docs:
                return "No matching schemes found for your query."

            summaries = []
            for doc in docs:
                scheme_name = doc.metadata.get('scheme_name', 'Unknown')
                scheme_details = doc.page_content
                summary = custom_summarize(scheme_name, scheme_details)
                summaries.append(summary)

            return "\n\n".join(summaries)

        except Exception as e:
            return f"Error while summarizing schemes: {str(e)}"

# ============================================================================
# AGENT SETUP
# ============================================================================
tools = [SchemeSearchTool(), EligibilityCheckTool(), DocumentRequirementTool(), BenefitsSearchTool(), SchemeSummaryTool()]

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True
)

prompt = PromptTemplate.from_template("""You are NitiAI, a helpful assistant specializing in Indian government schemes. You help citizens find schemes based on their demographics and needs.

CORE PRINCIPLES:
- Use the user's profile context provided with the question
- Be efficient - don't search for information you already have from recent conversation
- Choose the MOST APPROPRIATE single tool for each user question
- Only use multiple tools if the first tool's result is insufficient

DECISION MATRIX - Use ONLY ONE tool per query unless absolutely necessary:

🔍 **scheme_search**: When user asks for scheme discovery/recommendations
   - "What schemes can I apply for?"
   - "Show me employment schemes"
   - "Find schemes for women entrepreneurs"

✅ **eligibility_check**: When user asks about their eligibility for a SPECIFIC scheme
   - "Am I eligible for PM Kisan?"
   - "Check my eligibility for [scheme name] "

💰 **benefits_search**: When user asks about benefits/financial assistance
   - "What benefits does [scheme]  offer?"
   - "How much money will I get?"

📄 **document_requirements**: When user asks about application process/documents
   - "How to apply for [scheme] ?"
   - "What documents needed?"

📋 **scheme_summarizer**: When user asks for explanation of a specific scheme
   - "Explain [scheme name]"
   - "Tell me about this scheme in simple terms"

CONVERSATION CONTEXT AWARENESS:
- If discussing a scheme from previous exchanges, DON'T search again - use the appropriate specific tool
- Look at chat_history to see what schemes were recently discussed
- When user says "this scheme" or "it", they mean the scheme from recent conversation
- Always conclude with "Final Answer" after getting useful information
      FINAL RESPONSE GUIDELINES:
        FORMATTING:
        - Structure responses with clear sections using emojis as headers
        - Use bullet points for lists, not long paragraphs
        - Add line breaks between different topics
        - Keep sentences concise and actionable

        URL POLICY:
        - Include URLs only in a final "🔗 Useful Links:" section
        - Maximum 3 most relevant URLs per response
        - Don't repeatedly mention "visit the official website"
        - Group related links together

EFFICIENCY RULES:
- ONE tool call should answer most questions completely
- Only use additional tools if the first result lacks critical information the user specifically asked for
- Don't preemptively gather information the user didn't request

You have access to these tools:
{tools}

Use this exact format:
Question: the input question you must answer
Thought: think about what specific information the user is asking for and which ONE tool best provides that
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
...(repeat ONLY if the first result was insufficient for the user's specific question)
Thought: I now know the final answer
Final Answer: the final answer with specific recommendations and next steps

Previous conversation:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}
""")

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
    early_stopping_method="force"
)

def setup_user_profile():
    """Mandatory profile setup at the start"""
    print("\n" + "="*60)
    print("🏛️  WELCOME TO NitiAI - Government Scheme Assistant")
    print("="*60)
    print("To provide you with the most relevant schemes, I need some basic information about you.")
    print("This helps me filter schemes that you're actually eligible for!\n")

    while not user_profile.is_complete:
        try:
            print("📋 Profile Setup (all fields required):")
            age = input("1. Your Age: ").strip()
            if not age or not age.isdigit():
                print("❌ Please enter a valid age (numbers only)")
                continue

            income = input("2. Annual Income (e.g., 'Below 2 Lakhs', '2-5 Lakhs', '5-10 Lakhs'): ").strip()
            if not income:
                print("❌ Income information is required")
                continue

            location = input("3. Your State/City: ").strip()
            if not location:
                print("❌ Location is required")
                continue

            print("\n4. Category:")
            print("   a) General  b) SC  c) ST  d) OBC")
            category = input("   Select (a/b/c/d) or type category: ").strip()

            category_map = {'a': 'General', 'b': 'SC', 'c': 'ST', 'd': 'OBC'}
            if category.lower() in category_map:
                category = category_map[category.lower()]
            elif not category:
                print("❌ Category selection is required")
                continue

            occupation = input("5. Occupation (optional): ").strip()

            # Set the profile
            user_profile.set_profile(age, income, location, category, occupation)

            print(f"\n✅ Profile Created Successfully!")
            print(f"📊 Your Profile: {user_profile.get_profile_string()}")
            print(f"🎯 I'll now suggest schemes tailored to your profile!\n")
            print("="*60)
            break

        except KeyboardInterrupt:
            print("\n\n👋 Setup cancelled. Goodbye!")
            exit()
        except Exception as e:
            print(f"❌ Error during setup: {e}. Please try again.")

def chat_with_agent(user_input):
    """Enhanced chat function that includes user profile context"""
    try:
        if not user_profile.is_complete:
            return "❌ Please complete your profile setup first."

        # Format the input with user profile embedded in the prompt
        # Embed profile directly in input string
        profile_context = f"User Profile: Age: {user_profile.age}, Income: {user_profile.income}, Location: {user_profile.location}, Category: {user_profile.category}"
        formatted_input = f"{profile_context}\n\nUser Question: {user_input}"

        response = agent_executor.invoke({
            "input": formatted_input  # Only this parameter
        })
        return response.get("output", "I'm unable to find a clear answer. Please check the official site or provide more details.")

    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}. Please try rephrasing your question."

def interactive_chat():
    """Enhanced interactive chat with mandatory profile setup"""
    # Mandatory profile setup
    setup_user_profile()

    print("\n💬 You can now ask me about government schemes!")
    print("💡 Try asking:")
    print("   • 'What education schemes can I apply for?'")
    print("   • 'Show me schemes for employment'")
    print("   • 'Am I eligible for PM Kisan scheme?'")
    print("   • 'What documents do I need for...'")
    print("\n📝 Commands: 'profile' to update profile, 'quit' to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye', 'stop', 'end']:
                print("\n🙏 Thank you for using NitiAI! I hope you found useful schemes.")
                print("💡 Remember to bookmark the official URLs I shared!")
                print("👋 Goodbye and best of luck with your applications!")
                break

            elif user_input.lower() == 'profile':
                print(f"\n📊 Current Profile: {user_profile.get_profile_string()}")
                update = input("Do you want to update your profile? (y/n): ").strip().lower()
                if update == 'y':
                    user_profile.is_complete = False
                    setup_user_profile()
                continue

            elif not user_input:
                print("Please enter a question or command.")
                continue

            # Get agent response
            print("🤔 Thinking...")
            response = chat_with_agent(user_input)
            print(f"\n🤖 NitiAI: {response}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Take care!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_chat()