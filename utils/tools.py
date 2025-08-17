# utils/tools.py

from langchain.tools import BaseTool
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import streamlit as st
from typing import Any, Type
from pydantic import Field

# Initialize summarization model
@st.cache_resource
def load_summarization_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    return summarizer, tokenizer

def custom_summarize(scheme_name, scheme_details, max_tokens=1024):
    """Custom summarization function"""
    summarizer, tokenizer = load_summarization_model()
    
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

class SchemeSearchTool(BaseTool):
    name: str = "scheme_search"
    description: str = """Search for government schemes based on user query.
    Use this when users ask about schemes they can apply for or want to find schemes for specific purposes.
    Input should be the user's query."""


      # Declare the fields properly for Pydantic
    vectorstore: Any = Field(description="Vector store for searching schemes")
    user_profile: Any = Field(description="User profile for personalization")
    retriever: Any = None
    class Config:
        arbitrary_types_allowed = True


    # def __init__(self, vectorstore, user_profile,**kwargs):
    #     super().__init__(vectorstore=vectorstore, user_profile=user_profile, **kwargs)
    #     # self.vectorstore = vectorstore
    #     # self.user_profile = user_profile
    #     self.retriever = vectorstore.as_retriever(
    #         search_type="mmr",
    #         search_kwargs={
    #             "k": 5,
    #             "fetch_k": 20,
    #             "lambda_mult": 0.7
    #         }
    #     )

    def model_post_init(self, __context: Any) -> None:
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
        )


    def _run(self, query: str) -> str:
        try:
            # Add user context to search query
            search_context = self.user_profile.get_search_context()
            enhanced_query = f"{query} {search_context}".strip()

            docs = self.retriever.invoke(enhanced_query)

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
                user_age = int(self.user_profile.age) if self.user_profile.age and self.user_profile.age.isdigit() else 0
                user_location = self.user_profile.location.lower() if self.user_profile.location else ""

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

            result_text = f"🎯 Found {len(results)} employment schemes. Based on your profile (Age: {self.user_profile.age}, Location: {self.user_profile.location}), here are the most relevant ones:\n\n"

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


    vectorstore: Any = Field(description="Vector store for searching schemes")
    user_profile: Any = Field(description="User profile for personalization")

    class Config:
        arbitrary_types_allowed = True
    # def __init__(self, vectorstore, user_profile):
    #     super().__init__()
    #     self.vectorstore = vectorstore
    #     self.user_profile = user_profile

    def _run(self, scheme_name: str) -> str:
        try:
            if not self.user_profile.is_complete:
                return "I need your complete profile to check eligibility. Please set up your profile first."

            # Search for specific scheme
            docs = self.vectorstore.similarity_search(f"scheme_name:{scheme_name}", k=3)
            if not docs:
                docs = self.vectorstore.similarity_search(scheme_name, k=3)

            if not docs:
                return f"Could not find detailed information for scheme: {scheme_name}"

            doc = docs[0]

            result = f"🏛️ Eligibility Check for: {doc.metadata.get('scheme_name', '')}\n\n"
            result += f"👤 Your Profile: {self.user_profile.get_profile_string()}\n\n"
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

    # def __init__(self, vectorstore, user_profile):
    #     super().__init__()
    #     self.vectorstore = vectorstore
    #     self.user_profile = user_profile

      # Declare the fields properly for Pydantic
    vectorstore: Any = Field(description="Vector store for searching schemes")
    user_profile: Any = Field(description="User profile for personalization")

    class Config:
        arbitrary_types_allowed = True


    def _run(self, query: str) -> str:
        try:
            # Search with user context
            search_context = self.user_profile.get_search_context()
            search_query = f"benefits {query} {search_context}".strip()
            docs = self.vectorstore.similarity_search(search_query, k=4)

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

    # def __init__(self, vectorstore, user_profile):
    #     super().__init__()
    #     self.vectorstore = vectorstore
    #     self.user_profile = user_profile

      # Declare the fields properly for Pydantic
    vectorstore: Any = Field(description="Vector store for searching schemes")
    user_profile: Any = Field(description="User profile for personalization")

    class Config:
        arbitrary_types_allowed = True


    def _run(self, scheme_name: str) -> str:
        try:
            docs = self.vectorstore.similarity_search(f"documents {scheme_name}", k=2)
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

    # def __init__(self, vectorstore, user_profile):
    #     super().__init__()
    #     self.vectorstore = vectorstore
    #     self.user_profile = user_profile

      # Declare the fields properly for Pydantic
    vectorstore: Any = Field(description="Vector store for searching schemes")
    user_profile: Any = Field(description="User profile for personalization")

    class Config:
        arbitrary_types_allowed = True


    def _run(self, query: str) -> str:
        try:
            docs = self.vectorstore.similarity_search(query, k=3)

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