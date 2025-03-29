# llm_selector_app.py

import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
import os
import torch

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Smart LLM Recommender", layout="centered")
st.title("Smart LLM Recommender")
st.markdown("Answer a few quick questions and get the best-fit LLM for your task!")

# ---------------------- MAIN INTENT ----------------------
page = st.sidebar.radio("🔀 Select LLM Category:", [ "Big LLMs", "Small/Open-Source LLMs", "financial insights", "Keyword-Based Routing"])
st.sidebar.markdown("---")
st.sidebar.caption("Choose your AI path!")
if page == "Big LLMs":
    st.subheader("Step 1: What is your goal?")
    intent = st.radio(
        "Select your primary focus:",
        [
            "Ethical/Policy-Oriented",
            "Technical/Problem Solving",
            "Real-time web answers",
            "Policy writing",
            "Deep reasoning or technical writing",
            "General-purpose assistant",
            "Social trends or X.com insights"
        ]
    )

    # ---------------------- CHECKLISTS ----------------------

    # Question Set 1: Content Type
    st.subheader("Step 2: What kind of content are you working with?")
    content_type = st.multiselect(
        "Select applicable options:",
        ["Creative writing", "Code generation", "Factual research", "Data interpretation"]
    )

    # Question Set 2: Data Needs
    st.subheader("Step 3: What kind of data access do you need?")
    data_access = st.multiselect(
        "Choose what applies:",
        ["Real-time web data", "Social media trends (X.com)", "Offline/secure", "Multimodal (images/audio)"]
    )

    # Question Set 3: Response Style
    st.subheader("Step 4: Preferred response tone or handling?")
    response_style = st.multiselect(
        "What tone do you expect?",
        ["Structured reasoning", "Conversational", "Long-memory/threaded", "Fact-based with sources"]
    )

    # ---------------------- LLM Matching Logic ----------------------

    def determine_llm(intent, content_type, data_access, response_style):
        if "Real-time web data" in data_access or "Fact-based with sources" in response_style or "Real-time web answers" in intent:
            return "Perplexity"
        if "Social media trends (X.com)" in data_access or "Social trends or X.com insights" in intent:
            return "Grok 3"
        if intent in ["Ethical/Policy-Oriented", "Policy writing"] and "Creative writing" in content_type:
            return "Claude"
        if intent in ["Technical/Problem Solving", "General-purpose assistant"] and "Code generation" in content_type:
            return "ChatGPT"
        if "Structured reasoning" in response_style or "Data interpretation" in content_type or "Deep reasoning or technical writing" in intent:
            return "DeepSeek"
        return "ChatGPT"  # fallback default

    # ---------------------- Display Recommendation ----------------------

    if content_type and data_access and response_style:
        selected_llm = determine_llm(intent, content_type, data_access, response_style)
        st.success(f"✅ Best-Matching LLM: **{selected_llm}**")
    else:
        selected_llm = None
        st.info("Please complete all steps above to get a recommendation.")

    # ---------------------- ASK A QUESTION ----------------------

    if selected_llm:
        st.subheader("Step 5: Ask Your Question to the Recommended LLM")
        user_question = st.text_input("Enter your question here:")

        if st.button("Get Answer"):
            openai_api_key = os.getenv("OPENAI_API_KEY") or "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"

            # Dummy implementations for non-OpenAI models
            if selected_llm == "ChatGPT":
                llm = ChatOpenAI(model="gpt-4", temperature=0.6, api_key=openai_api_key)
            elif selected_llm == "Claude":
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, api_key=openai_api_key)  # Placeholder
            elif selected_llm == "Perplexity":
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, api_key=openai_api_key)  # Placeholder
            elif selected_llm == "Grok 3":
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, api_key=openai_api_key)  # Placeholder
            elif selected_llm == "DeepSeek":
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, api_key=openai_api_key)  # Placeholder
            else:
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, api_key=openai_api_key)

            # Prompt and chain
            prompt = PromptTemplate(
                input_variables=["question"],
                template="You are a helpful assistant. Answer this: {question}"
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            response = chain.run(user_question)

            st.markdown("### 🗨️ Response:")
            st.write(response)

    # ---------------------- Footer ----------------------

    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit | Choose Smart. Prompt Smarter.")
elif page == "Small/Open-Source LLMs":
    # st.title("🧩 Small/Open-Source LLM Recommender")
    import os
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

    use_case = st.radio("Step 1: What’s your use case?", [
        "Fast local inference", "Lightweight edge deployment", "Open-source R&D", "Educational/demo purpose"
    ])

    skill_level = st.radio("Step 2: Your technical comfort level?", ["Beginner", "Intermediate", "Advanced"])

    response_pref = st.multiselect(
        "Step 3: What response quality matters to you?",
        ["Speed over accuracy", "Accurate reasoning", "Minimal memory usage", "Code-friendly"]
    )

    def get_small_llm(use_case, skill_level, response_pref):
        if "Speed over accuracy" in response_pref:
            return "Mistral"
        if "Minimal memory usage" in response_pref or skill_level == "Beginner":
            return "TinyLlama"
        if "Accurate reasoning" in response_pref or use_case == "Open-source R&D":
            return "LLaMA 3"
        if "Code-friendly" in response_pref:
            return "Falcon"
        return "Mistral"

    if use_case and response_pref:
        selected_llm = get_small_llm(use_case, skill_level, response_pref)
        st.success(f"✅ Recommended Small LLM: **{selected_llm}**")
    else:
        selected_llm = None
        st.info("Complete the steps to see your recommended LLM.")

# -------------------- ASK YOUR QUESTION -------------------- #
    if selected_llm:
        st.subheader("💬 Ask a Question to the Recommended LLM")

        user_input = st.text_input("Enter your question here:")
        if st.button("Get Answer"):
            if "ChatGPT" in selected_llm or "Claude" in selected_llm:
                llm = ChatOpenAI(model="gpt-4", temperature=0.6, api_key="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")
            elif selected_llm == "LLaMA 3":
                llm = HuggingFaceHub(repo_id="meta-llama/Llama-3-8b", model_kwargs={"temperature": 0.6})
            elif selected_llm == "Mistral":
                llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1", model_kwargs={"temperature": 0.6})
            elif selected_llm == "TinyLlama":
                llm = HuggingFaceHub(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", model_kwargs={"temperature": 0.6})
            elif selected_llm == "Falcon":
                llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.6})
            else:
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6,  api_key="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")

            prompt = PromptTemplate(
                input_variables=["question"],
                template="You are a helpful assistant. Answer the following: {question}"
            )

            chain = LLMChain(llm=llm, prompt=prompt)
            response = chain.run(user_input)

            st.markdown("### 🧠 Response:")
            st.write(response)
elif page == "financial insights":
    import pandas as pd
    st.title("📊 Federated Financial Insights")
    st.markdown("Upload financial data from multiple clients. We’ll simulate federated learning to build a global model and extract insights using LLMs.")

    uploaded_files = st.file_uploader(
        "📂 Upload client CSV files (minimum 2):", type=["csv"], accept_multiple_files=True
    )

    if uploaded_files and len(uploaded_files) >= 2:
        client_dataframes = []
        st.subheader("📁 Client Data Summary")

        for file in uploaded_files:
            df = pd.read_csv(file)
            client_name = file.name.replace(".csv", "")
            client_dataframes.append(df)

            with st.expander(f"Preview: {client_name}"):
                st.write(f"🔹 Rows: {len(df)} | 🔹 Columns: {len(df.columns)}")
                st.dataframe(df.head())

        # Step 1: Federated Aggregation
        st.subheader("🌍 Simulated Aggregation (Global Model)")
        global_df = pd.concat(client_dataframes, ignore_index=True)

        st.markdown("**🧾 Combined Global Dataset Preview:**")
        st.dataframe(global_df.head(10))

        st.markdown("**📈 Global Statistical Summary:**")
        st.dataframe(global_df.describe())

        # Step 2: Generate Insights via LLM
        st.subheader("LLM-Based Global Insight Summary")

        if st.button("Generate Insights"):
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            from langchain_openai import ChatOpenAI

            # Set OpenAI key
            llm = ChatOpenAI(model="gpt-4", temperature=0.6, api_key="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")

            # Format the statistical summary for LLM
            stats_summary = global_df.describe().to_string()

            # Prompt template
            prompt = PromptTemplate(
                input_variables=["stats"],
                template="""
                You are a senior financial analyst. Analyze the global statistics from multiple clients’ datasets provided below.
                Identify:
                - Overall financial trends
                - Noteworthy patterns
                - Risk or anomaly indicators
                - Business recommendations

                Use a professional and concise tone.

                DATA:
                {stats}
                """
            )

            # Run LangChain LLM chain
            chain = LLMChain(llm=llm, prompt=prompt)
            response = chain.run(stats_summary)

            st.markdown("### 📝 Global Federated Financial Insights:")
            st.write(response)

    else:
        st.info("Please upload at least two financial CSV files to simulate federated learning.")
# -------------------- FOOTER -------------------- #

elif page == "Keyword-Based Routing":
    import re
    from collections import Counter

    st.title("Keyword-Based LLM Router")
    st.markdown("Enter a query and we'll extract keywords to automatically choose the best LLM for your task.")

    user_query = st.text_area("🔍 Enter your query:")

    # Keyword → LLM routing rules
    keyword_llm_map = {
        # ChatGPT
        "code": "ChatGPT", "python": "ChatGPT", "developer": "ChatGPT",
        "build": "ChatGPT", "creative": "ChatGPT", "story": "ChatGPT",
        "fiction": "ChatGPT", "chatbot": "ChatGPT", "image": "ChatGPT",
        "audio": "ChatGPT", "multimodal": "ChatGPT",

        # Claude
        "ethical": "Claude", "morality": "Claude", "values": "Claude",
        "bias": "Claude", "fairness": "Claude", "inclusive": "Claude",
        "policy": "Claude", "regulation": "Claude", "guidelines": "Claude",
        "trust": "Claude",

        # Perplexity
        "real-time": "Perplexity", "web": "Perplexity", "internet": "Perplexity",
        "news": "Perplexity", "search": "Perplexity", "current": "Perplexity",
        "update": "Perplexity", "now": "Perplexity",

        # Grok 3
        "x.com": "Grok 3", "elon": "Grok 3", "musk": "Grok 3",
        "trending": "Grok 3", "tweets": "Grok 3", "threads": "Grok 3",
        "social": "Grok 3", "buzz": "Grok 3",

        # DeepSeek
        "analyze": "DeepSeek", "reasoning": "DeepSeek", "logic": "DeepSeek",
        "explain": "DeepSeek", "insight": "DeepSeek", "data": "DeepSeek",
        "technical": "DeepSeek", "evaluate": "DeepSeek", "statistics": "DeepSeek"
    }

    # Explanation dictionary
    model_explanations = {
        "ChatGPT": "🧠 Best for creative tasks, coding, and multimodal content.",
        "Claude": "⚖️ Ideal for ethical reasoning and policy-driven decisions.",
        "Perplexity": "🌐 Designed for web-aware, real-time information queries.",
        "Grok 3": "📈 Great for trend analysis and X.com (Twitter) content.",
        "DeepSeek": "🔍 Strong in logic, reasoning, and structured data analysis."
    }

    def extract_keywords(text):
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    if st.button("🔎 Extract & Route"):
        if not user_query.strip():
            st.warning("⚠️ Please enter a valid query.")
        else:
            keywords = extract_keywords(user_query)
            match_counts = {}

            for word in keywords:
                if word in keyword_llm_map:
                    model = keyword_llm_map[word]
                    match_counts[model] = match_counts.get(model, 0) + 1

            if match_counts:
                ranked = sorted(match_counts.items(), key=lambda x: x[1], reverse=True)
                best_llm = ranked[0][0]
                matched_keywords = [k for k in keywords if keyword_llm_map.get(k) == best_llm]

                st.success(f"✅ Routed to: **{best_llm}**")
                st.markdown(f"🔑 Matched Keywords: `{', '.join(matched_keywords)}`")
                st.info(model_explanations.get(best_llm, "No explanation available."))

                # Call actual model
                openai_api_key = os.getenv("OPENAI_API_KEY") or "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"


                if best_llm == "ChatGPT":
                    llm = ChatOpenAI(model="gpt-4", api_key=openai_api_key)
                elif best_llm in ["Claude", "Perplexity", "Grok 3", "DeepSeek"]:
                    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key="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")  # Simulated for now
                else:
                    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key="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")

                from langchain.prompts import PromptTemplate
                from langchain.chains import LLMChain

                prompt = PromptTemplate(
                    input_variables=["question"],
                    template="You are a helpful assistant. Answer this: {question}"
                )
                chain = LLMChain(llm=llm, prompt=prompt)
                response = chain.run(user_query)

                st.markdown("### 🧠 Response:")
                st.write(response)

            else:
                st.warning("⚠️ No matched keywords. Using GPT-4 fallback.")
                openai_api_key= "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"
                fallback_llm = ChatOpenAI(model="gpt-4", api_key=openai_api_key)
                prompt = PromptTemplate(
                    input_variables=["question"],
                    template="You are a helpful assistant. Answer this: {question}"
                )
                chain = LLMChain(llm=fallback_llm, prompt=prompt)
                response = chain.run(user_query)
                st.markdown("### 🧠 GPT-4 Fallback Response:")
                st.write(response)
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, LangChain, and the power of open LLMs.")
