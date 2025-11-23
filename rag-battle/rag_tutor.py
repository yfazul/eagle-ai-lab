import os
import streamlit as st
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

import chromadb

# ==========================================================
# CONFIGURATION
# ==========================================================
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ==========================================================
# LANGCHAIN COMPONENTS INITIALIZATION
# ==========================================================

@st.cache_resource
def get_embeddings():
    """Initialize LangChain OpenAI embeddings (1536 dims)."""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
        # No dimensions param → full 1536
    )

@st.cache_resource
def get_llm(temperature: float = 0.7):
    """Initialize LangChain ChatOpenAI"""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        openai_api_key=OPENAI_API_KEY
    )

@st.cache_resource
def get_chroma_client():
    """Initialize Chroma Cloud client"""
    return chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )

@st.cache_resource
def get_neo4j_graph():
    """Initialize LangChain Neo4j Graph"""
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )

@st.cache_resource
def get_text_vectorstore():
    """Get LangChain Chroma vectorstore for text"""
    client = get_chroma_client()
    embeddings = get_embeddings()

    # Ensure collection exists (must be recreated with 1536-dim vectors)
    _ = client.get_collection("tn_class6_text")

    return Chroma(
        client=client,
        collection_name="tn_class6_text",
        embedding_function=embeddings,
    )

@st.cache_resource
def get_image_vectorstore():
    """Get LangChain Chroma vectorstore for images"""
    client = get_chroma_client()
    embeddings = get_embeddings()

    _ = client.get_collection("tn_class6_images")

    return Chroma(
        client=client,
        collection_name="tn_class6_images",
        embedding_function=embeddings,
    )

# ==========================================================
# KEYWORD EXTRACTION (IMPROVED)
# ==========================================================

STOPWORDS = {
    "what", "who", "whom", "whose", "which", "where", "when", "why", "how",
    "tell", "about", "explain", "describe", "show", "give", "find", "get",
    "help", "need", "want", "like", "know", "think", "say", "said", "do",
    "does", "did", "done", "doing", "is", "are", "was", "were", "be", "been",
    "being", "has", "have", "had", "having", "can", "could", "may", "might",
    "must", "shall", "should", "will", "would", "the", "a", "an", "and", "or",
    "but", "if", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "from", "up", "on", "off", "out", "in", "over", "under", "me",
    "you", "your", "he", "she", "it", "we", "they", "this", "that", "please"
}

def extract_keywords(query: str) -> List[str]:
    """
    Improved keyword extractor:
    - keeps important verbs (talked, said, explained, etc)
    - keeps 'who', 'what', etc. for name-based questions
    - keeps other non-stopword tokens with len > 2
    """
    import re

    tokens = re.findall(r"\w+", query)
    important_verbs = {"talked", "said", "told", "discussed", "explained", "asked"}
    important_w_words = {"who", "what", "which"}

    keywords: List[str] = []

    for word in tokens:
        lower = word.lower()
        if lower in important_verbs or lower in important_w_words:
            keywords.append(word)
        elif len(lower) > 2 and lower not in STOPWORDS:
            keywords.append(word)

    # Deduplicate case-insensitively
    seen = set()
    result: List[str] = []
    for kw in keywords:
        l = kw.lower()
        if l not in seen:
            seen.add(l)
            result.append(kw)
    return result

# ==========================================================
# MULTI-QUERY RETRIEVER (KEYWORD-AUGMENTED)
# ==========================================================

class MultiQueryRetriever:
    """Custom retriever that combines base query with keyword queries."""

    def __init__(self, vectorstore, k: int = 8):
        self.vectorstore = vectorstore
        self.k = k

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents using query + keyword-based queries."""
        docs: List[Document] = []

        # Base retrieval
        base_docs = self.vectorstore.similarity_search(query, k=self.k)
        docs.extend(base_docs)
        seen_ids = {(d.metadata.get("id"), d.metadata.get("page")) for d in docs}

        # Keyword-based retrieval
        keywords = extract_keywords(query)
        for kw in keywords:
            kw_docs = self.vectorstore.similarity_search(kw, k=self.k)
            for doc in kw_docs:
                doc_key = (doc.metadata.get("id"), doc.metadata.get("page"))
                if doc_key not in seen_ids:
                    docs.append(doc)
                    seen_ids.add(doc_key)

        return docs

# ==========================================================
# KNOWLEDGE GRAPH RETRIEVAL
# ==========================================================

class KnowledgeGraphRetriever:
    """Custom retriever for Neo4j Knowledge Graph."""

    def __init__(self, graph: Neo4jGraph):
        self.graph = graph

    def get_relevant_documents(self, query: str, top_k: int = 8) -> List[Document]:
        """Retrieve from KG using keyword matching."""
        keywords = extract_keywords(query)
        if not keywords:
            return []

        cypher_query = """
        // Text chunks
        MATCH (t:TextChunk)
        WHERE any(keyword IN $keywords WHERE toLower(t.text) CONTAINS toLower(keyword))
        WITH t, size([keyword IN $keywords WHERE toLower(t.text) CONTAINS toLower(keyword)]) AS matches
        RETURN t.id AS id, t.text AS text, t.page AS page, 'text' AS type, matches
        ORDER BY matches DESC, t.page
        LIMIT $top_k

        UNION

        // Image OCR chunks
        MATCH (i:ImageChunk)
        WHERE i.has_text = true
          AND any(keyword IN $keywords WHERE toLower(i.ocr_text) CONTAINS toLower(keyword))
        WITH i, size([keyword IN $keywords WHERE toLower(i.ocr_text) CONTAINS toLower(keyword)]) AS matches
        RETURN i.id AS id, i.ocr_text AS text, i.page AS page, 'image' AS type, matches
        ORDER BY matches DESC, i.page
        LIMIT 5
        """

        try:
            results = self.graph.query(
                cypher_query,
                params={
                    "keywords": [kw.lower() for kw in keywords],
                    "top_k": top_k
                },
            )

            documents: List[Document] = []
            for rec in results:
                documents.append(
                    Document(
                        page_content=rec.get("text", "") or "",
                        metadata={
                            "id": rec.get("id"),
                            "page": rec.get("page"),
                            "type": rec.get("type"),
                            "source": "knowledge_graph",
                            "matches": rec.get("matches", 0),
                        },
                    )
                )
            return documents
        except Exception as e:
            st.warning(f"KG retrieval error: {e}")
            return []

# ==========================================================
# HYBRID RETRIEVER
# ==========================================================

class HybridRetriever:
    """Combines vector search and knowledge graph retrieval."""

    def __init__(
        self,
        text_vectorstore,
        image_vectorstore,
        kg_retriever: KnowledgeGraphRetriever,
        text_k: int = 10,
        image_k: int = 5,
    ):
        self.text_retriever = MultiQueryRetriever(text_vectorstore, k=text_k)
        self.image_retriever = MultiQueryRetriever(image_vectorstore, k=image_k)
        self.kg_retriever = kg_retriever

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve from all sources."""
        docs: List[Document] = []

        # Vector search - text
        text_docs = self.text_retriever.get_relevant_documents(query)
        for d in text_docs:
            d.metadata["source_type"] = "text_vector"
            docs.append(d)

        # Vector search - images
        image_docs = self.image_retriever.get_relevant_documents(query)
        for d in image_docs:
            d.metadata["source_type"] = "image_vector"
            docs.append(d)

        # Knowledge graph
        kg_docs = self.kg_retriever.get_relevant_documents(query)
        docs.extend(kg_docs)

        return docs

# ==========================================================
# CORRECTIVE RAG CHAINS
# ==========================================================

def create_retrieval_critic_chain():
    """Chain to evaluate retrieval quality and suggest better queries."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a retrieval quality critic for a textbook RAG system.
You are given a QUESTION and the RETRIEVED CONTEXT (snippets from a textbook).

Your job:
1. Judge how well this context can answer the question.
2. If the context is weak or missing key info, propose a BETTER_SEARCH_QUERY.

Output strictly in this format:
ADEQUACY: HIGH/MEDIUM/LOW
BETTER_QUERY: <new query or NONE>""",
            ),
            (
                "user",
                """QUESTION:
{query}

RETRIEVED CONTEXT:
{context}""",
            ),
        ]
    )

    llm = get_llm(temperature=0.2)
    return prompt | llm | StrOutputParser()

def create_answer_chain():
    """Chain to generate initial answer from context."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful tutor for the TN Class 6 English textbook.
Use ONLY the provided context from the textbook to answer.
If the context is not enough, say that some information is missing,
but still summarize what is present.

Give clear, student-friendly answers and reference page numbers where possible.""",
            ),
            (
                "user",
                """CONTEXT:
{context}

QUESTION: {query}

Give a clear, student-friendly answer using only this context.""",
            ),
        ]
    )

    llm = get_llm(temperature=0.7)
    return prompt | llm | StrOutputParser()

def create_verification_chain():
    """Chain to verify and correct the answer."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a STRICT verifier for a textbook-based tutor.
You will be given a QUESTION, CONTEXT, and a DRAFT ANSWER.

Your job:
1. Check if every important claim in the DRAFT ANSWER is supported by the CONTEXT.
2. If some parts are not supported or contradict the context, fix/remove them.
3. If the context does not contain enough info, say that clearly.

At the end, output:
ADEQUACY: HIGH/MEDIUM/LOW""",
            ),
            (
                "user",
                """QUESTION:
{query}

CONTEXT:
{context}

DRAFT ANSWER:
{draft_answer}

Now produce the final corrected answer for the student, then the ADEQUACY line.""",
            ),
        ]
    )

    llm = get_llm(temperature=0.3)
    return prompt | llm | StrOutputParser()

# ==========================================================
# CORRECTIVE RAG PIPELINE
# ==========================================================

def format_docs(docs: List[Document]) -> str:
    """Format documents into a context string."""
    parts: List[str] = []

    text_docs = [d for d in docs if d.metadata.get("source_type") == "text_vector"]
    image_docs = [d for d in docs if d.metadata.get("source_type") == "image_vector"]
    kg_docs = [d for d in docs if d.metadata.get("source") == "knowledge_graph"]

    if text_docs:
        parts.append("=== RELEVANT TEXT CONTENT ===")
        for d in text_docs:
            page = d.metadata.get("page", "?")
            parts.append(f"[Page {page}] {d.page_content}")

    if image_docs:
        parts.append("\n=== RELEVANT IMAGES/DIAGRAMS ===")
        for d in image_docs:
            page = d.metadata.get("page", "?")
            parts.append(f"[Page {page}, Image] {d.page_content}")

    if kg_docs:
        parts.append("\n=== ADDITIONAL CONTEXT FROM KNOWLEDGE GRAPH ===")
        for d in kg_docs:
            page = d.metadata.get("page", "?")
            t = d.metadata.get("type", "")
            parts.append(f"[Page {page} {t}] {d.page_content}")

    return "\n\n".join(parts).strip()

def run_corrective_rag_pipeline(
    query: str, retriever: HybridRetriever
) -> Tuple[str, List[Document], int]:
    """
    Full Corrective RAG pipeline:
    1. Hybrid retrieval
    2. Evaluate retrieval quality
    3. Re-retrieve if needed
    4. Generate draft answer
    5. Verify and correct answer
    6. Compute confidence
    """
    # Step 1: initial retrieval
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return (
            "I couldn't find any relevant information about this question in the pages that were indexed.",
            [],
            5,
        )

    # Step 2: evaluate retrieval
    context_preview = format_docs(docs[:8])[:1500]
    critic_chain = create_retrieval_critic_chain()
    critic_result = critic_chain.invoke({"query": query, "context": context_preview})

    retrieval_adequacy = "medium"
    better_query = None

    for line in critic_result.split("\n"):
        l = line.strip()
        if l.upper().startswith("ADEQUACY:"):
            val = l.split(":", 1)[1].strip().upper()
            if "HIGH" in val:
                retrieval_adequacy = "high"
            elif "LOW" in val:
                retrieval_adequacy = "low"
        elif l.upper().startswith("BETTER_QUERY:"):
            val = l.split(":", 1)[1].strip()
            if val and val.upper() != "NONE":
                better_query = val

    # Step 3: re-retrieve if needed
    if retrieval_adequacy == "low" and better_query:
        improved_docs = retriever.get_relevant_documents(better_query)
        seen = {(d.metadata.get("id"), d.metadata.get("page")) for d in docs}
        for d in improved_docs:
            key = (d.metadata.get("id"), d.metadata.get("page"))
            if key not in seen:
                docs.append(d)
                seen.add(key)

    # Step 4: draft answer
    context_str = format_docs(docs)
    answer_chain = create_answer_chain()
    draft_answer = answer_chain.invoke({"query": query, "context": context_str})

    # Step 5: verify
    verification_chain = create_verification_chain()
    verified = verification_chain.invoke(
        {"query": query, "context": context_str, "draft_answer": draft_answer}
    )

    answer_adequacy = "medium"
    if "ADEQUACY:" in verified:
        parts = verified.split("ADEQUACY:", 1)
        final_answer = parts[0].strip()
        marker = parts[1].strip().upper()
        if "HIGH" in marker:
            answer_adequacy = "high"
        elif "LOW" in marker:
            answer_adequacy = "low"
    else:
        final_answer = verified.strip()
        if any(
            p in final_answer.lower()
            for p in ["not enough information", "no details", "no information"]
        ):
            answer_adequacy = "low"

    # Step 6: confidence score
    confidence = compute_confidence_score(
        docs, answer_adequacy=answer_adequacy, retrieval_adequacy=retrieval_adequacy
    )

    return final_answer, docs, confidence

# ==========================================================
# CONFIDENCE SCORE
# ==========================================================

def compute_confidence_score(
    docs: List[Document],
    answer_adequacy: str = "medium",
    retrieval_adequacy: str = "medium",
) -> int:
    """Compute confidence score (0–100)."""

    if not docs:
        return 0

    def adequacy_to_score(a: str) -> float:
        a = (a or "").lower()
        if a == "high":
            return 1.0
        if a == "low":
            return 0.3
        return 0.65

    answer_score = adequacy_to_score(answer_adequacy)
    retrieval_score = adequacy_to_score(retrieval_adequacy)

    # result count
    total = len(docs)
    if total >= 10:
        count_score = 1.0
    elif total >= 6:
        count_score = 0.85
    elif total >= 3:
        count_score = 0.7
    else:
        count_score = 0.5

    # content length
    total_len = sum(len(d.page_content) for d in docs)
    if total_len >= 1000:
        content_score = 1.0
    elif total_len >= 500:
        content_score = 0.85
    elif total_len >= 250:
        content_score = 0.7
    else:
        content_score = max(0.3, total_len / 400.0)

    # source diversity
    src_types = set(
        d.metadata.get("source_type", d.metadata.get("source", "unknown")) for d in docs
    )
    diversity_score = min(len(src_types) / 3.0, 1.0)

    combined = (
        0.25 * answer_score
        + 0.2 * retrieval_score
        + 0.2 * count_score
        + 0.2 * content_score
        + 0.15 * diversity_score
    )

    score = int(round(max(0.0, min(1.0, combined)) * 100))

    if answer_adequacy == "low":
        score = min(score, 55)

    return score

# ==========================================================
# STREAMLIT UI
# ==========================================================

def main():
    st.set_page_config(
        page_title="TN Class 6 English Tutor",
        page_icon="📚",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-header">📚 TN Class 6 English Tutor</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">LangChain + Hybrid RAG + Knowledge Graph + Corrective RAG</div>',
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True

    # Initialize retrievers
    try:
        text_vs = get_text_vectorstore()
        image_vs = get_image_vectorstore()
        neo4j_graph = get_neo4j_graph()
        kg_retriever = KnowledgeGraphRetriever(neo4j_graph)

        hybrid_retriever = HybridRetriever(
            text_vectorstore=text_vs,
            image_vectorstore=image_vs,
            kg_retriever=kg_retriever,
            text_k=10,
            image_k=5,
        )
    except Exception as e:
        st.error(f"Failed to initialize retrievers: {e}")
        return

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        st.session_state.show_sources = st.checkbox("Show sources", value=True)

        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        example_questions = [
            "Who talked about beach cleanliness?",
            "Tell me about the characters at the beach.",
            "What is happening on page 60?",
            "Explain the main idea of the chilli story.",
        ]
        for q in example_questions:
            if st.button(q, key=q):
                st.session_state.pending_question = q

        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.markdown("### 📊 System Info")
        st.info(
            """
**Data Source:** TN Class 6 English  
(text + images in Chroma & Neo4j)

**Technologies:**
- 🔍 LangChain Hybrid RAG
- 🗄️ Chroma Cloud (1536-dim OpenAI embeddings)
- 🕸️ Neo4j Knowledge Graph
- 🧠 Corrective RAG (critic + verifier)
- 🤖 LLM: GPT-4o-mini
"""
        )

    # Chat container
    chat_container = st.container()

    # History
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                if msg["role"] == "assistant":
                    score = msg.get("confidence_score")
                    if score is not None:
                        score_clamped = max(0, min(100, int(score)))
                        st.caption(f"✅ Confidence score: {score_clamped} / 100")
                        st.progress(score_clamped / 100.0)

                    if msg.get("sources") and st.session_state.show_sources:
                        docs = msg["sources"]
                        text_docs = [
                            d
                            for d in docs
                            if d.metadata.get("source_type") == "text_vector"
                        ]
                        image_docs = [
                            d
                            for d in docs
                            if d.metadata.get("source_type") == "image_vector"
                        ]
                        kg_docs = [
                            d
                            for d in docs
                            if d.metadata.get("source") == "knowledge_graph"
                        ]

                        with st.expander("📖 View Sources"):
                            if text_docs:
                                st.markdown("**📄 Text Chunks:**")
                                for d in text_docs[:5]:
                                    page = d.metadata.get("page", "?")
                                    preview = d.page_content[:150].replace(
                                        "\n", " "
                                    )
                                    st.markdown(f"- Page {page}: {preview}...")

                            if image_docs:
                                st.markdown("**🖼️ Images/Diagrams:**")
                                for d in image_docs:
                                    page = d.metadata.get("page", "?")
                                    preview = d.page_content[:100].replace(
                                        "\n", " "
                                    )
                                    st.markdown(f"- Page {page}: {preview}...")

                            if kg_docs:
                                st.markdown("**🕸️ Knowledge Graph:**")
                                for d in kg_docs:
                                    page = d.metadata.get("page", "?")
                                    t = d.metadata.get("type", "")
                                    preview = d.page_content[:100].replace(
                                        "\n", " "
                                    )
                                    st.markdown(
                                        f"- Page {page} ({t}): {preview}..."
                                    )

    # Input
    if "pending_question" in st.session_state:
        user_input = st.session_state.pending_question
        del st.session_state.pending_question
    else:
        user_input = st.chat_input("Ask me anything about the textbook...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner(
                    "🔍 LangChain Hybrid + Corrective RAG in progress... 🤔"
                ):
                    try:
                        answer, docs, confidence = run_corrective_rag_pipeline(
                            user_input, hybrid_retriever
                        )

                        st.markdown(answer)

                        if confidence is not None:
                            score_clamped = max(0, min(100, int(confidence)))
                            st.caption(f"✅ Confidence score: {score_clamped} / 100")
                            st.progress(score_clamped / 100.0)

                        if st.session_state.show_sources and docs:
                            text_docs = [
                                d
                                for d in docs
                                if d.metadata.get("source_type") == "text_vector"
                            ]
                            image_docs = [
                                d
                                for d in docs
                                if d.metadata.get("source_type") == "image_vector"
                            ]
                            kg_docs = [
                                d
                                for d in docs
                                if d.metadata.get("source") == "knowledge_graph"
                            ]

                            with st.expander("📖 View Sources"):
                                if text_docs:
                                    st.markdown("**📄 Text Chunks:**")
                                    for d in text_docs[:5]:
                                        page = d.metadata.get("page", "?")
                                        preview = d.page_content[:150].replace(
                                            "\n", " "
                                        )
                                        st.markdown(
                                            f"- Page {page}: {preview}..."
                                        )

                                if image_docs:
                                    st.markdown("**🖼️ Images/Diagrams:**")
                                    for d in image_docs:
                                        page = d.metadata.get("page", "?")
                                        preview = d.page_content[:100].replace(
                                            "\n", " "
                                        )
                                        st.markdown(
                                            f"- Page {page}: {preview}..."
                                        )

                                if kg_docs:
                                    st.markdown("**🕸️ Knowledge Graph:**")
                                    for d in kg_docs:
                                        page = d.metadata.get("page", "?")
                                        t = d.metadata.get("type", "")
                                        preview = d.page_content[:100].replace(
                                            "\n", " "
                                        )
                                        st.markdown(
                                            f"- Page {page} ({t}): {preview}..."
                                        )

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": docs,
                                "confidence_score": confidence,
                            }
                        )

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.markdown(
                            "I encountered an error while processing your question. Please try again."
                        )

        st.rerun()

if __name__ == "__main__":
    main()
