import os
import re
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# ==========================================================
# INITIALIZERS (CACHED)
# ==========================================================

@st.cache_resource
def get_embeddings() -> OpenAIEmbeddings:
    """OpenAI embeddings (1536-dim, text-embedding-3-small)."""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY,
    )


@st.cache_resource
def get_llm(temperature: float = 0.5) -> ChatOpenAI:
    """Chat LLM."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        openai_api_key=OPENAI_API_KEY,
    )


@st.cache_resource
def get_chroma_client() -> chromadb.CloudClient:
    """Chroma Cloud client."""
    return chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )


@st.cache_resource
def get_text_vectorstore() -> Chroma:
    """
    Chroma vectorstore for textbook text.
    Expects collection 'tn_class6_text' already populated with:
       - documents: chunk text
       - metadatas: { "page": <int>, ... }
    """
    client = get_chroma_client()
    embeddings = get_embeddings()

    # Ensure collection exists
    _ = client.get_collection("tn_class6_text")

    return Chroma(
        client=client,
        collection_name="tn_class6_text",
        embedding_function=embeddings,
    )


@st.cache_resource
def get_neo4j_graph() -> Optional[Neo4jGraph]:
    """Neo4j graph – returns None if connection fails."""
    if not (NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD):
        return None
    try:
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
        )
        _ = graph.query("RETURN 1 AS ok LIMIT 1")
        return graph
    except Exception as e:
        st.sidebar.warning(f"⚠️ Neo4j not available: {e}")
        return None


# ==========================================================
# PAGE NUMBER PARSER (for page-aware RAG)
# ==========================================================

PAGE_REGEX = re.compile(r"\bpage\s+(\d+)\b", re.IGNORECASE)

def extract_page_number(query: str) -> Optional[int]:
    """
    Look for patterns like 'page 55', 'Page 60' etc.
    Returns page number as int or None.
    NOTE: This uses the PDF page index that you stored in metadata.
    """
    m = PAGE_REGEX.search(query)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


# ==========================================================
# KEYWORD EXTRACTOR (for keyword / multi-query RAG)
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
    "you", "your", "he", "she", "it", "we", "they", "this", "that", "please",
}

def extract_keywords(query: str) -> List[str]:
    """
    Basic keyword extractor that removes stopwords, keeps length > 2.
    This powers "keyword-boosted" RAG.
    """
    tokens = re.findall(r"\w+", query)
    keywords: List[str] = []
    for t in tokens:
        l = t.lower()
        if len(l) > 2 and l not in STOPWORDS:
            keywords.append(t)
    # Deduplicate case-insensitive
    seen = set()
    result: List[str] = []
    for kw in keywords:
        lw = kw.lower()
        if lw not in seen:
            seen.add(lw)
            result.append(kw)
    return result


# ==========================================================
# MULTI-QUERY TEXT RETRIEVER (Normal + Keyword RAG)
# ==========================================================

class MultiQueryRetriever:
    """
    Combines:
      - normal similarity_search(query)
      - similarity_search(keyword) for extracted keywords
    This gives us:
      1) Normal RAG
      2) Keyword-boosted RAG
    """

    def __init__(self, vectorstore: Chroma, k: int = 8):
        self.vectorstore = vectorstore
        self.k = k

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs: List[Document] = []

        # Base query
        try:
            base_docs = self.vectorstore.similarity_search(query, k=self.k)
        except Exception as e:
            st.warning(f"Vector search error: {e}")
            return []
        docs.extend(base_docs)

        seen = {(d.metadata.get("id"), d.metadata.get("page")) for d in docs}

        # Keyword queries
        for kw in extract_keywords(query):
            try:
                kw_docs = self.vectorstore.similarity_search(kw, k=self.k)
            except Exception:
                continue

            for d in kw_docs:
                key = (d.metadata.get("id"), d.metadata.get("page"))
                if key not in seen:
                    docs.append(d)
                    seen.add(key)

        return docs


# ==========================================================
# KNOWLEDGE GRAPH RETRIEVER (Entity / Relation RAG)
# ==========================================================

class KnowledgeGraphRetriever:
    """
    Simple keyword-based retriever over Neo4j TextChunk nodes.
    This gives KG-based RAG.
    """

    def __init__(self, graph: Optional[Neo4jGraph]):
        self.graph = graph

    def get_relevant_documents(self, query: str, top_k: int = 10) -> List[Document]:
        if self.graph is None:
            return []

        keywords = extract_keywords(query)
        if not keywords:
            return []

        cypher = """
        MATCH (t:TextChunk)
        WHERE any(kw IN $keywords WHERE toLower(t.text) CONTAINS toLower(kw))
        RETURN t.id AS id, t.text AS text, t.page AS page
        ORDER BY page ASC
        LIMIT $top_k
        """

        try:
            rows = self.graph.query(cypher, params={"keywords": keywords, "top_k": top_k})
        except Exception as e:
            st.warning(f"Neo4j query error: {e}")
            return []

        docs: List[Document] = []
        for r in rows:
            docs.append(
                Document(
                    page_content=r.get("text", ""),
                    metadata={
                        "id": r.get("id"),
                        "page": r.get("page"),
                        "source_type": "kg",
                    },
                )
            )
        return docs

    def get_page_documents(self, page: int) -> List[Document]:
        """Specifically fetch all TextChunk nodes for a single page."""
        if self.graph is None:
            return []

        cypher = """
        MATCH (t:TextChunk {page: $page})
        RETURN t.id AS id, t.text AS text, t.page AS page
        ORDER BY id ASC
        """
        try:
            rows = self.graph.query(cypher, params={"page": page})
        except Exception:
            return []

        docs: List[Document] = []
        for r in rows:
            docs.append(
                Document(
                    page_content=r.get("text", ""),
                    metadata={
                        "id": r.get("id"),
                        "page": r.get("page"),
                        "source_type": "kg_page",
                    },
                )
            )
        return docs


# ==========================================================
# PAGE-AWARE HYBRID RETRIEVER
# ==========================================================

class PageAwareHybridRetriever:
    """
    Uses all of:
      - Page-aware direct lookup in Chroma & KG (if 'page N' in query)
      - Normal vector RAG + keyword RAG (MultiQueryRetriever)
      - KG keyword RAG

    This is the main retriever used by the Corrective RAG pipeline.
    """

    def __init__(
        self,
        text_vectorstore: Chroma,
        kg_retriever: KnowledgeGraphRetriever,
        chroma_client: chromadb.CloudClient,
        text_k: int = 10,
    ):
        self.text_retriever = MultiQueryRetriever(text_vectorstore, k=text_k)
        self.kg_retriever = kg_retriever
        self.client = chroma_client

    def _get_page_docs_from_chroma(self, page: int) -> List[Document]:
        try:
            col = self.client.get_collection("tn_class6_text")
            res = col.get(
                where={"page": page}, 
                include=["documents", "metadatas"]
            )
        except Exception as e:
            st.warning(f"Chroma page fetch error: {e}")
            return []

        docs: List[Document] = []
        docs_list = res.get("documents") or []
        metas_list = res.get("metadatas") or []

        for doc_text, meta in zip(docs_list, metas_list):
            md = meta or {}
            md["source_type"] = "text_page_direct"
            docs.append(Document(page_content=doc_text, metadata=md))

        return docs

    def get_relevant_documents(self, query: str) -> List[Document]:
        page_no = extract_page_number(query)

        # If query mentions a page explicitly → page-aware override
        if page_no is not None:
            docs: List[Document] = []
            # 1) direct Chroma page fetch
            page_docs = self._get_page_docs_from_chroma(page_no)
            docs.extend(page_docs)

            # 2) KG page fetch (if present)
            kg_page_docs = self.kg_retriever.get_page_documents(page_no)
            docs.extend(kg_page_docs)

            if docs:
                return docs
            # If nothing found for that page, fall back to normal hybrid

        # Normal hybrid retrieval
        docs: List[Document] = []

        # Text vectors (normal + keyword)
        text_docs = self.text_retriever.get_relevant_documents(query)
        for d in text_docs:
            d.metadata["source_type"] = d.metadata.get("source_type", "text_vector")
            docs.append(d)

        # KG docs
        kg_docs = self.kg_retriever.get_relevant_documents(query)
        for d in kg_docs:
            d.metadata["source_type"] = d.metadata.get("source_type", "kg")
            docs.append(d)

        return docs


# ==========================================================
# ANSWER GENERATION – NORMAL RAG ONLY (for comparison)
# ==========================================================

def format_docs_simple(docs: List[Document]) -> str:
    out = []
    for d in docs:
        page = d.metadata.get("page", "?")
        out.append(f"[Page {page}] {d.page_content}")
    return "\n\n".join(out)


def normal_rag_answer(query: str, text_vs: Chroma) -> Tuple[str, List[Document]]:
    """Plain vector-only RAG (no KG, no page override, no critic)."""
    try:
        docs = text_vs.similarity_search(query, k=8)
    except Exception as e:
        return f"Normal RAG error: {e}", []

    if not docs:
        return "Normal RAG could not find relevant text.", []

    ctx = format_docs_simple(docs)[:4000]

    llm = get_llm(temperature=0.6)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful tutor. Use ONLY the given context from the textbook to answer.",
            ),
            (
                "user",
                "CONTEXT:\n{context}\n\nQUESTION: {query}\n\nGive a clear, student-friendly answer.",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    ans = chain.invoke({"context": ctx, "query": query})
    return ans, docs


# ==========================================================
# ANSWER GENERATION – KG-ONLY RAG (for comparison)
# ==========================================================

def kg_rag_answer(query: str, kg_retriever: KnowledgeGraphRetriever) -> Tuple[str, List[Document]]:
    docs = kg_retriever.get_relevant_documents(query, top_k=10)
    if not docs:
        return "KG RAG could not find any relevant nodes for this question.", []

    ctx = format_docs_simple(docs)[:4000]
    llm = get_llm(temperature=0.6)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful tutor using ONLY the context retrieved from a Knowledge Graph of the textbook.",
            ),
            (
                "user",
                "CONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer clearly using only this context.",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    ans = chain.invoke({"context": ctx, "query": query})
    return ans, docs


# ==========================================================
# CORRECTIVE RAG – CRITIC + VERIFIER (MAIN ANSWER)
# ==========================================================

def format_docs_for_context(docs: List[Document]) -> str:
    parts: List[str] = []
    for d in docs:
        page = d.metadata.get("page", "?")
        src = d.metadata.get("source_type", "unknown")
        parts.append(f"[Page {page} | {src}] {d.page_content}")
    return "\n\n".join(parts)


def create_retrieval_critic_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a retrieval quality critic for a textbook RAG system.
You are given a QUESTION and the RETRIEVED CONTEXT.

Your job:
1. Judge how well this context can answer the question.
2. If the context is weak or missing key info, propose a BETTER_SEARCH_QUERY.

Output strictly in this format:
ADEQUACY: HIGH/MEDIUM/LOW
BETTER_QUERY: <new query or NONE>""",
            ),
            (
                "user",
                "QUESTION:\n{query}\n\nRETRIEVED CONTEXT:\n{context}",
            ),
        ]
    )
    llm = get_llm(temperature=0.2)
    return prompt | llm | StrOutputParser()


def create_answer_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful tutor for the TN Class 6 English textbook.
Use ONLY the provided context from the textbook to answer.
If the question asks about a specific page number, focus on that page.
If there is not enough information, say that clearly.""",
            ),
            (
                "user",
                "CONTEXT:\n{context}\n\nQUESTION: {query}\n\nGive a clear, student-friendly answer.",
            ),
        ]
    )
    llm = get_llm(temperature=0.6)
    return prompt | llm | StrOutputParser()


def create_verification_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a STRICT verifier for a textbook-based tutor.
You will be given a QUESTION, CONTEXT, and a DRAFT ANSWER.

Your job:
1. Check if each important claim in the DRAFT ANSWER is supported by the CONTEXT.
2. Remove or fix any unsupported or contradictory claims.
3. If the context is incomplete, say so clearly.

At the end, include a line:
ADEQUACY: HIGH/MEDIUM/LOW""",
            ),
            (
                "user",
                "QUESTION:\n{query}\n\nCONTEXT:\n{context}\n\nDRAFT ANSWER:\n{draft_answer}\n\nNow produce the corrected final answer, then the ADEQUACY line.",
            ),
        ]
    )
    llm = get_llm(temperature=0.3)
    return prompt | llm | StrOutputParser()


def compute_confidence(
    docs: List[Document],
    answer_adequacy: str = "MEDIUM",
    retrieval_adequacy: str = "MEDIUM",
) -> int:
    if not docs:
        return 5

    def adequacy_to_score(a: str) -> float:
        a = (a or "").upper()
        if a == "HIGH":
            return 1.0
        if a == "LOW":
            return 0.3
        return 0.65

    ans_score = adequacy_to_score(answer_adequacy)
    ret_score = adequacy_to_score(retrieval_adequacy)

    total_len = sum(len(d.page_content) for d in docs)
    if total_len >= 1500:
        len_score = 1.0
    elif total_len >= 800:
        len_score = 0.8
    elif total_len >= 400:
        len_score = 0.65
    else:
        len_score = 0.45

    num_docs = len(docs)
    if num_docs >= 10:
        count_score = 1.0
    elif num_docs >= 6:
        count_score = 0.85
    elif num_docs >= 3:
        count_score = 0.7
    else:
        count_score = 0.5

    combined = 0.3 * ans_score + 0.25 * ret_score + 0.25 * len_score + 0.2 * count_score
    score = int(round(max(0.0, min(1.0, combined)) * 100))

    if answer_adequacy.upper() == "LOW":
        score = min(score, 55)
    return score


def run_corrective_rag_pipeline(
    query: str, retriever: PageAwareHybridRetriever
) -> Tuple[str, List[Document], int]:
    # Step 1: initial retrieval
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return (
            "I couldn't find any relevant information in the indexed pages for this question.",
            [],
            10,
        )

    # Step 2: critic for retrieval adequacy
    critic_chain = create_retrieval_critic_chain()
    preview_ctx = format_docs_for_context(docs[:8])[:2000]
    critic_raw = critic_chain.invoke({"query": query, "context": preview_ctx})

    retrieval_adequacy = "MEDIUM"
    better_query: Optional[str] = None
    for line in critic_raw.splitlines():
        line = line.strip()
        if line.upper().startswith("ADEQUACY:"):
            retrieval_adequacy = line.split(":", 1)[1].strip().upper()
        elif line.upper().startswith("BETTER_QUERY:"):
            candidate = line.split(":", 1)[1].strip()
            if candidate and candidate.upper() != "NONE":
                better_query = candidate

    # Step 3: optional re-retrieval
    if retrieval_adequacy == "LOW" and better_query:
        improved_docs = retriever.get_relevant_documents(better_query)
        seen = {(d.metadata.get("id"), d.metadata.get("page")) for d in docs}
        for d in improved_docs:
            key = (d.metadata.get("id"), d.metadata.get("page"))
            if key not in seen:
                docs.append(d)
                seen.add(key)

    # Step 4: draft answer
    full_ctx = format_docs_for_context(docs)[:6000]
    answer_chain = create_answer_chain()
    draft = answer_chain.invoke({"query": query, "context": full_ctx})

    # Step 5: verification
    verify_chain = create_verification_chain()
    verified = verify_chain.invoke(
        {"query": query, "context": full_ctx, "draft_answer": draft}
    )

    answer_adequacy = "MEDIUM"
    if "ADEQUACY:" in verified:
        body, marker = verified.rsplit("ADEQUACY:", 1)
        final_answer = body.strip()
        answer_adequacy = marker.strip().upper()
    else:
        final_answer = verified.strip()

    # Step 6: confidence
    conf = compute_confidence(docs, answer_adequacy, retrieval_adequacy)
    return final_answer, docs, conf


# ==========================================================
# STREAMLIT APP
# ==========================================================

def main():
    st.set_page_config(
        page_title="TN Class 6 English – Multi-RAG Tutor",
        page_icon="📚",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.3rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 0.3rem;
        }
        .sub-header {
            text-align: center;
            color: #666;
            margin-bottom: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-header">📚 TN Class 6 English – Multi-RAG Tutor</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Normal RAG + Keyword RAG + Page-aware + KG RAG + Corrective RAG</div>',
        unsafe_allow_html=True,
    )

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True

    # Init components
    try:
        text_vs = get_text_vectorstore()
        graph = get_neo4j_graph()
        kg_retriever = KnowledgeGraphRetriever(graph)
        chroma_client = get_chroma_client()
        hybrid_retriever = PageAwareHybridRetriever(
            text_vectorstore=text_vs,
            kg_retriever=kg_retriever,
            chroma_client=chroma_client,
            text_k=10,
        )
    except Exception as e:
        st.error(f"Failed to initialize RAG components: {e}")
        return

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        st.session_state.show_sources = st.checkbox("Show sources", value=True)

        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        examples = [
            "Who went to the beach?",
            "Tell me about the characters at the beach.",
            "Summarize page 55.",
            "What is happening on page 60?",
            "Explain the main idea of the chilli story.",
        ]
        for i, q in enumerate(examples):
            if st.button(q, key=f"ex_{i}"):
                st.session_state.pending_question = q

        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.markdown("### 📊 RAG Modes Used")
        st.info(
            """
**This tutor uses:**
- 🔹 **Normal RAG** – plain vector search (Chroma)
- 🔹 **Keyword RAG** – multi-query over names / key terms
- 🔹 **Page-aware RAG** – detects "page N" and focuses that page
- 🔹 **KG RAG** – queries Neo4j `TextChunk` nodes
- 🔹 **Corrective RAG** – critic + verifier pipeline for the main answer
"""
        )

    # Chat container
    chat_container = st.container()

    # Show history
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                if msg["role"] == "assistant":
                    conf = msg.get("confidence_score")
                    if conf is not None:
                        c = max(0, min(100, int(conf)))
                        st.caption(f"✅ Main Hybrid+Corrective RAG confidence: {c}/100")
                        st.progress(c / 100.0)

                    if st.session_state.show_sources and msg.get("sources"):
                        docs = msg["sources"]
                        with st.expander("📖 View Main RAG Sources"):
                            text_docs = [
                                d for d in docs
                                if d.metadata.get("source_type", "").startswith("text")
                            ]
                            kg_docs = [
                                d for d in docs
                                if d.metadata.get("source_type", "").startswith("kg")
                            ]

                            if text_docs:
                                st.markdown("**📄 Text Chunks:**")
                                for d in text_docs[:6]:
                                    page = d.metadata.get("page", "?")
                                    preview = d.page_content[:200].replace("\n", " ")
                                    st.markdown(f"- Page {page}: {preview}...")

                            if kg_docs:
                                st.markdown("**🕸️ KG Chunks:**")
                                for d in kg_docs[:4]:
                                    page = d.metadata.get("page", "?")
                                    preview = d.page_content[:160].replace("\n", " ")
                                    st.markdown(f"- Page {page}: {preview}...")

    # Input
    if "pending_question" in st.session_state:
        user_query = st.session_state.pending_question
        del st.session_state.pending_question
    else:
        user_query = st.chat_input("Ask me anything about the textbook...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Running all RAG pipelines (Normal + KG + Hybrid Corrective)…"):
                    try:
                        # 1) Main hybrid + corrective RAG
                        main_answer, main_docs, main_conf = run_corrective_rag_pipeline(
                            user_query, hybrid_retriever
                        )

                        st.markdown("### ✅ Main Tutor Answer (Hybrid + Corrective RAG)")
                        st.markdown(main_answer)
                        c = max(0, min(100, int(main_conf)))
                        st.caption(f"Main RAG confidence: {c}/100")
                        st.progress(c / 100.0)

                        # 2) Normal RAG only
                        norm_answer, norm_docs = normal_rag_answer(user_query, text_vs)
                        st.markdown("---")
                        st.markdown("### 📘 Normal RAG (Vector only)")
                        st.markdown(norm_answer)

                        # 3) KG-only RAG
                        kg_answer, kg_docs = kg_rag_answer(user_query, kg_retriever)
                        st.markdown("---")
                        st.markdown("### 🕸️ KG-only RAG")
                        st.markdown(kg_answer)

                        # Save to history (only main answer + sources)
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": main_answer,
                                "sources": main_docs,
                                "confidence_score": main_conf,
                                "normal_rag_answer": norm_answer,
                                "kg_rag_answer": kg_answer,
                            }
                        )

                    except Exception as e:
                        st.error(f"Error while answering: {e}")
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": "I hit an error while processing your question. Please try again.",
                            }
                        )

        st.rerun()


if __name__ == "__main__":
    main()
