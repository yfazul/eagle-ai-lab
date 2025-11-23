import os
import re
from typing import List, Tuple, Optional

import streamlit as st
from dotenv import load_dotenv

import chromadb
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =========================================================
# ENV + CONFIG
# =========================================================
load_dotenv()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
CHROMA_API_KEY   = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT    = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE  = os.getenv("CHROMA_DATABASE")
NEO4J_URI        = os.getenv("NEO4J_URI")
NEO4J_USERNAME   = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD   = os.getenv("NEO4J_PASSWORD")

# printed-page numbering already handled during ingestion:
# text metadata: {"page": <printed_page>, "pdf_page": <pdf_page>}
# image captions in collection "tn_page64_images"


# =========================================================
# CACHED CLIENTS
# =========================================================

@st.cache_resource
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-small",   # 1536-dim
        openai_api_key=OPENAI_API_KEY,
    )


@st.cache_resource
def get_llm(temp: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temp,
        openai_api_key=OPENAI_API_KEY,
    )


@st.cache_resource
def get_chroma_client() -> chromadb.CloudClient:
    return chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )


@st.cache_resource
def get_text_vectorstore() -> Chroma:
    client = get_chroma_client()
    # collection "tn_class6_text" already created in ingestion
    _ = client.get_collection("tn_class6_text")

    return Chroma(
        client=client,
        collection_name="tn_class6_text",
        embedding_function=get_embeddings(),
    )


@st.cache_resource
def get_text_collection():
    """
    Raw Chroma collection for textbook text.
    """
    client = get_chroma_client()
    return client.get_collection("tn_class6_text")


@st.cache_resource
def get_image_collection():
    """
    Raw Chroma collection for page-64 image captions.
    Ingest script created 'tn_page64_images' with 11 images.
    """
    client = get_chroma_client()
    return client.get_collection("tn_page64_images")


@st.cache_resource
def get_graph() -> Optional[Neo4jGraph]:
    try:
        g = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
        )
        g.query("RETURN 1 AS ok LIMIT 1")
        return g
    except Exception as e:
        st.sidebar.warning(f"⚠️ Neo4j unavailable: {e}")
        return None


# =========================================================
# HELPERS: PAGE PARSING + KEYWORDS
# =========================================================

PAGE_REGEX = re.compile(r"\bpage\s+(\d+)\b", re.IGNORECASE)

def extract_page_number(q: str) -> Optional[int]:
    m = PAGE_REGEX.search(q)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


STOPWORDS = {
    "what","who","whom","whose","which","where","when","why","how",
    "tell","about","explain","describe","show","give","find","get",
    "help","need","want","like","know","think","say","said","do",
    "does","did","done","doing","is","are","was","were","be","been",
    "being","has","have","had","having","can","could","may","might",
    "must","shall","should","will","would","the","a","an","and","or",
    "but","if","because","as","until","while","of","at","by","for",
    "with","from","up","on","off","out","in","over","under","me",
    "you","your","he","she","it","we","they","this","that","please"
}

def extract_keywords(q: str) -> List[str]:
    tokens = re.findall(r"\w+", q)
    kws: List[str] = []
    for t in tokens:
        lower = t.lower()
        if len(lower) > 2 and lower not in STOPWORDS:
            kws.append(t)
    # de-dup case-insensitive
    seen = set()
    result: List[str] = []
    for kw in kws:
        l = kw.lower()
        if l not in seen:
            seen.add(l)
            result.append(kw)
    return result


# =========================================================
# 1️⃣ CHROMA HELPERS (TEXT + PAGE 64 IMAGES)
# =========================================================

def chroma_page_fetch(printed_page: int) -> List[Document]:
    """
    Get all chunks where metadata.page == printed_page.
    Uses raw Chroma collection filter.
    """
    col = get_text_collection()
    try:
        res = col.get(
            where={"page": printed_page},
            include=["documents", "metadatas"],
        )
    except Exception as e:
        st.warning(f"Chroma page fetch error: {e}")
        return []

    docs: List[Document] = []
    docs_list = res.get("documents") or []
    metas_list = res.get("metadatas") or []

    for text, meta in zip(docs_list, metas_list):
        md = meta or {}
        md["source_type"] = "text_page"
        docs.append(Document(page_content=text, metadata=md))

    return docs


def chroma_vector_search(query: str, k: int = 6) -> List[Document]:
    """
    Normal text-vector search on tn_class6_text.
    """
    vs = get_text_vectorstore()
    try:
        docs = vs.similarity_search(query, k=k)
    except Exception as e:
        st.warning(f"Chroma similarity_search error: {e}")
        return []

    for d in docs:
        d.metadata["source_type"] = d.metadata.get("source_type", "text_vector")
    return docs


def get_page64_image_docs(query: str, k: int = 11) -> List[Document]:
    """
    Semantic search over page-64 image captions in Chroma collection 'tn_page64_images'.
    Returns up to k image-caption Documents.
    """
    try:
        col = get_image_collection()
    except Exception as e:
        # If image collection does not exist, just skip images
        st.warning(f"Page-64 image collection not available: {e}")
        return []

    # Use same embedding model as ingestion (text-embedding-3-small, 1536-dim)
    emb = get_embeddings()
    try:
        vec = emb.embed_query(query)
    except Exception as e:
        st.warning(f"Embedding error for image search: {e}")
        return []

    try:
        res = col.query(
            query_embeddings=[vec],
            n_results=k,
            include=["documents", "metadatas"],
        )
    except Exception as e:
        st.warning(f"Chroma image query error: {e}")
        return []

    docs_out: List[Document] = []
    docs_list = res.get("documents") or []
    metas_list = res.get("metadatas") or []

    if not docs_list:
        return []

    for text, meta in zip(docs_list[0], metas_list[0]):
        md = meta or {}
        md.setdefault("page", 64)
        md["source_type"] = md.get("source_type", "image_page64")
        docs_out.append(Document(page_content=text, metadata=md))

    return docs_out


# =========================================================
# 2️⃣ KG RETRIEVAL (Neo4j)
# =========================================================

def kg_keyword_search(query: str, top_k: int = 8) -> List[Document]:
    graph = get_graph()
    if graph is None:
        return []

    kws = extract_keywords(query)
    if not kws:
        return []

    cypher = """
    MATCH (t:TextChunk)
    WHERE any(kw IN $kws WHERE toLower(t.text) CONTAINS toLower(kw))
    RETURN t.id AS id, t.text AS text, t.page AS page
    LIMIT $top_k
    """

    try:
        rows = graph.query(cypher, params={"kws": [k.lower() for k in kws], "top_k": top_k})
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


def kg_page_search(printed_page: int) -> List[Document]:
    graph = get_graph()
    if graph is None:
        return []

    cypher = """
    MATCH (t:TextChunk {page: $page})
    RETURN t.id AS id, t.text AS text, t.page AS page
    ORDER BY id
    """

    try:
        rows = graph.query(cypher, params={"page": printed_page})
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


# =========================================================
# 3️⃣ GENERIC ANSWER GENERATOR
# =========================================================

def format_docs(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        page = d.metadata.get("page", "?")
        src  = d.metadata.get("source_type", "text")
        parts.append(f"[Page {page} | {src}] {d.page_content}")
    return "\n\n".join(parts)


def generate_answer(query: str, docs: List[Document], mode_name: str) -> Tuple[str, int]:
    """
    Common answer generator for all modes.
    """
    if not docs:
        return (
            f"I couldn't find relevant content for this question in **{mode_name}** mode. "
            "It may not be covered in the indexed pages.",
            35,
        )

    context = format_docs(docs)[:6500]

    # Stricter system prompt (no guessing)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Act as a careful English tutor for the TN Class 6 English textbook.\n"
                "You MUST answer **only** using the information in the supplied CONTEXT.\n"
                "Do not add outside knowledge or guess. If the CONTEXT does not contain\n"
                "enough information to answer, clearly say so.\n\n"
                f"This answer is produced using mode: {mode_name}."
            ),
            (
                "user",
                "CONTEXT:\n{context}\n\nQUESTION: {query}\n\n"
                "Give a clear, student-friendly answer."
            ),
        ]
    )

    chain = prompt | get_llm(0.3) | StrOutputParser()
    ans = chain.invoke({"context": context, "query": query})

    # simple confidence heuristic
    n = len(docs)
    total_len = sum(len(d.page_content) for d in docs)
    if n >= 10 and total_len > 2000:
        conf = 90
    elif n >= 5 and total_len > 900:
        conf = 80
    else:
        conf = 65
    return ans, conf


# =========================================================
# STREAMLIT APP (3 MODES)
# =========================================================

def main():
    st.set_page_config(
        page_title="TN Class 6 English – RAG Tutor",
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
            margin-bottom: 0.2rem;
        }
        .sub-header {
            text-align: center;
            color: #666;
            margin-bottom: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-header">📚 TN Class 6 English – RAG Tutor</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Modes: Normal • Page-Aware • KG-only (with page-64 images)</div>',
        unsafe_allow_html=True,
    )

    # session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # sidebar
    with st.sidebar:
        st.title("⚙️ Settings")

        st.markdown("### 💡 Example Questions")
        examples = [
            "Who talked about beach cleanliness?",
            "Tell me about the characters at the beach.",
            "Summarize page 134.",
            "Summarize page 135.",
            "What objects are shown in the pictures on page 64?",
        ]
        for i, q in enumerate(examples):
            if st.button(q, key=f"ex_{i}"):
                st.session_state.pending_question = q

        st.markdown("---")
        if st.button("🗑 Clear chat"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.info(
            """
**Backends used:**
- Chroma Cloud (`tn_class6_text`, `tn_page64_images`)
- Neo4j `TextChunk` nodes
- OpenAI `text-embedding-3-small` (1536-dim)
- GPT-4o-mini for answers

**Image support:**
- Page-64 images are ingested as captions in `tn_page64_images`
- All modes can see page-64 image captions (especially questions about page 64 or picture objects)
"""
        )

    # show history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # input
    if "pending_question" in st.session_state:
        user_q = st.session_state.pending_question
        del st.session_state.pending_question
    else:
        user_q = st.chat_input("Ask about the textbook (you can also say 'page 134', or ask about the pictures on page 64)")

    if not user_q:
        return

    # append user msg
    st.session_state.messages.append({"role": "user", "content": user_q})
    st.chat_message("user").markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Running 3 RAG modes (Normal, Page-Aware, KG-only)…"):

            # ---- 1) Normal RAG: Text vectors + page-64 images ----
            normal_text_docs = chroma_vector_search(user_q, k=8)
            normal_img_docs  = get_page64_image_docs(user_q, k=11)
            normal_docs = normal_text_docs + normal_img_docs
            normal_ans, normal_conf = generate_answer(user_q, normal_docs, "Normal-RAG (Text + Page-64 Images)")

            # ---- 2) Page-Aware RAG ----
            page_no = extract_page_number(user_q)
            if page_no is not None:
                page_text_docs = chroma_page_fetch(page_no)
                # fallback if no direct page match
                if not page_text_docs:
                    page_text_docs = chroma_vector_search(user_q, k=8)
            else:
                page_text_docs = chroma_vector_search(user_q, k=8)

            page_img_docs = []
            # If printed page 64, strongly pull in all page-64 images as context
            if page_no == 64:
                page_img_docs = get_page64_image_docs(user_q, k=11)
                # if someone literally asks "how many images on page 64"
                # we want all 11 captions available in context

            page_docs = page_text_docs + page_img_docs
            page_ans, page_conf = generate_answer(
                user_q,
                page_docs,
                "Page-Aware-RAG (Page + Images)",
            )

            # ---- 3) KG-only RAG ----
            kg_text_docs = kg_keyword_search(user_q, top_k=10)
            # Optional: also allow image captions to appear in KG tab
            kg_img_docs = get_page64_image_docs(user_q, k=11)
            kg_docs = kg_text_docs + kg_img_docs
            kg_ans, kg_conf = generate_answer(
                user_q,
                kg_docs,
                "KG-only-RAG (Graph + Images)",
            )

            # ---- Show in tabs ----
            tabs = st.tabs(
                [
                    "1️⃣ Normal RAG",
                    "2️⃣ Page-Aware RAG",
                    "3️⃣ KG-only RAG",
                ]
            )

            # Normal
            with tabs[0]:
                st.markdown(normal_ans)
                st.caption(f"Confidence: {normal_conf}/100")
                if normal_docs:
                    with st.expander("Sources (Normal)"):
                        text_srcs = [d for d in normal_docs if str(d.metadata.get("source_type", "")).startswith("text")]
                        img_srcs  = [d for d in normal_docs if "image" in str(d.metadata.get("source_type", ""))]
                        if text_srcs:
                            st.markdown("**📄 Text Chunks:**")
                            for d in text_srcs[:8]:
                                page = d.metadata.get("page", "?")
                                prev = d.page_content[:200].replace("\n", " ")
                                st.markdown(f"- Page {page}: {prev}...")
                        if img_srcs:
                            st.markdown("**🖼 Page-64 Image Captions:**")
                            for d in img_srcs[:11]:
                                page = d.metadata.get("page", "?")
                                prev = d.page_content[:200].replace("\n", " ")
                                st.markdown(f"- Page {page} (image): {prev}...")

            # Page-aware
            with tabs[1]:
                if page_no is not None:
                    st.markdown(f"*(Page-aware mode focused on printed page **{page_no}**.)*")
                st.markdown(page_ans)
                st.caption(f"Confidence: {page_conf}/100")
                if page_docs:
                    with st.expander("Sources (Page-Aware)"):
                        text_srcs = [d for d in page_docs if str(d.metadata.get("source_type", "")).startswith("text")]
                        img_srcs  = [d for d in page_docs if "image" in str(d.metadata.get("source_type", ""))]
                        if text_srcs:
                            st.markdown("**📄 Text Chunks:**")
                            for d in text_srcs[:8]:
                                page = d.metadata.get("page", "?")
                                prev = d.page_content[:200].replace("\n", " ")
                                st.markdown(f"- Page {page}: {prev}...")
                        if img_srcs:
                            st.markdown("**🖼 Page-64 Image Captions:**")
                            for d in img_srcs[:11]:
                                page = d.metadata.get("page", "?")
                                prev = d.page_content[:200].replace("\n", " ")
                                st.markdown(f"- Page {page} (image): {prev}...")

            # KG-only
            with tabs[2]:
                st.markdown(kg_ans)
                st.caption(f"Confidence: {kg_conf}/100")
                if kg_docs:
                    with st.expander("Sources (KG-only)"):
                        text_srcs = [d for d in kg_docs if "kg" in str(d.metadata.get("source_type", ""))]
                        img_srcs  = [d for d in kg_docs if "image" in str(d.metadata.get("source_type", ""))]
                        if text_srcs:
                            st.markdown("**🕸 KG Text Chunks:**")
                            for d in text_srcs[:8]:
                                page = d.metadata.get("page", "?")
                                prev = d.page_content[:200].replace("\n", " ")
                                st.markdown(f"- Page {page}: {prev}...")
                        if img_srcs:
                            st.markdown("**🖼 Page-64 Image Captions:**")
                            for d in img_srcs[:11]:
                                page = d.metadata.get("page", "?")
                                prev = d.page_content[:200].replace("\n", " ")
                                st.markdown(f"- Page {page} (image): {prev}...")

        # store a compact summary into chat history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "### Summary (Normal / Page-Aware / KG-only)\n"
                    + normal_ans
                    + f"\n\n_(Normal RAG confidence: {normal_conf}/100)_"
                ),
            }
        )


if __name__ == "__main__":
    main()
