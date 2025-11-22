import os
import streamlit as st
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from neo4j import GraphDatabase

# ==========================================================
# CONFIGURATION
# ==========================================================

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Chroma Cloud
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ==========================================================
# INITIALIZE CONNECTIONS
# ==========================================================

@st.cache_resource
def get_chroma_client():
    """Initialize and cache Chroma Cloud client"""
    return chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )


@st.cache_resource
def get_neo4j_driver():
    """Initialize and cache Neo4j driver"""
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )


@st.cache_resource
def get_collections():
    """Get Chroma collections"""
    client = get_chroma_client()
    text_coll = client.get_collection("tn_class6_text")
    image_coll = client.get_collection("tn_class6_images")
    return text_coll, image_coll

# ==========================================================
# EMBEDDING FUNCTION
# ==========================================================

def embed_query(text: str) -> List[float]:
    """Create embedding for query using OpenAI"""
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text.strip(),
        dimensions=384
    )
    return resp.data[0].embedding

# ==========================================================
# BASE RAG RETRIEVAL FUNCTIONS (VECTOR + KG)
# ==========================================================

def retrieve_text_chunks_raw(
    query_embedding: List[float],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Internal: retrieve text with precomputed embedding"""
    text_coll, _ = get_collections()
    results = text_coll.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    if not results["ids"]:
        return chunks

    for i in range(len(results["ids"][0])):
        chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "page": results["metadatas"][0][i].get("page", 0),
            "distance": results["distances"][0][i],
        })
    return chunks


def retrieve_image_chunks_raw(
    query_embedding: List[float],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """Internal: retrieve image OCR with precomputed embedding"""
    _, image_coll = get_collections()
    results = image_coll.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    images = []
    if not results["ids"]:
        return images

    for i in range(len(results["ids"][0])):
        ocr_text = results["documents"][0][i]
        metadata = results["metadatas"][0][i]

        if ocr_text and len(ocr_text.strip()) > 0:
            images.append({
                "id": results["ids"][0][i],
                "ocr_text": ocr_text,
                "page": metadata.get("page", 0),
                "image_path": metadata.get("image_path", ""),
                "distance": results["distances"][0][i],
            })
    return images


def retrieve_text_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Semantic retrieval for text chunks"""
    emb = embed_query(query)
    return retrieve_text_chunks_raw(emb, top_k=top_k)


def retrieve_image_chunks(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Semantic retrieval for image OCR chunks"""
    emb = embed_query(query)
    return retrieve_image_chunks_raw(emb, top_k=top_k)


def retrieve_from_knowledge_graph(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve context from Neo4j Knowledge Graph (simple keyword search)"""
    driver = get_neo4j_driver()

    # Very simple keyword extraction for KG
    tokens = query.replace("?", " ").replace(".", " ").split()
    keywords = [word.lower() for word in tokens if len(word) > 3]

    if not keywords:
        return []

    cypher_query = """
    // Find text chunks matching keywords
    MATCH (t:TextChunk)
    WHERE any(keyword IN $keywords WHERE toLower(t.text) CONTAINS keyword)
    RETURN t.id as id, t.text as text, t.page as page, 'text' as type
    ORDER BY t.page
    LIMIT $top_k

    UNION

    // Find image chunks with OCR text matching keywords
    MATCH (i:ImageChunk)
    WHERE i.has_text = true
      AND any(keyword IN $keywords WHERE toLower(i.ocr_text) CONTAINS keyword)
    RETURN i.id as id, i.ocr_text as text, i.page as page, 'image' as type
    ORDER BY i.page
    LIMIT 3
    """

    results = []
    with driver.session() as session:
        records = session.run(cypher_query, keywords=keywords, top_k=top_k)
        for record in records:
            results.append({
                "id": record["id"],
                "text": record["text"],
                "page": record["page"],
                "type": record["type"],
            })

    return results

# ==========================================================
# KEYWORD-AWARE SEMANTIC BOOST
# ==========================================================

STOPWORDS = {
    "what", "who", "whom", "whose", "which", "where", "when", "why", "how",
    "tell", "about", "explain", "describe", "show", "give", "find", "get",
    "help", "need", "want", "like", "know", "think", "say", "said",
    "do", "does", "did", "done", "doing",
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "having",
    "can", "could", "may", "might", "must", "shall", "should", "will", "would",
    "me", "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "we", "us", "our", "ours", "they", "them", "their", "theirs",
    "this", "that", "these", "those",
    "the", "a", "an", "and", "or", "but", "if", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "from", "up", "on", "off",
    "out", "in", "over", "under", "again", "further", "then", "once",
    "please", "thanks", "thank", "sorry",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
}

def extract_keywords(query: str) -> List[str]:
    """
    Extract potentially important keywords from the query.
    Keeps words >2 chars and not in the stopword list.
    Also preserves capitalized words (likely proper nouns).
    """
    kws: List[str] = []
    tokens = query.strip().replace("?", " ").replace(".", " ").split()

    for token in tokens:
        original = token.strip().strip(".,!?\"'()")
        if not original:
            continue
        clean = original.lower()

        if (len(clean) > 2 and clean not in STOPWORDS) or original[0].isupper():
            kws.append(original)

    # Remove duplicates, preserve order
    seen = set()
    result = []
    for k in kws:
        if k not in seen:
            seen.add(k)
            result.append(k)
    return result


def keyword_augmented_retrieval(query: str) -> Dict[str, Any]:
    """
    1. Retrieve using the full query.
    2. Also retrieve using each extracted keyword separately
       and merge results (good for names like 'Lilly').
    """
    base_embedding = embed_query(query)

    text_chunks = retrieve_text_chunks_raw(base_embedding, top_k=5)
    image_chunks = retrieve_image_chunks_raw(base_embedding, top_k=3)

    keywords = extract_keywords(query)

    existing_text_keys = {(c["id"], c["page"]) for c in text_chunks}
    existing_img_keys = {(c["id"], c["page"]) for c in image_chunks}

    for kw in keywords:
        kw_emb = embed_query(kw)

        kw_text_chunks = retrieve_text_chunks_raw(kw_emb, top_k=5)
        for c in kw_text_chunks:
            key = (c["id"], c["page"])
            if key not in existing_text_keys:
                text_chunks.append(c)
                existing_text_keys.add(key)

        kw_image_chunks = retrieve_image_chunks_raw(kw_emb, top_k=3)
        for c in kw_image_chunks:
            key = (c["id"], c["page"])
            if key not in existing_img_keys:
                image_chunks.append(c)
                existing_img_keys.add(key)

    return {
        "text_chunks": text_chunks,
        "image_chunks": image_chunks,
    }

# ==========================================================
# BASE HYBRID RAG (BEFORE CORRECTIVE)
# ==========================================================

def hybrid_rag_retrieve_basic(query: str) -> Dict[str, Any]:
    """
    Basic hybrid retrieval:
    1. Vector + keyword-augmented retrieval from Chroma
    2. Knowledge graph retrieval from Neo4j
    """
    vec_results = keyword_augmented_retrieval(query)
    text_chunks = vec_results["text_chunks"]
    image_chunks = vec_results["image_chunks"]

    kg_results = retrieve_from_knowledge_graph(query, top_k=5)

    return {
        "text_chunks": text_chunks,
        "image_chunks": image_chunks,
        "kg_results": kg_results,
    }

# ==========================================================
# CORRECTIVE RAG: STEP A – RETRIEVAL CRITIC & REWRITE
# ==========================================================

def summarize_context_for_critic(context: Dict[str, Any], max_chars: int = 1500) -> str:
    """Create a short summary string of the retrieved context for the retrieval critic."""
    parts: List[str] = []

    text_chunks = context.get("text_chunks") or []
    img_chunks = context.get("image_chunks") or []
    kg_results = context.get("kg_results") or []

    if text_chunks:
        parts.append("TEXT CHUNKS:")
        for c in text_chunks[:5]:
            parts.append(f"[Page {c.get('page', '?')}] {c.get('text','')[:200]}")

    if img_chunks:
        parts.append("\nIMAGE OCR:")
        for c in img_chunks[:3]:
            parts.append(f"[Page {c.get('page', '?')}] {c.get('ocr_text','')[:120]}")

    if kg_results:
        parts.append("\nKG RESULTS:")
        for c in kg_results[:5]:
            parts.append(f"[Page {c.get('page', '?')} {c.get('type','')}] {c.get('text','')[:150]}")

    summary = "\n".join(parts)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n...[truncated]..."
    return summary


def evaluate_retrieval_and_maybe_rewrite(
    query: str,
    context: Dict[str, Any]
) -> Tuple[str, str]:
    """
    Ask the LLM to judge the retrieval context and optionally
    propose a better search query.

    Returns:
        adequacy: "high" | "medium" | "low"
        improved_query: str (may be empty if not needed)
    """
    context_summary = summarize_context_for_critic(context)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a retrieval quality critic for a textbook RAG system. "
                "You are given a QUESTION and the RETRIEVED CONTEXT (snippets from a textbook). "
                "Your job:\n"
                "1. Judge how well this context can answer the question.\n"
                "2. If the context is weak or missing key info, propose a BETTER_SEARCH_QUERY "
                "that would help retrieve more relevant passages.\n\n"
                "Output strictly in this format:\n"
                "ADEQUACY: HIGH/MEDIUM/LOW\n"
                "BETTER_QUERY: <either a better query or 'NONE'>"
            )
        },
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{query}\n\n"
                f"RETRIEVED CONTEXT SUMMARY:\n{context_summary}"
            )
        }
    ]

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=200
    )

    text = resp.choices[0].message.content or ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    adequacy = "medium"
    better_query = ""

    for line in lines:
        if line.upper().startswith("ADEQUACY:"):
            val = line.split(":", 1)[1].strip().upper()
            if "HIGH" in val:
                adequacy = "high"
            elif "LOW" in val:
                adequacy = "low"
            else:
                adequacy = "medium"
        elif line.upper().startswith("BETTER_QUERY:"):
            val = line.split(":", 1)[1].strip()
            if val and val.upper() != "NONE":
                better_query = val

    return adequacy, better_query


def merge_contexts(ctx1: Dict[str, Any], ctx2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two context dicts, deduplicating by (id, page)."""
    def merge_list(key: str, field_id: str = "id") -> List[Dict[str, Any]]:
        lst1 = ctx1.get(key) or []
        lst2 = ctx2.get(key) or []
        seen = set()
        merged = []
        for item in lst1 + lst2:
            idv = (item.get(field_id), item.get("page"))
            if idv not in seen:
                seen.add(idv)
                merged.append(item)
        return merged

    return {
        "text_chunks": merge_list("text_chunks"),
        "image_chunks": merge_list("image_chunks"),
        "kg_results": merge_list("kg_results"),
    }


def hybrid_rag_retrieve_corrective(query: str) -> Tuple[Dict[str, Any], str]:
    """
    Corrective RAG (Part A):
      1. Run base hybrid retrieval.
      2. Ask critic if context is good enough.
      3. If adequacy == LOW and we get a better query, retrieve again and merge.
    Returns:
      (final_context, retrieval_adequacy)
    """
    base_context = hybrid_rag_retrieve_basic(query)
    adequacy, better_query = evaluate_retrieval_and_maybe_rewrite(query, base_context)

    final_context = base_context

    if adequacy == "low" and better_query:
        improved_context = hybrid_rag_retrieve_basic(better_query)
        final_context = merge_contexts(base_context, improved_context)

    return final_context, adequacy

# ==========================================================
# CONFIDENCE SCORE (HEURISTIC)
# ==========================================================

def compute_confidence_score(
    context: Dict[str, Any],
    answer_adequacy: str = "medium",
    retrieval_adequacy: str = "medium"
) -> int:
    """
    Compute a confidence score (0–100) using:
      - Retrieval adequacy
      - Answer adequacy
      - Result count & content length
      - Vector similarity
      - Source diversity
    """
    text_chunks = context.get("text_chunks") or []
    image_chunks = context.get("image_chunks") or []
    kg_results = context.get("kg_results") or []

    if not (text_chunks or image_chunks or kg_results):
        return 0

    # Map adequacy → numeric
    def adequacy_to_score(a: str) -> float:
        a = (a or "").lower()
        if a == "high":
            return 1.0
        if a == "low":
            return 0.3
        return 0.65

    answer_score = adequacy_to_score(answer_adequacy)
    retrieval_score = adequacy_to_score(retrieval_adequacy)

    # Result count
    total_results = len(text_chunks) + len(image_chunks) + len(kg_results)
    if total_results >= 8:
        result_count_score = 1.0
    elif total_results >= 5:
        result_count_score = 0.85
    elif total_results >= 3:
        result_count_score = 0.7
    elif total_results >= 1:
        result_count_score = 0.55
    else:
        result_count_score = 0.3

    # Content length
    total_len = 0
    total_len += sum(len(c.get("text", "")) for c in text_chunks)
    total_len += sum(len(c.get("ocr_text", "")) for c in image_chunks)
    total_len += sum(len(str(c.get("text", ""))) for c in kg_results)

    if total_len >= 800:
        content_quality_score = 1.0
    elif total_len >= 400:
        content_quality_score = 0.85
    elif total_len >= 200:
        content_quality_score = 0.7
    elif total_len >= 100:
        content_quality_score = 0.55
    else:
        content_quality_score = max(0.3, total_len / 300.0)

    # Vector similarity
    distances = []
    distances += [c.get("distance") for c in text_chunks if c.get("distance") is not None]
    distances += [c.get("distance") for c in image_chunks if c.get("distance") is not None]

    if distances:
        best_d = min(distances)
        avg_d = sum(distances) / len(distances)

        def dist_to_score(d: float) -> float:
            if d <= 0.3:
                return 1.0
            elif d <= 0.5:
                return 0.9
            elif d <= 0.7:
                return 0.75
            elif d <= 1.0:
                return 0.6
            else:
                return max(0.3, 1.0 - (d / 2.0))

        best_sim = dist_to_score(best_d)
        avg_sim = dist_to_score(avg_d)
        retrieval_strength_score = 0.7 * best_sim + 0.3 * avg_sim
    else:
        retrieval_strength_score = 0.6

    # Source diversity
    types_present = 0
    if text_chunks:
        types_present += 1
    if image_chunks:
        types_present += 1
    if kg_results:
        types_present += 1
    source_diversity_score = types_present / 3.0

    # Weighted combination
    combined = (
        0.25 * answer_score +
        0.20 * retrieval_score +
        0.20 * result_count_score +
        0.20 * content_quality_score +
        0.10 * retrieval_strength_score +
        0.05 * source_diversity_score
    )

    score = int(round(max(0.0, min(1.0, combined)) * 100))

    # Caps if answer is weak
    if answer_adequacy == "low":
        score = min(score, 55)
    return score

# ==========================================================
# SELF-VERIFICATION (Corrective RAG C)
# ==========================================================

def build_context_string(context: Dict[str, Any]) -> str:
    parts: List[str] = []

    text_chunks = context.get("text_chunks") or []
    image_chunks = context.get("image_chunks") or []
    kg_results = context.get("kg_results") or []

    if text_chunks:
        parts.append("=== RELEVANT TEXT CONTENT ===")
        for c in text_chunks:
            parts.append(f"[Page {c.get('page','?')}] {c.get('text','')}")

    if image_chunks:
        parts.append("\n=== RELEVANT IMAGES/DIAGRAMS ===")
        for c in image_chunks:
            parts.append(f"[Page {c.get('page','?')}, Image] {c.get('ocr_text','')}")

    if kg_results:
        parts.append("\n=== ADDITIONAL CONTEXT FROM KNOWLEDGE GRAPH ===")
        for c in kg_results:
            parts.append(f"[Page {c.get('page','?')} {c.get('type','')}] {c.get('text','')}")

    return "\n\n".join(parts).strip()


def generate_draft_answer(query: str, context_str: str, chat_history: List[Dict[str, str]]) -> str:
    """First-pass answer (may contain hallucinations)."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful tutor for the TN Class 6 English textbook. "
                "Use ONLY the provided context from the textbook to answer. "
                "If the context is not enough, you can reasonably say that some information is missing, "
                "but still summarize what is present."
            ),
        }
    ]

    # Attach small bit of history for conversational feel
    for msg in chat_history[-5:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": (
            f"CONTEXT:\n{context_str}\n\n"
            f"QUESTION: {query}\n\n"
            "Give a clear, student-friendly answer using only this context."
        )
    })

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=800
    )
    return resp.choices[0].message.content or ""


def verify_and_correct_answer(
    query: str,
    context_str: str,
    draft_answer: str
) -> Tuple[str, str]:
    """
    Self-verification step:
    - Check draft answer against context.
    - Fix hallucinations.
    - Output final answer with ADEQUACY marker.
    Returns: (final_answer_without_marker, adequacy: high/medium/low)
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a STRICT verifier for a textbook-based tutor. "
                "You will be given a QUESTION, CONTEXT, and a DRAFT ANSWER.\n"
                "Your job:\n"
                "1. Check if every important claim in the DRAFT ANSWER is supported by the CONTEXT.\n"
                "2. If some parts are not supported or contradict the context, fix/remove them.\n"
                "3. If the context does not contain enough info, say that clearly.\n\n"
                "At the end, output:\n"
                "ADEQUACY: HIGH/MEDIUM/LOW\n"
                "Where HIGH = fully answered with specific details from context,\n"
                "MEDIUM = partially answered / some info missing,\n"
                "LOW = almost no direct answer in context."
            )
        },
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{query}\n\n"
                f"CONTEXT:\n{context_str}\n\n"
                f"DRAFT ANSWER:\n{draft_answer}\n\n"
                "Now produce the final corrected answer for the student, then the ADEQUACY line."
            )
        }
    ]

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=800
    )
    full = resp.choices[0].message.content or ""

    # Split off ADEQUACY marker
    adequacy = "medium"
    if "ADEQUACY:" in full:
        parts = full.split("ADEQUACY:", 1)
        answer_part = parts[0].strip()
        marker = parts[1].strip().upper()
        if "HIGH" in marker:
            adequacy = "high"
        elif "LOW" in marker:
            adequacy = "low"
        else:
            adequacy = "medium"
        return answer_part, adequacy
    else:
        # Heuristic fallback
        lower = full.lower()
        if any(p in lower for p in ["not enough information", "no details about"]):
            adequacy = "low"
        return full.strip(), adequacy

# ==========================================================
# HIGH LEVEL: FULL CORRECTIVE RAG (A + C)
# ==========================================================

def run_corrective_rag_pipeline(query: str, chat_history: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any], int]:
    """
    Full pipeline:
      1. Hybrid retrieval (Chroma + KG) + Corrective query (A).
      2. Draft answer from context.
      3. Self-verification & correction (C).
      4. Confidence score.
    Returns:
      final_answer, context, confidence_score
    """
    # 1) Corrective retrieval (A)
    context, retrieval_adequacy = hybrid_rag_retrieve_corrective(query)

    # If absolutely no context, answer gracefully
    if not (context.get("text_chunks") or context.get("image_chunks") or context.get("kg_results")):
        answer = (
            "I couldn't find any relevant information about this question in the pages that were indexed. "
            "It might be from a part of the book that has not been ingested yet."
        )
        confidence = 5
        return answer, context, confidence

    # 2) Build context string
    context_str = build_context_string(context)

    # 3) Draft answer
    draft = generate_draft_answer(query, context_str, chat_history)

    # 4) Self-verification & correction (C)
    final_answer, answer_adequacy = verify_and_correct_answer(query, context_str, draft)

    # 5) Confidence score
    confidence = compute_confidence_score(context, answer_adequacy, retrieval_adequacy)

    return final_answer, {
        **context,
        "answer_adequacy": answer_adequacy,
        "retrieval_adequacy": retrieval_adequacy,
    }, confidence

# ==========================================================
# STREAMLIT UI
# ==========================================================

def main():
    st.set_page_config(
        page_title="TN Class 6 English Tutor",
        page_icon="📚",
        layout="wide",
    )

    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">📚 TN Class 6 English Tutor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Hybrid RAG + Knowledge Graph + Corrective RAG</div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")

        st.session_state.show_sources = st.checkbox("Show sources", value=True)

        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        example_questions = [
            "Who talked about beach cleanliness?",
            "Tell me about Lilly and her friends at the beach.",
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
        st.info("""
        *Data Source:* TN Class 6 English (text + images ingested into Chroma & Neo4j)

        *Technologies:*
        - 🔍 Vector Search (Chroma + OpenAI embeddings)
        - 🕸️ Knowledge Graph (Neo4j)
        - 🧠 Corrective RAG (query rewrite + self-check)
        - 🤖 LLM: GPT-4o-mini family
        """)

    chat_container = st.container()

    # Show history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message["role"] == "assistant":
                    score = message.get("confidence_score")
                    if score is not None:
                        score_clamped = max(0, min(100, int(score)))
                        st.caption(f"✅ Confidence score: {score_clamped} / 100")
                        st.progress(score_clamped / 100.0)

                if (
                    message["role"] == "assistant"
                    and "sources" in message
                    and st.session_state.show_sources
                ):
                    with st.expander("📖 View Sources"):
                        sources = message["sources"]

                        if sources.get("text_chunks"):
                            st.markdown("*📄 Text Chunks:*")
                            for chunk in sources["text_chunks"]:
                                preview = chunk["text"][:150].replace("\n", " ")
                                st.markdown(f"- Page {chunk['page']}: {preview}...")

                        if sources.get("image_chunks"):
                            st.markdown("*🖼️ Images/Diagrams:*")
                            for img in sources["image_chunks"]:
                                preview = img["ocr_text"][:100].replace("\n", " ")
                                st.markdown(f"- Page {img['page']}: {preview}...")

                        if sources.get("kg_results"):
                            st.markdown("*🕸️ Knowledge Graph:*")
                            for item in sources["kg_results"]:
                                preview = item["text"][:100].replace("\n", " ")
                                st.markdown(f"- Page {item['page']} ({item['type']}): {preview}...")

    # input
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
                with st.spinner("🔍 Corrective RAG in progress... 🤔"):
                    try:
                        answer, context, confidence = run_corrective_rag_pipeline(
                            user_input,
                            st.session_state.messages
                        )

                        st.markdown(answer)

                        if confidence is not None:
                            score_clamped = max(0, min(100, int(confidence)))
                            st.caption(f"✅ Confidence score: {score_clamped} / 100")
                            st.progress(score_clamped / 100.0)

                        if st.session_state.show_sources:
                            with st.expander("📖 View Sources"):
                                if context.get("text_chunks"):
                                    st.markdown("*📄 Text Chunks:*")
                                    for chunk in context["text_chunks"]:
                                        preview = chunk["text"][:150].replace("\n", " ")
                                        st.markdown(f"- Page {chunk['page']}: {preview}...")

                                if context.get("image_chunks"):
                                    st.markdown("*🖼️ Images/Diagrams:*")
                                    for img in context["image_chunks"]:
                                        preview = img["ocr_text"][:100].replace("\n", " ")
                                        st.markdown(f"- Page {img['page']}: {preview}...")

                                if context.get("kg_results"):
                                    st.markdown("*🕸️ Knowledge Graph:*")
                                    for item in context["kg_results"]:
                                        preview = item["text"][:100].replace("\n", " ")
                                        st.markdown(f"- Page {item['page']} ({item['type']}): {preview}...")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": context,
                            "confidence_score": confidence,
                        })

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.markdown(
                            "I encountered an error while processing your question. Please try again."
                        )

        st.rerun()

# ==========================================================
# RUN APP
# ==========================================================
if __name__ == "__main__":
    main()
