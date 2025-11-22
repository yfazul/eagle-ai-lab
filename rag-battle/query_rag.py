import os
import streamlit as st
from typing import List, Dict, Any
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
    for i in range(len(results["ids"][0])):
        chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "page": results["metadatas"][0][i]["page"],
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
    """Retrieve context from Neo4j Knowledge Graph"""
    driver = get_neo4j_driver()

    # Extract keywords from query (simple approach)
    keywords = [word.lower() for word in query.split() if len(word) > 3]

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
# KEYWORD-AWARE SEMANTIC BOOST (FIXED)
# ==========================================================

# Expanded stopwords list
STOPWORDS = {
    # Question words
    "what", "who", "whom", "whose", "which", "where", "when", "why", "how",
    # Common verbs
    "tell", "about", "explain", "describe", "show", "give", "find", "get",
    "help", "need", "want", "like", "know", "think", "say", "said",
    "do", "does", "did", "done", "doing",
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "having",
    "can", "could", "may", "might", "must", "shall", "should", "will", "would",
    # Pronouns
    "me", "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "we", "us", "our", "ours", "they", "them", "their", "theirs",
    "this", "that", "these", "those",
    # Articles and common words
    "the", "a", "an", "and", "or", "but", "if", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "from", "up", "on", "off",
    "out", "in", "over", "under", "again", "further", "then", "once",
    # Polite words
    "please", "thanks", "thank", "sorry",
    # Common adjectives
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
}

def extract_keywords(query: str) -> list:
    """
    Extract potentially important keywords from the query.
    Keeps words >2 chars and not in the stopword list.
    Also preserves capitalized words (likely proper nouns) even if short.
    """
    kws = []
    tokens = query.strip().replace("?", " ").replace(".", " ").split()
    
    for token in tokens:
        # Preserve original for capitalization check
        original = token.strip().strip(".,!?\"'()")
        clean = original.lower()
        
        # Keep if:
        # 1. Length > 2 and not a stopword, OR
        # 2. Capitalized (likely a proper noun like "Ayesha")
        if (len(clean) > 2 and clean not in STOPWORDS) or \
           (len(original) > 0 and original[0].isupper() and clean not in STOPWORDS):
            kws.append(clean)
    
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
       and merge results (good for names like 'ayesha').
    """
    # 1. full-query embedding
    base_embedding = embed_query(query)

    text_chunks = retrieve_text_chunks_raw(base_embedding, top_k=5)
    image_chunks = retrieve_image_chunks_raw(base_embedding, top_k=3)

    # 2. keyword-specific searches
    keywords = extract_keywords(query)

    existing_text_keys = {(c["id"], c["page"]) for c in text_chunks}
    existing_img_keys = {(c["id"], c["page"]) for c in image_chunks}

    for kw in keywords:
        kw_emb = embed_query(kw)

        # Increase top_k for keyword searches to get better coverage
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
# CONFIDENCE SCORE (HEURISTIC)
# ==========================================================

def compute_confidence_score(context: Dict[str, Any], answer_adequacy: str = "medium") -> int:
    """
    Compute a balanced confidence score (0–100) based on:
      - Answer adequacy (how well context addresses the query)
      - Result count and quality
      - Content relevance and length
      - Vector similarity strength
      - Source diversity (bonus)
    """
    text_chunks = context.get("text_chunks") or []
    image_chunks = context.get("image_chunks") or []
    kg_results = context.get("kg_results") or []

    if not (text_chunks or image_chunks or kg_results):
        return 0

    # 1. ANSWER ADEQUACY (40% weight) - MOST IMPORTANT!
    # Based on LLM's assessment of whether context answers the question
    if answer_adequacy == "high":
        adequacy_score = 1.0
    elif answer_adequacy == "medium":
        adequacy_score = 0.65
    elif answer_adequacy == "low":
        adequacy_score = 0.35
    else:  # "none"
        adequacy_score = 0.1

    # 2. RESULT COUNT & QUALITY (25% weight)
    # Having multiple relevant results indicates confidence
    total_results = len(text_chunks) + len(image_chunks) + len(kg_results)
    
    if total_results >= 8:
        result_count_score = 1.0
    elif total_results >= 5:
        result_count_score = 0.85
    elif total_results >= 3:
        result_count_score = 0.70
    elif total_results >= 1:
        result_count_score = 0.55
    else:
        result_count_score = 0.3

    # 3. CONTENT QUALITY (20% weight)
    # Check if we have substantial, relevant content
    total_content_length = 0
    total_content_length += sum(len(c.get("text", "")) for c in text_chunks)
    total_content_length += sum(len(c.get("ocr_text", "")) for c in image_chunks)
    total_content_length += sum(len(str(c.get("text", ""))) for c in kg_results)
    
    # Scoring based on content length
    if total_content_length >= 800:
        content_quality_score = 1.0
    elif total_content_length >= 400:
        content_quality_score = 0.85
    elif total_content_length >= 200:
        content_quality_score = 0.70
    elif total_content_length >= 100:
        content_quality_score = 0.55
    else:
        content_quality_score = max(0.35, total_content_length / 300)

    # 4. VECTOR SIMILARITY (10% weight)
    # Based on distances - lower is better
    distances = []
    distances += [c.get("distance") for c in text_chunks if c.get("distance") is not None]
    distances += [c.get("distance") for c in image_chunks if c.get("distance") is not None]

    if distances:
        # Get the best (lowest) distance
        best_distance = min(distances)
        avg_distance = sum(distances) / len(distances)
        
        # Convert to similarity scores
        def distance_to_score(d):
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
        
        best_sim = distance_to_score(best_distance)
        avg_sim = distance_to_score(avg_distance)
        
        # Weight best match more heavily
        retrieval_strength_score = 0.7 * best_sim + 0.3 * avg_sim
    else:
        retrieval_strength_score = 0.6  # neutral default

    # 5. SOURCE DIVERSITY BONUS (5% weight)
    # Multiple source types add confidence
    source_types_present = 0
    if text_chunks:
        source_types_present += 1
    if image_chunks:
        source_types_present += 1
    if kg_results:
        source_types_present += 1
    
    source_diversity_score = source_types_present / 3.0

    # Calculate final weighted score
    combined = (
        0.40 * adequacy_score +
        0.25 * result_count_score +
        0.20 * content_quality_score +
        0.10 * retrieval_strength_score +
        0.05 * source_diversity_score
    )

    score = int(round(max(0.0, min(1.0, combined)) * 100))
    
    # Apply caps based on adequacy
    if answer_adequacy == "low":
        score = min(score, 50)  # Cap at 50 for low adequacy
    elif answer_adequacy == "none":
        score = min(score, 20)  # Cap at 20 for no answer
    
    return score

# ==========================================================
# HYBRID RAG: Vector + KG + keyword-augmented retrieval
# ==========================================================

def hybrid_rag_retrieve(query: str) -> Dict[str, Any]:
    """
    Perform hybrid retrieval using:
    1. Vector similarity (Chroma) on the full question
    2. Extra vector retrieval on extracted keywords (names etc.)
    3. Knowledge Graph (Neo4j) keyword search
    """
    vec_results = keyword_augmented_retrieval(query)
    text_chunks = vec_results["text_chunks"]
    image_chunks = vec_results["image_chunks"]

    kg_results = retrieve_from_knowledge_graph(query, top_k=5)

    context: Dict[str, Any] = {
        "text_chunks": text_chunks,
        "image_chunks": image_chunks,
        "kg_results": kg_results,
    }

    return context

# ==========================================================
# GENERATE RESPONSE WITH GPT
# ==========================================================

def generate_response(query: str, context: Dict[str, Any], chat_history: List[Dict]) -> tuple[str, dict]:
    """Generate response using GPT-4 with retrieved context
    
    Returns:
        tuple: (response_text, metadata_dict)
        metadata contains: answer_adequacy (low/medium/high)
    """

    # Build context string
    context_parts = []

    # Add text chunks
    if context.get('text_chunks'):
        context_parts.append("=== RELEVANT TEXT CONTENT ===")
        for chunk in context['text_chunks']:
            context_parts.append(f"[Page {chunk['page']}] {chunk['text']}")

    # Add image OCR
    if context.get('image_chunks'):
        context_parts.append("\n=== RELEVANT IMAGES/DIAGRAMS ===")
        for img in context['image_chunks']:
            context_parts.append(f"[Page {img['page']}, Image] {img['ocr_text']}")

    # Add knowledge graph context
    if context.get('kg_results'):
        context_parts.append("\n=== ADDITIONAL CONTEXT FROM KNOWLEDGE GRAPH ===")
        for item in context['kg_results']:
            context_parts.append(f"[Page {item['page']}, {item['type']}] {item['text'][:200]}...")

    context_str = "\n\n".join(context_parts).strip()

    # If we truly have no context at all, handle fallback here
    if not context_str:
        return (
            "I'm sorry, but I couldn't find any relevant information about your question "
            "in the indexed pages of the textbook. Please try rephrasing your question or "
            "ask about a different topic from the book.",
            {"answer_adequacy": "none"}
        )

    # Build messages with chat history
    messages = [
        {
            "role": "system",
            "content": """You are a helpful tutor for the TN Class 6 English textbook.
Use the provided context from the textbook to answer questions.
If the context doesn't contain enough information to fully answer the question,
say so honestly and explain what IS known from the context instead of guessing.
Always cite the page numbers when providing information.
Be encouraging and educational in your responses.

IMPORTANT: At the very end of your response, add a line starting with "ADEQUACY:" followed by one of:
- HIGH: if the context fully answers the question with detailed information
- MEDIUM: if the context partially answers with some relevant details
- LOW: if the context has only brief mentions or doesn't really answer the question"""
        }
    ]

    # Add chat history (last 5 messages)
    for msg in chat_history[-5:]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # Add current query with context
    messages.append({
        "role": "user",
        "content": f"""Here is the retrieved context from the textbook:

{context_str}

User question: {query}

Using only the context above, answer the question.
If there is only a brief mention (for example just a single line with a name),
still summarize that brief information instead of saying there is no information.

Remember to end with "ADEQUACY: [HIGH/MEDIUM/LOW]" based on how well the context answers the question."""
    })

    # Generate response
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )

    full_response = response.choices[0].message.content
    
    # Extract adequacy marker
    answer_adequacy = "medium"  # default
    if "ADEQUACY:" in full_response:
        parts = full_response.split("ADEQUACY:")
        clean_response = parts[0].strip()
        adequacy_marker = parts[1].strip().upper()
        
        if "HIGH" in adequacy_marker:
            answer_adequacy = "high"
        elif "LOW" in adequacy_marker:
            answer_adequacy = "low"
        else:
            answer_adequacy = "medium"
    else:
        clean_response = full_response
        
        # Fallback: detect inadequacy from response text
        inadequacy_phrases = [
            "does not contain",
            "no specific information",
            "no details about",
            "doesn't mention",
            "not enough information",
            "cannot find",
            "only mentioned briefly",
            "brief mention",
            "no information about"
        ]
        
        lower_response = clean_response.lower()
        if any(phrase in lower_response for phrase in inadequacy_phrases):
            answer_adequacy = "low"

    return clean_response, {"answer_adequacy": answer_adequacy}

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
    st.markdown('<div class="sub-header">Powered by RAG + Knowledge Graph</div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")

        st.session_state.show_sources = st.checkbox("Show sources", value=True)

        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        example_questions = [
            "What is the main theme of the first lesson?",
            "Tell me about the characters in the story",
            "What are the key vocabulary words?",
            "Explain the grammar concept from page 5",
            "What exercises are there in unit 1?",
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
        *Data Source:* TN Class 6 English (pages ingested into Chroma/Neo4j)

        *Technologies:*
        - 🔍 Vector Search (Chroma)
        - 🕸️ Knowledge Graph (Neo4j)
        - 🔑 Keyword-aware retrieval (names/entities)
        - 🤖 LLM (GPT-4 family)
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
                with st.spinner("🔍 Searching textbook... 🤔 Thinking..."):
                    try:
                        context = hybrid_rag_retrieve(user_input)
                        response, metadata = generate_response(user_input, context, st.session_state.messages)
                        answer_adequacy = metadata.get("answer_adequacy", "medium")
                        confidence_score = compute_confidence_score(context, answer_adequacy)

                        st.markdown(response)

                        if confidence_score is not None:
                            score_clamped = max(0, min(100, int(confidence_score)))
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
                            "content": response,
                            "sources": context,
                            "confidence_score": confidence_score,
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