import os
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pyvis.network import Network

# ==========================================================
# LOAD ENV
# ==========================================================
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise RuntimeError("Neo4j env vars missing: NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD")


# ==========================================================
# NEO4J CONNECTION
# ==========================================================
@st.cache_resource
def get_driver():
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    driver.verify_connectivity()
    return driver


# ==========================================================
# QUERY NEO4J: GET SUBGRAPH
# ==========================================================
def fetch_subgraph(
    min_page: int,
    max_page: int,
    max_nodes: int,
    keyword: str = ""
) -> Dict[str, Any]:
    """
    Returns a dict with:
      - nodes: {id: {labels, props}}
      - rels: [{start, end, type}]
    """
    driver = get_driver()

    # If you later add explicit relationships (e.g. :NEXT, :MENTIONS, etc.)
    # this query will show them. For now it just pulls any existing relations.
    cypher = """
    MATCH (n)
    WHERE (n:TextChunk OR n:ImageChunk)
      AND n.page >= $min_page
      AND n.page <= $max_page
      AND ($keyword = '' OR toLower(
            coalesce(n.text, n.ocr_text, '')
          ) CONTAINS toLower($keyword))
    WITH n
    LIMIT $max_nodes

    OPTIONAL MATCH (n)-[r]-(m)
    RETURN n, r, m
    """

    nodes = {}
    rels = []

    with driver.session() as session:
        for record in session.run(
            cypher,
            min_page=min_page,
            max_page=max_page,
            max_nodes=max_nodes,
            keyword=keyword or ""
        ):
            n = record["n"]
            r = record["r"]
            m = record["m"]

            # Collect node n
            nid = n.id
            if nid not in nodes:
                nodes[nid] = {
                    "labels": list(n.labels),
                    "props": dict(n.items())
                }

            # Collect node m if exists
            if m is not None:
                mid = m.id
                if mid not in nodes:
                    nodes[mid] = {
                        "labels": list(m.labels),
                        "props": dict(m.items())
                    }

            # Collect relationship if exists
            if r is not None:
                rels.append({
                    "start": r.start_node.id,
                    "end": r.end_node.id,
                    "type": r.type
                })

    return {"nodes": nodes, "rels": rels}


# ==========================================================
# BUILD PYVIS GRAPH
# ==========================================================
def build_pyvis_graph(subgraph: Dict[str, Any]) -> Network:
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#0e1117",
        font_color="white",
        notebook=False,
        directed=False
    )

    net.barnes_hut()  # nice layout

    nodes = subgraph["nodes"]
    rels = subgraph["rels"]

    # Add nodes
    for nid, data in nodes.items():
        labels = data["labels"]
        props = data["props"]

        page = props.get("page", "?")
        node_id = props.get("id", str(nid))

        # Decide type & color
        if "TextChunk" in labels:
            color = "#42a5f5"  # blue
            kind = "TextChunk"
            text = props.get("text", "")
        elif "ImageChunk" in labels:
            color = "#66bb6a"  # green
            kind = "ImageChunk"
            text = props.get("ocr_text", "")
        else:
            color = "#bdbdbd"  # grey
            kind = ",".join(labels)
            text = str(props)

        # Tooltip content
        snippet = (text[:250] + "...") if text and len(text) > 250 else text
        title = f"<b>{kind}</b><br>Page: {page}<br><br>{snippet.replace(chr(10), '<br>')}"

        # Node size based on text length
        text_len = len(text or "")
        size = min(40, max(10, text_len // 20))

        net.add_node(
            nid,
            label=f"{kind[0]}{page}",  # e.g. "T14", "I32"
            title=title,
            color=color,
            size=size
        )

    # Add edges
    for r in rels:
        net.add_edge(
            r["start"],
            r["end"],
            title=r["type"],
            color="#9e9e9e"
        )

    return net


# ==========================================================
# STREAMLIT APP
# ==========================================================
def main():
    st.set_page_config(
        page_title="TN Class 6 KG Viewer",
        page_icon="🕸️",
        layout="wide"
    )

    st.title("🕸️ TN Class 6 Knowledge Graph Viewer")
    st.caption("Neo4j + PyVis • Explore TextChunk & ImageChunk nodes")

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")

        page_min, page_max = st.slider(
            "Page range",
            min_value=1,
            max_value=80,   # adjust if your book has different max
            value=(10, 60)
        )

        max_nodes = st.slider(
            "Max nodes",
            min_value=10,
            max_value=500,
            value=150,
            step=10
        )

        keyword = st.text_input(
            "Keyword filter (optional)",
            placeholder="e.g. turtle, Lilly, beach, chilli..."
        )

        if st.button("Refresh graph"):
            st.session_state["refresh"] = True

    # Fetch & display graph
    with st.spinner("Querying Neo4j and building graph..."):
        subgraph = fetch_subgraph(
            min_page=page_min,
            max_page=page_max,
            max_nodes=max_nodes,
            keyword=keyword.strip()
        )

        node_count = len(subgraph["nodes"])
        rel_count = len(subgraph["rels"])

        st.subheader("Graph Summary")
        st.markdown(
            f"- **Nodes:** {node_count}<br>"
            f"- **Relationships:** {rel_count}<br>"
            f"- **Pages:** {page_min}–{page_max}"
            , unsafe_allow_html=True
        )

        if node_count == 0:
            st.warning("No nodes found for this filter. Try widening the page range or removing the keyword.")
            return

        net = build_pyvis_graph(subgraph)

        # Write to HTML & embed
        html_file = "kg_viewer.html"
        net.write_html(html_file, notebook=False)

        with open(html_file, "r", encoding="utf-8") as f:
            html = f.read()

        st.components.v1.html(html, height=750, scrolling=True)


if __name__ == "__main__":
    main()
