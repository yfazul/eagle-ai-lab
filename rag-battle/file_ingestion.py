"""
Enhanced Multimodal Knowledge Graph Pipeline
LangChain + LangGraph + Neo4j

Improvements over original:
- Better error handling and validation
- Progress tracking with tqdm
- Configurable transformer settings
- Schema validation and optimization
- Proper cleanup and connection management
- Enhanced logging
- Relationship extraction improvements
- Better batching strategy
"""

import os
import sys
from typing import TypedDict, Dict, Any, List, Optional
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# LangChain core + community
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document

# Fallback for older LangChain versions
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain experimental KG transformer
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer

# LangGraph
from langgraph.graph import StateGraph, START, END

# Logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================================================
# CONFIGURATION
# ==========================================================

class PipelineConfig:
    """Centralized configuration"""
    
    def __init__(self):
        load_dotenv()
        
        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_username = os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        # Pipeline settings
        self.pdf_path = os.getenv("PDF_PATH", "./data/tn_class6_english.pdf")
        self.max_pages = int(os.getenv("MAX_PAGES", "0"))  # 0 = all pages
        
        # Chunking settings
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        
        # Processing settings
        self.text_batch_size = int(os.getenv("TEXT_BATCH_SIZE", "3"))
        self.image_batch_size = int(os.getenv("IMAGE_BATCH_SIZE", "5"))
        
        # Model settings
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.temperature = float(os.getenv("TEMPERATURE", "0"))
        
        # Validate
        self._validate()
    
    def _validate(self):
        """Validate configuration"""
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY missing in .env")
        
        if not all([self.neo4j_uri, self.neo4j_username, self.neo4j_password]):
            raise RuntimeError("Neo4j configuration incomplete")
        
        if not Path(self.pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")


# ==========================================================
# ENHANCED SCHEMA DEFINITION
# ==========================================================

# Define comprehensive schema for textbook domain
ALLOWED_NODES = [
    "Person",
    "Character",
    "Place",
    "Location",
    "Organization",
    "Event",
    "Object",
    "Thing",
    "Concept",
    "Topic",
    "Activity",
    "Animal",
    "Food",
]

ALLOWED_RELATIONSHIPS = [
    # Family relationships
    "CHILD_OF",
    "PARENT_OF",
    "SIBLING_OF",
    
    # Social relationships
    "FRIEND_OF",
    "KNOWS",
    "TALKS_TO",
    "MEETS",
    
    # Actions
    "GOES_TO",
    "VISITS",
    "LIVES_IN",
    "WORKS_AT",
    "STUDIES_AT",
    
    # Educational
    "TEACHES",
    "LEARNS",
    "HELPS",
    "ASKS",
    "EXPLAINS",
    
    # Spatial
    "LOCATED_IN",
    "NEAR",
    "PART_OF",
    
    # Possessive
    "HAS",
    "OWNS",
    "BELONGS_TO",
    
    # Descriptive
    "DESCRIBES",
    "MENTIONS",
    "RELATED_TO",
    "ABOUT",
]


# ==========================================================
# INITIALIZE COMPONENTS
# ==========================================================

class KGComponents:
    """Manages LangChain components with proper lifecycle"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.llm = None
        self.graph = None
        self.transformer = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize all components"""
        logger.info("Initializing KG components...")
        
        # LLM for entity extraction
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            api_key=self.config.openai_api_key,
            max_retries=3,
        )
        
        # Neo4j connection
        self.graph = Neo4jGraph(
            url=self.config.neo4j_uri,
            username=self.config.neo4j_username,
            password=self.config.neo4j_password,
            refresh_schema=False,
        )
        
        # Test connection
        try:
            self.graph.query("RETURN 1 as test")
            logger.info("✓ Neo4j connection successful")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            raise
        
        # KG Transformer with enhanced configuration
        self.transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=ALLOWED_NODES,
            allowed_relationships=ALLOWED_RELATIONSHIPS,
            node_properties=["description"],  # Extract node descriptions
            relationship_properties=False,
            strict_mode=False,  # Allow fuzzy matching
        )
        
        logger.info("✓ All components initialized")
    
    def create_indexes(self):
        """Create Neo4j indexes for better performance"""
        logger.info("Creating Neo4j indexes...")
        
        # CRITICAL: Drop ALL existing indexes and constraints that might conflict
        logger.info("Cleaning up any existing indexes and constraints...")
        
        # Get all indexes and drop them
        try:
            indexes = self.graph.query("SHOW INDEXES")
            for idx in indexes:
                name = idx.get('name', '')
                if name:
                    try:
                        self.graph.query(f"DROP INDEX `{name}` IF EXISTS")
                        logger.info(f"Dropped index: {name}")
                    except Exception as e:
                        logger.debug(f"Could not drop index {name}: {e}")
        except Exception as e:
            logger.warning(f"Could not list indexes: {e}")
        
        # Get all constraints and drop them
        try:
            constraints = self.graph.query("SHOW CONSTRAINTS")
            for cons in constraints:
                name = cons.get('name', '')
                if name:
                    try:
                        self.graph.query(f"DROP CONSTRAINT `{name}` IF EXISTS")
                        logger.info(f"Dropped constraint: {name}")
                    except Exception as e:
                        logger.debug(f"Could not drop constraint {name}: {e}")
        except Exception as e:
            logger.warning(f"Could not list constraints: {e}")
        
        # Now create only non-conflicting indexes
        # NOTE: We do NOT create an index on __Entity__.id because
        # LangChain's add_graph_documents will try to create a CONSTRAINT on it
        index_queries = [
            # Document indexes
            "CREATE INDEX document_id IF NOT EXISTS FOR (n:Document) ON (n.id)",
            
            # Text chunk indexes
            "CREATE INDEX text_chunk_id IF NOT EXISTS FOR (n:TextChunk) ON (n.id)",
            "CREATE INDEX text_chunk_page IF NOT EXISTS FOR (n:TextChunk) ON (n.page)",
            
            # Image chunk indexes
            "CREATE INDEX image_chunk_id IF NOT EXISTS FOR (n:ImageChunk) ON (n.id)",
            "CREATE INDEX image_chunk_page IF NOT EXISTS FOR (n:ImageChunk) ON (n.page)",
            
            # Entity indexes for specific types (not __Entity__)
            "CREATE INDEX person_id IF NOT EXISTS FOR (n:Person) ON (n.id)",
            "CREATE INDEX character_id IF NOT EXISTS FOR (n:Character) ON (n.id)",
            "CREATE INDEX place_id IF NOT EXISTS FOR (n:Place) ON (n.id)",
            "CREATE INDEX location_id IF NOT EXISTS FOR (n:Location) ON (n.id)",
        ]
        
        for query in index_queries:
            try:
                self.graph.query(query)
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.debug(f"Index already exists: {e}")
                else:
                    logger.warning(f"Index creation warning: {e}")
        
        logger.info("✓ Indexes created")
    
    def refresh_schema(self):
        """Refresh Neo4j schema"""
        try:
            self.graph.refresh_schema()
            logger.info("✓ Schema refreshed")
        except Exception as e:
            logger.warning(f"Schema refresh warning: {e}")
    
    def close(self):
        """Clean up connections"""
        if self.graph:
            try:
                self.graph._driver.close()
                logger.info("✓ Neo4j connection closed")
            except:
                pass


# ==========================================================
# TEXT INGESTION WITH IMPROVEMENTS
# ==========================================================

def ingest_pdf_text_to_kg(
    components: KGComponents,
    config: PipelineConfig
) -> Dict[str, Any]:
    """
    Enhanced PDF text ingestion with:
    - Better error handling
    - Progress tracking
    - Improved metadata
    - Deduplication
    """
    logger.info(f"=== TEXT → KG: Loading PDF from {config.pdf_path} ===")
    
    # Load PDF
    try:
        loader = PyMuPDFLoader(config.pdf_path)
        docs: List[Document] = loader.load()
    except Exception as e:
        logger.error(f"Failed to load PDF: {e}")
        raise
    
    # Limit pages if specified
    original_count = len(docs)
    if config.max_pages > 0:
        docs = docs[:config.max_pages]
    
    logger.info(f"Loaded {original_count} pages, using {len(docs)} pages")
    
    # Ensure page metadata
    for i, doc in enumerate(docs):
        if "page" not in doc.metadata:
            doc.metadata["page"] = i + 1
        if "source" not in doc.metadata:
            doc.metadata["source"] = config.pdf_path
    
    # Split into chunks with better configuration
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    try:
        chunks: List[Document] = splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} text chunks")
    except Exception as e:
        logger.error(f"Failed to split documents: {e}")
        raise
    
    # Assign stable IDs and metadata
    for idx, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", 1)
        chunk_id = f"p{page}_t{idx+1}"
        
        chunk.metadata.update({
            "id": chunk_id,
            "source_type": "text",
            "page": page,
            "chunk_index": idx + 1,
            "total_chunks": len(chunks),
        })
    
    # CRITICAL FIX: Ensure constraints are set up before processing
    logger.info("Setting up graph schema constraints...")
    try:
        # Let's manually create the constraint that LangChain needs
        # This way it won't fail when trying to create it during add_graph_documents
        components.graph.query("""
            CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
            FOR (e:__Entity__) REQUIRE e.id IS UNIQUE
        """)
        logger.info("✓ Created __Entity__ constraint")
    except Exception as e:
        logger.warning(f"Could not create __Entity__ constraint (may already exist): {e}")
    
    # Process in batches with progress bar
    batch_size = config.text_batch_size
    total_graph_docs = 0
    failed_batches = 0
    
    with tqdm(total=len(chunks), desc="Processing text chunks") as pbar:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            
            try:
                # Convert to graph documents
                graph_docs = components.transformer.convert_to_graph_documents(batch)
                
                # Persist to Neo4j
                components.graph.add_graph_documents(
                    graph_docs,
                    include_source=True,
                    baseEntityLabel=True,
                )
                
                total_graph_docs += len(graph_docs)
                pbar.update(len(batch))
                
            except Exception as e:
                error_msg = str(e)
                # Check if it's the constraint error
                if "IndexAlreadyExists" in error_msg or "constraint" in error_msg.lower():
                    logger.warning(f"Schema conflict on batch {start}-{start+len(batch)}, retrying without baseEntityLabel...")
                    try:
                        # Retry without baseEntityLabel
                        components.graph.add_graph_documents(
                            graph_docs,
                            include_source=True,
                            baseEntityLabel=False,  # Disable this to avoid constraint issues
                        )
                        total_graph_docs += len(graph_docs)
                        pbar.update(len(batch))
                    except Exception as e2:
                        logger.error(f"Retry failed for batch {start}-{start+len(batch)}: {e2}")
                        failed_batches += 1
                        pbar.update(len(batch))
                else:
                    logger.error(f"Failed to process batch {start}-{start+len(batch)}: {e}")
                    failed_batches += 1
                    pbar.update(len(batch))
                
                # Continue processing other batches
                if failed_batches > 5:
                    logger.error("Too many failures, stopping text ingestion")
                    break
    
    # Create relationships between chunks on same page
    logger.info("Creating sequential chunk relationships...")
    components.graph.query("""
        MATCH (d1:Document), (d2:Document)
        WHERE d1.page = d2.page 
        AND d1.chunk_index = d2.chunk_index - 1
        AND d1.source_type = 'text'
        AND d2.source_type = 'text'
        MERGE (d1)-[:NEXT_CHUNK]->(d2)
    """)
    
    stats = {
        "num_pages": len(docs),
        "num_chunks": len(chunks),
        "num_graph_docs": total_graph_docs,
        "failed_batches": failed_batches,
    }
    
    logger.info(f"✓ Text KG built: {total_graph_docs} GraphDocuments, {failed_batches} failures")
    return stats


# ==========================================================
# IMAGE OCR ENRICHMENT WITH IMPROVEMENTS
# ==========================================================

def enrich_kg_from_image_ocr(
    components: KGComponents,
    config: PipelineConfig
) -> Dict[str, Any]:
    """
    Enhanced OCR enrichment with:
    - Better filtering of OCR text
    - Metadata preservation
    - Error handling per image
    - Progress tracking
    """
    logger.info("=== OCR IMAGE TEXT → KG: Using existing ImageChunk nodes ===")
    
    # Query existing ImageChunk nodes with OCR text
    try:
        rows = components.graph.query("""
            MATCH (i:ImageChunk)
            WHERE i.ocr_text IS NOT NULL 
            AND trim(i.ocr_text) <> ''
            AND length(trim(i.ocr_text)) > 20
            RETURN i.id AS id, 
                   i.ocr_text AS text, 
                   i.page AS page,
                   i.image_path AS image_path
            ORDER BY i.page
        """)
    except Exception as e:
        logger.error(f"Failed to query ImageChunk nodes: {e}")
        return {"num_images": 0, "num_image_docs": 0}
    
    if not rows:
        logger.warning("No ImageChunk nodes with OCR text found")
        return {"num_images": 0, "num_image_docs": 0}
    
    logger.info(f"Found {len(rows)} ImageChunk nodes with OCR text")
    
    # Build documents from OCR text
    docs: List[Document] = []
    for row in rows:
        img_id = row["id"]
        text = row["text"]
        page = row.get("page", 0)
        image_path = row.get("image_path", "")
        
        # Clean OCR text
        text = text.strip()
        
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "id": img_id,
                    "source_type": "image",
                    "page": page,
                    "image_path": image_path,
                    "ocr_length": len(text),
                },
            )
        )
    
    # Process in batches with progress
    batch_size = config.image_batch_size
    total_graph_docs = 0
    failed_batches = 0
    
    with tqdm(total=len(docs), desc="Processing image OCR") as pbar:
        for start in range(0, len(docs), batch_size):
            batch = docs[start : start + batch_size]
            
            try:
                graph_docs = components.transformer.convert_to_graph_documents(batch)
                
                components.graph.add_graph_documents(
                    graph_docs,
                    include_source=True,
                    baseEntityLabel=True,
                )
                
                total_graph_docs += len(graph_docs)
                pbar.update(len(batch))
                
            except Exception as e:
                logger.error(f"Failed to process image batch {start}-{start+len(batch)}: {e}")
                failed_batches += 1
                pbar.update(len(batch))
    
    # Link ImageChunk → Document nodes
    logger.info("Linking ImageChunk nodes to Document nodes...")
    try:
        result = components.graph.query("""
            MATCH (i:ImageChunk)
            MATCH (d:Document)
            WHERE d.id = i.id 
            AND d.source_type = 'image'
            MERGE (i)-[:HAS_SOURCE_DOCUMENT]->(d)
            RETURN count(*) as links_created
        """)
        links = result[0]["links_created"] if result else 0
        logger.info(f"Created {links} ImageChunk→Document links")
    except Exception as e:
        logger.error(f"Failed to link ImageChunk→Document: {e}")
    
    # Link ImageChunk → Entities via Document
    logger.info("Linking ImageChunk nodes to Entity nodes...")
    try:
        result = components.graph.query("""
            MATCH (i:ImageChunk)-[:HAS_SOURCE_DOCUMENT]->(d:Document)
            MATCH (d)-[:MENTIONS]->(e:__Entity__)
            MERGE (i)-[:MENTIONS]->(e)
            RETURN count(DISTINCT e) as entities_linked
        """)
        entities = result[0]["entities_linked"] if result else 0
        logger.info(f"Linked {entities} entities to ImageChunk nodes")
    except Exception as e:
        logger.error(f"Failed to link ImageChunk→Entity: {e}")
    
    stats = {
        "num_images": len(rows),
        "num_image_docs": total_graph_docs,
        "failed_batches": failed_batches,
    }
    
    logger.info(f"✓ OCR KG enrichment: {total_graph_docs} GraphDocuments, {failed_batches} failures")
    return stats


# ==========================================================
# POST-PROCESSING & OPTIMIZATION
# ==========================================================

def post_process_kg(components: KGComponents) -> Dict[str, Any]:
    """
    Post-processing steps:
    - Merge duplicate entities
    - Create page-level aggregations
    - Add computed properties
    """
    logger.info("=== POST-PROCESSING KG ===")
    
    stats = {}
    
    # Merge entities with same name (case-insensitive)
    logger.info("Merging duplicate entities...")
    try:
        result = components.graph.query("""
            MATCH (e1:__Entity__), (e2:__Entity__)
            WHERE toLower(e1.id) = toLower(e2.id)
            AND id(e1) < id(e2)
            WITH e1, e2
            LIMIT 1000
            CALL apoc.refactor.mergeNodes([e1, e2], {
                properties: 'combine',
                mergeRels: true
            })
            YIELD node
            RETURN count(*) as merged
        """)
        stats["merged_entities"] = result[0]["merged"] if result else 0
    except Exception as e:
        logger.warning(f"Entity merging failed (APOC may not be available): {e}")
        stats["merged_entities"] = 0
    
    # Create Page nodes for better navigation
    logger.info("Creating Page nodes...")
    try:
        result = components.graph.query("""
            MATCH (d:Document)
            WHERE d.page IS NOT NULL
            WITH DISTINCT d.page as page_num
            MERGE (p:Page {number: page_num})
            RETURN count(*) as pages_created
        """)
        stats["pages_created"] = result[0]["pages_created"] if result else 0
        
        # Link Documents to Pages
        components.graph.query("""
            MATCH (d:Document), (p:Page)
            WHERE d.page = p.number
            MERGE (d)-[:ON_PAGE]->(p)
        """)
    except Exception as e:
        logger.warning(f"Page node creation failed: {e}")
        stats["pages_created"] = 0
    
    # Calculate entity mention counts
    logger.info("Computing entity statistics...")
    try:
        components.graph.query("""
            MATCH (e:__Entity__)<-[:MENTIONS]-(d)
            WITH e, count(d) as mention_count
            SET e.mention_count = mention_count
        """)
    except Exception as e:
        logger.warning(f"Entity stats computation failed: {e}")
    
    logger.info(f"✓ Post-processing complete: {stats}")
    return stats


# ==========================================================
# LANGGRAPH ORCHESTRATION
# ==========================================================

class PipelineState(TypedDict):
    """Enhanced state with more tracking"""
    status: str
    current_step: str
    num_text_chunks: int
    num_image_docs: int
    meta: Dict[str, Any]
    errors: List[str]


# Global variables to hold components and config
_components: Optional[KGComponents] = None
_config: Optional[PipelineConfig] = None


def ingest_text_node(state: PipelineState) -> PipelineState:
    """LangGraph node: PDF text → KG"""
    logger.info("\n[LangGraph] Node: ingest_text_node")
    state["current_step"] = "text_ingestion"
    
    try:
        stats = ingest_pdf_text_to_kg(_components, _config)
        state["status"] = "text_ingested"
        state["num_text_chunks"] = stats["num_chunks"]
        state["meta"]["text"] = stats
    except Exception as e:
        error_msg = f"Text ingestion failed: {e}"
        logger.error(error_msg, exc_info=True)
        state["errors"].append(error_msg)
        state["status"] = "text_failed"
    
    return state


def enrich_images_node(state: PipelineState) -> PipelineState:
    """LangGraph node: OCR → KG"""
    logger.info("\n[LangGraph] Node: enrich_images_node")
    state["current_step"] = "image_enrichment"
    
    # Skip if text ingestion failed
    if state.get("status") == "text_failed":
        logger.warning("Skipping image enrichment due to text ingestion failure")
        return state
    
    try:
        stats = enrich_kg_from_image_ocr(_components, _config)
        state["status"] = "images_enriched"
        state["num_image_docs"] = stats["num_image_docs"]
        state["meta"]["images"] = stats
    except Exception as e:
        error_msg = f"Image enrichment failed: {e}"
        logger.error(error_msg, exc_info=True)
        state["errors"].append(error_msg)
        state["status"] = "images_failed"
    
    return state


def post_process_node(state: PipelineState) -> PipelineState:
    """LangGraph node: Post-processing"""
    logger.info("\n[LangGraph] Node: post_process_node")
    state["current_step"] = "post_processing"
    
    try:
        stats = post_process_kg(_components)
        state["status"] = "completed"
        state["meta"]["post_processing"] = stats
    except Exception as e:
        error_msg = f"Post-processing failed: {e}"
        logger.error(error_msg, exc_info=True)
        state["errors"].append(error_msg)
    
    return state


def build_workflow():
    """
    Enhanced LangGraph workflow:
    START → ingest_text → enrich_images → post_process → END
    """
    workflow = StateGraph(PipelineState)
    
    workflow.add_node("ingest_text", ingest_text_node)
    workflow.add_node("enrich_images", enrich_images_node)
    workflow.add_node("post_process", post_process_node)
    
    workflow.add_edge(START, "ingest_text")
    workflow.add_edge("ingest_text", "enrich_images")
    workflow.add_edge("enrich_images", "post_process")
    workflow.add_edge("post_process", END)
    
    return workflow.compile()


# ==========================================================
# MAIN ENTRYPOINT
# ==========================================================

def main():
    """Main execution with proper cleanup"""
    global _components, _config
    
    logger.info("="*60)
    logger.info("MULTIMODAL KNOWLEDGE GRAPH PIPELINE")
    logger.info("="*60)
    
    # Load configuration
    try:
        _config = PipelineConfig()
        logger.info("✓ Configuration loaded")
    except Exception as e:
        logger.error(f"Configuration failed: {e}")
        return 1
    
    # Initialize components
    try:
        _components = KGComponents(_config)
        _components.create_indexes()
    except Exception as e:
        logger.error(f"Component initialization failed: {e}")
        return 1
    
    # Build workflow
    app = build_workflow()
    
    # Initial state (without components and config)
    initial_state: PipelineState = {
        "status": "start",
        "current_step": "initialization",
        "num_text_chunks": 0,
        "num_image_docs": 0,
        "meta": {},
        "errors": [],
    }
    
    # Run pipeline
    try:
        final_state = app.invoke(initial_state)
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED")
        logger.info("="*60)
        logger.info(f"Final status: {final_state['status']}")
        logger.info(f"Text chunks processed: {final_state['num_text_chunks']}")
        logger.info(f"Image OCR docs processed: {final_state['num_image_docs']}")
        
        if final_state.get("errors"):
            logger.warning(f"Errors encountered: {len(final_state['errors'])}")
            for error in final_state["errors"]:
                logger.warning(f"  - {error}")
        
        logger.info(f"\nDetailed stats: {final_state['meta']}")
        
        # Refresh schema
        _components.refresh_schema()
        
        return 0 if final_state["status"] == "completed" else 1
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return 1
    
    finally:
        # Cleanup
        if _components:
            _components.close()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)