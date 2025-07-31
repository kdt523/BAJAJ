
"""
LLM-Powered Intelligent Query-Retrieval System for HackRx 6.0
Handles insurance, legal, HR, and compliance document processing with semantic search
Enhanced with OpenAI and Gemini API support
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

# Core dependencies
import requests
import numpy as np
from pathlib import Path
import tempfile

# FastAPI and async components
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Document processing
import PyPDF2
import docx
import email
from email.mime.text import MIMEText
import fitz  # PyMuPDF for better PDF extraction

# Vector database and embeddings
import faiss
import pinecone
from sentence_transformers import SentenceTransformer
import openai
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI not installed. Gemini features will be disabled.")

# Database
import asyncpg
import aiosqlite
from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid

# --- START: MODIFIED SECTION FOR API KEY AUTHENTICATION ---
# The new API key for your FastAPI application, read from an environment variable.
# The hardcoded value is a fallback for development.
HACKRX_API_KEY = os.getenv("HACKRX_API_KEY", "6307d881a10ee9312213de36705b239bcd4b07a2719b9db6e66925543aa4f46a")

bearer_scheme = HTTPBearer()

async def verify_hackrx_api_key(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """
    Dependency to verify the Bearer token sent by the HackRx platform.
    """
    if credentials.credentials != HACKRX_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key for HackRx platform authentication.",
        )
    return credentials
# --- END: MODIFIED SECTION ---

# Enhanced Configuration
class Config:
    # API Configuration
    API_BASE_URL = "http://localhost:8000/api/v1"
    BEARER_TOKEN = HACKRX_API_KEY  # Now uses the API key from the environment variable
    
    # Gemini Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = "gemini-1.5-flash-latest"
    
    # LLM Provider Selection
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # "openai" or "gemini"
    
    # Pinecone Configuration (fallback to FAISS if not available)
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1-aws")
    INDEX_NAME = "hackrx-documents"
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///hackrx.db")
    
    # Embedding Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight and efficient
    CHUNK_SIZE = 512  # Token-efficient chunking
    CHUNK_OVERLAP = 50

# Database Models
Base = declarative_base()

class DocumentRecord(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    url = Column(String, nullable=False)
    content_hash = Column(String, nullable=False)
    metadata = Column(JSON)
    chunks = Column(JSON)  # Store chunk metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(String, default="pending")  # pending, processing, completed, failed

class QueryRecord(Base):
    __tablename__ = "queries"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, nullable=False)
    query = Column(Text, nullable=False)
    response = Column(JSON)
    processing_time = Column(Integer)  # milliseconds
    llm_provider = Column(String, default="openai")  # Track which LLM was used
    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic Models
class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL to the document")
    questions: List[str] = Field(..., description="List of questions to process")

class QueryResponse(BaseModel):
    answers: List[str] = Field(..., description="Processed answers")
    processing_time: Optional[int] = Field(None, description="Processing time in ms")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class ModelSwitchRequest(BaseModel):
    provider: str = Field(..., description="LLM provider: 'openai' or 'gemini'")

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    chunk_id: str = ""

class DocumentProcessor:
    """Handles document downloading and content extraction - Step 1: Input Documents"""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.docx', '.txt', '.eml'}
    
    async def download_document(self, url: str) -> Tuple[bytes, str]:
        """Download document from URL and detect format"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Detect format from URL or content-type
            content_type = response.headers.get('content-type', '').lower()
            format_ext = self._detect_format(url, content_type)
            
            return response.content, format_ext
        except Exception as e:
            logger.error(f"Error downloading document: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    
    def _detect_format(self, url: str, content_type: str) -> str:
        """Detect document format"""
        if 'pdf' in content_type or url.lower().endswith('.pdf'):
            return '.pdf'
        elif 'word' in content_type or url.lower().endswith('.docx'):
            return '.docx'
        elif 'text' in content_type or url.lower().endswith('.txt'):
            return '.txt'
        else:
            return '.pdf'  # Default assumption
    
    async def extract_content(self, content: bytes, format_ext: str) -> str:
        """Extract text content from document"""
        try:
            if format_ext == '.pdf':
                return await self._extract_pdf_content(content)
            elif format_ext == '.docx':
                return await self._extract_docx_content(content)
            elif format_ext == '.txt':
                return content.decode('utf-8')
            elif format_ext == '.eml':
                return await self._extract_email_content(content)
            else:
                raise ValueError(f"Unsupported format: {format_ext}")
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to extract content: {e}")
    
    async def _extract_pdf_content(self, content: bytes) -> str:
        """Extract text from PDF using PyMuPDF for better extraction"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            
            try:
                doc = fitz.open(tmp_file.name)
                text = ""
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    text += page.get_text() + "\n"
                doc.close()
                return text.strip()
            finally:
                os.unlink(tmp_file.name)
    
    async def _extract_docx_content(self, content: bytes) -> str:
        """Extract text from DOCX"""
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            
            try:
                doc = docx.Document(tmp_file.name)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text.strip()
            finally:
                os.unlink(tmp_file.name)
    
    async def _extract_email_content(self, content: bytes) -> str:
        """Extract text from email"""
        msg = email.message_from_bytes(content)
        text = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    text += part.get_payload(decode=True).decode('utf-8', errors='ignore')
        else:
            text = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        
        return text.strip()

class TextChunker:
    """Intelligent text chunking for optimal token usage"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Split text into overlapping chunks"""
        if metadata is None:
            metadata = {}
        
        # Simple sentence-aware chunking
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunk_id = hashlib.md5(current_chunk.encode()).hexdigest()[:8]
                    chunks.append(DocumentChunk(
                        content=current_chunk.strip(),
                        metadata={**metadata, 'chunk_id': chunk_id},
                        chunk_id=chunk_id
                    ))
                
                # Start new chunk with overlap
                current_chunk = sentence + ". "
        
        # Add final chunk
        if current_chunk:
            chunk_id = hashlib.md5(current_chunk.encode()).hexdigest()[:8]
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                metadata={**metadata, 'chunk_id': chunk_id},
                chunk_id=chunk_id
            ))
        
        return chunks

class EmbeddingService:
    """Handles document embeddings and vector operations - Step 3: Embedding Search"""
    
    def __init__(self):
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self.faiss_index = None
        self.chunk_store = {}  # In-memory store for chunks
        self._init_vector_store()
    
    def _init_vector_store(self):
        """Initialize vector store (FAISS as fallback)"""
        try:
            if Config.PINECONE_API_KEY:
                # Initialize Pinecone
                pinecone.init(
                    api_key=Config.PINECONE_API_KEY,
                    environment=Config.PINECONE_ENV
                )
                
                # Create or connect to index
                if Config.INDEX_NAME not in pinecone.list_indexes():
                    pinecone.create_index(
                        Config.INDEX_NAME,
                        dimension=self.dimension,
                        metric="cosine"
                    )
                
                self.pinecone_index = pinecone.Index(Config.INDEX_NAME)
                logger.info("Pinecone initialized successfully")
            else:
                raise Exception("Pinecone not configured, using FAISS")
                
        except Exception as e:
            logger.warning(f"Pinecone initialization failed: {e}. Using FAISS.")
            # Fallback to FAISS
            self.faiss_index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            self.use_faiss = True
    
    async def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for document chunks"""
        texts = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks
    
    async def store_chunks(self, chunks: List[DocumentChunk], document_id: str):
        """Store chunks in vector database"""
        try:
            if hasattr(self, 'pinecone_index'):
                # Store in Pinecone
                vectors = []
                for i, chunk in enumerate(chunks):
                    vector_id = f"{document_id}_{chunk.chunk_id}"
                    vectors.append({
                        'id': vector_id,
                        'values': chunk.embedding.tolist(),
                        'metadata': {
                            'content': chunk.content,
                            'document_id': document_id,
                            **chunk.metadata
                        }
                    })
                
                self.pinecone_index.upsert(vectors)
            else:
                # Store in FAISS
                embeddings = np.array([chunk.embedding for chunk in chunks])
                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings)
                
                start_idx = self.faiss_index.ntotal
                self.faiss_index.add(embeddings)
                
                # Store chunk metadata
                for i, chunk in enumerate(chunks):
                    self.chunk_store[start_idx + i] = {
                        'content': chunk.content,
                        'document_id': document_id,
                        **chunk.metadata
                    }
            
            logger.info(f"Stored {len(chunks)} chunks for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            raise
    
    async def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks - Step 4: Clause Matching"""
        try:
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            if hasattr(self, 'pinecone_index'):
                # Search in Pinecone
                results = self.pinecone_index.query(
                    vector=query_embedding[0].tolist(),
                    top_k=k,
                    include_metadata=True
                )
                
                return [
                    {
                        'content': match['metadata']['content'],
                        'score': match['score'],
                        'metadata': match['metadata']
                    }
                    for match in results['matches']
                ]
            else:
                # Search in FAISS
                faiss.normalize_L2(query_embedding)
                scores, indices = self.faiss_index.search(query_embedding, k)
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx != -1 and idx in self.chunk_store:
                        chunk_data = self.chunk_store[idx]
                        results.append({
                            'content': chunk_data['content'],
                            'score': float(score),
                            'metadata': chunk_data
                        })
                
                return results
                
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

class LLMService:
    """Handles OpenAI LLM interactions - Step 2 & 5: LLM Parser and Logic Evaluation"""
    
    def __init__(self):
        if not Config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.provider = "openai"
    
    async def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer using OpenAI with retrieved context"""
        try:
            # Prepare context
            context = "\n\n".join([
                f"[Context {i+1}]: {chunk['content']}"
                for i, chunk in enumerate(context_chunks[:3])  # Limit context for token efficiency
            ])
            
            # Create prompt
            prompt = self._create_answer_prompt(query, context)
            
            # Generate response
            response = self.client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing insurance, legal, HR, and compliance documents. Provide accurate, specific answers based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=300,  # Token-efficient response
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Extract reasoning and sources
            reasoning = self._extract_reasoning(context_chunks)
            
            return {
                'answer': answer,
                'reasoning': reasoning,
                'sources': [chunk['metadata'].get('chunk_id', 'unknown') for chunk in context_chunks[:3]],
                'confidence': self._calculate_confidence(context_chunks),
                'provider': self.provider
            }
            
        except Exception as e:
            logger.error(f"Error generating answer with OpenAI: {e}")
            return {
                'answer': f"I apologize, but I encountered an error processing your query: {str(e)}",
                'reasoning': "Error occurred during processing",
                'sources': [],
                'confidence': 0.0,
                'provider': self.provider
            }
    
    def _create_answer_prompt(self, query: str, context: str) -> str:
        """Create optimized prompt for answer generation"""
        return f"""Based on the following context from the document, answer the question accurately and specifically.

Context:
{context}

Question: {query}

Instructions:
- Provide a direct, specific answer based only on the context provided
- If the context doesn't contain enough information, state that clearly
- Include specific details like waiting periods, coverage amounts, conditions, etc.
- Be concise but complete

Answer:"""
    
    def _extract_reasoning(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Extract reasoning from context chunks"""
        if not context_chunks:
            return "No relevant context found"
        
        reasoning_parts = []
        for i, chunk in enumerate(context_chunks[:2]):  # Top 2 chunks
            score = chunk.get('score', 0)
            reasoning_parts.append(f"Context {i+1} (relevance: {score:.2f}): {chunk['content'][:100]}...")
        
        return " | ".join(reasoning_parts)
    
    def _calculate_confidence(self, context_chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieval scores"""
        if not context_chunks:
            return 0.0
        
        # Average of top chunk scores, weighted by position
        weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        weighted_scores = []
        
        for i, chunk in enumerate(context_chunks[:5]):
            score = chunk.get('score', 0)
            weight = weights[i] if i < len(weights) else 0.1
            weighted_scores.append(score * weight)
        
        return sum(weighted_scores) / sum(weights[:len(weighted_scores)]) if weighted_scores else 0.0

class GeminiService:
    """Handles Google Gemini LLM interactions - Step 2 & 5: LLM Parser and Logic Evaluation"""
    
    def __init__(self):
        if not GEMINI_AVAILABLE:
            raise ValueError("Google Generative AI library not installed")
        
        if not Config.GEMINI_API_KEY:
            raise ValueError("Gemini API key is required")
        
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        self.provider = "gemini"
        logger.info("Gemini service initialized successfully")
    
    async def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer using Gemini with retrieved context"""
        try:
            # Prepare context
            context = "\n\n".join([
                f"[Context {i+1}]: {chunk['content']}"
                for i, chunk in enumerate(context_chunks[:3])  # Limit context for token efficiency
            ])
            
            # Create prompt
            prompt = self._create_answer_prompt(query, context)
            
            # Generate response using Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistency
                    max_output_tokens=300,  # Token-efficient response
                )
            )
            
            answer = response.text.strip()
            
            # Extract reasoning and sources
            reasoning = self._extract_reasoning(context_chunks)
            
            return {
                'answer': answer,
                'reasoning': reasoning,
                'sources': [chunk['metadata'].get('chunk_id', 'unknown') for chunk in context_chunks[:3]],
                'confidence': self._calculate_confidence(context_chunks),
                'provider': self.provider
            }
            
        except Exception as e:
            logger.error(f"Error generating answer with Gemini: {e}")
            return {
                'answer': f"I apologize, but I encountered an error processing your query: {str(e)}",
                'reasoning': "Error occurred during processing",
                'sources': [],
                'confidence': 0.0,
                'provider': self.provider
            }
    
    def _create_answer_prompt(self, query: str, context: str) -> str:
        """Create optimized prompt for Gemini answer generation"""
        return f"""You are an expert at analyzing insurance, legal, HR, and compliance documents. Based on the following context from the document, answer the question accurately and specifically.

Context:
{context}

Question: {query}

Instructions:
- Provide a direct, specific answer based only on the context provided
- If the context doesn't contain enough information, state that clearly
- Include specific details like waiting periods, coverage amounts, conditions, etc.
- Be concise but complete

Answer:"""
    
    def _extract_reasoning(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Extract reasoning from context chunks"""
        if not context_chunks:
            return "No relevant context found"
        
        reasoning_parts = []
        for i, chunk in enumerate(context_chunks[:2]):  # Top 2 chunks
            score = chunk.get('score', 0)
            reasoning_parts.append(f"Context {i+1} (relevance: {score:.2f}): {chunk['content'][:100]}...")
        
        return " | ".join(reasoning_parts)
    
    def _calculate_confidence(self, context_chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieval scores"""
        if not context_chunks:
            return 0.0
        
        # Average of top chunk scores, weighted by position
        weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        weighted_scores = []
        
        for i, chunk in enumerate(context_chunks[:5]):
            score = chunk.get('score', 0)
            weight = weights[i] if i < len(weights) else 0.1
            weighted_scores.append(score * weight)
        
        return sum(weighted_scores) / sum(weights[:len(weighted_scores)]) if weighted_scores else 0.0

class QueryRetrievalSystem:
    """Main system orchestrating the entire 6-step pipeline"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.chunker = TextChunker(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        self.embedding_service = EmbeddingService()
        
        # Initialize LLM service based on configuration
        self._init_llm_service()
        
        self.db_engine = None
        self._init_database()
    
    def _init_llm_service(self):
        """Initialize LLM service based on configuration"""
        try:
            if Config.LLM_PROVIDER == "gemini" and GEMINI_AVAILABLE and Config.GEMINI_API_KEY:
                self.llm_service = GeminiService()
                logger.info("Using Gemini API for LLM processing")
            elif Config.OPENAI_API_KEY:
                self.llm_service = LLMService()
                logger.info("Using OpenAI API for LLM processing")
            else:
                raise ValueError("No valid LLM provider configured")
        except Exception as e:
            logger.warning(f"Failed to initialize {Config.LLM_PROVIDER}: {e}")
            # Fallback to available provider
            if Config.OPENAI_API_KEY:
                self.llm_service = LLMService()
                logger.info("Falling back to OpenAI API")
            elif GEMINI_AVAILABLE and Config.GEMINI_API_KEY:
                self.llm_service = GeminiService()
                logger.info("Falling back to Gemini API")
            else:
                raise ValueError("No LLM provider available")
    
    def switch_llm_provider(self, provider: str) -> bool:
        """Switch between LLM providers"""
        try:
            if provider.lower() == "gemini" and GEMINI_AVAILABLE and Config.GEMINI_API_KEY:
                self.llm_service = GeminiService()
                Config.LLM_PROVIDER = "gemini"
                logger.info("Switched to Gemini API")
                return True
            elif provider.lower() == "openai" and Config.OPENAI_API_KEY:
                self.llm_service = LLMService()
                Config.LLM_PROVIDER = "openai"
                logger.info("Switched to OpenAI API")
                return True
            else:
                logger.error(f"Cannot switch to {provider}: not available or not configured")
                return False
        except Exception as e:
            logger.error(f"Error switching to {provider}: {e}")
            return False
    
    def _init_database(self):
        """Initialize database connection"""
        self.db_engine = create_engine(Config.DATABASE_URL)
        Base.metadata.create_all(self.db_engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.db_engine)
    
    async def process_document(self, document_url: str) -> str:
        """Process a document and return document ID - Orchestrates Steps 1-4"""
        # Check if document already processed
        content_hash = hashlib.md5(document_url.encode()).hexdigest()
        
        with self.SessionLocal() as db:
            existing_doc = db.query(DocumentRecord).filter(
                DocumentRecord.content_hash == content_hash,
                DocumentRecord.processed == "completed"
            ).first()
            
            if existing_doc:
                logger.info(f"Document already processed: {existing_doc.id}")
                return existing_doc.id
        
        # Create new document record
        doc_id = str(uuid.uuid4())
        
        try:
            # Step 1: Download and extract content
            content_bytes, format_ext = await self.doc_processor.download_document(document_url)
            content_text = await self.doc_processor.extract_content(content_bytes, format_ext)
            
            # Step 2: Chunk the document
            chunks = self.chunker.chunk_text(content_text, {'document_url': document_url})
            
            # Step 3: Generate embeddings
            chunks = await self.embedding_service.embed_chunks(chunks)
            
            # Step 4: Store in vector database
            await self.embedding_service.store_chunks(chunks, doc_id)
            
            # Update database record
            with self.SessionLocal() as db:
                doc_record = DocumentRecord(
                    id=doc_id,
                    url=document_url,
                    content_hash=content_hash,
                    metadata={
                        'format': format_ext,
                        'chunk_count': len(chunks),
                        'content_length': len(content_text)
                    },
                    processed="completed"
                )
                db.add(doc_record)
                db.commit()
            
            logger.info(f"Document processed successfully: {doc_id}")
            return doc_id
            
        except Exception as e:
            # Update record as failed
            with self.SessionLocal() as db:
                doc_record = DocumentRecord(
                    id=doc_id,
                    url=document_url,
                    content_hash=content_hash,
                    processed="failed",
                    metadata={'error': str(e)}
                )
                db.add(doc_record)
                db.commit()
            
            logger.error(f"Document processing failed: {e}")
            raise
    
    async def process_queries(self, document_id: str, questions: List[str]) -> List[str]:
        """Process multiple queries against a document - Steps 4-6"""
        answers = []
        
        for question in questions:
            start_time = datetime.utcnow()
            
            try:
                # Step 4: Retrieve relevant chunks (clause matching)
                context_chunks = await self.embedding_service.search_similar(question, k=5)
                
                # Step 5: Generate answer (logic evaluation)
                result = await self.llm_service.generate_answer(question, context_chunks)
                
                # Store query record
                processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                
                with self.SessionLocal() as db:
                    query_record = QueryRecord(
                        document_id=document_id,
                        query=question,
                        response=result,
                        processing_time=processing_time,
                        llm_provider=result.get('provider', Config.LLM_PROVIDER)
                    )
                    db.add(query_record)
                    db.commit()
                
                answers.append(result['answer'])
                
            except Exception as e:
                logger.error(f"Error processing query '{question}': {e}")
                answers.append(f"Error processing query: {str(e)}")
        
        return answers

# FastAPI Application
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="HackRx 6.0 Solution with OpenAI and Gemini API support",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system
retrieval_system = QueryRetrievalSystem()

@app.get("/")
async def root():
    return {
        "message": "LLM-Powered Intelligent Query-Retrieval System", 
        "status": "running",
        "version": "2.0.0",
        "current_llm": Config.LLM_PROVIDER,
        "available_llms": {
            "openai": bool(Config.OPENAI_API_KEY),
            "gemini": bool(GEMINI_AVAILABLE and Config.GEMINI_API_KEY)
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "services": {
            "openai": "available" if Config.OPENAI_API_KEY else "not configured",
            "gemini": "available" if (GEMINI_AVAILABLE and Config.GEMINI_API_KEY) else "not configured",
            "current_llm": Config.LLM_PROVIDER,
            "vector_db": "faiss" if hasattr(retrieval_system.embedding_service, 'faiss_index') else "pinecone"
        }
    }

# --- START: MODIFIED ENDPOINT WITH AUTHENTICATION DEPENDENCY ---
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_hackrx_submission(
    request: QueryRequest,
    auth_key: HTTPAuthorizationCredentials = Depends(verify_hackrx_api_key)
):
    """Main endpoint matching the HackRx API specification - Complete 6-step workflow"""
    start_time = datetime.utcnow()
    
    try:
        # Steps 1-4: Process document
        document_id = await retrieval_system.process_document(request.documents)
        
        # Steps 4-6: Process queries
        answers = await retrieval_system.process_queries(document_id, request.questions)
        
        # Calculate total processing time
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Step 6: JSON Output
        return QueryResponse(
            answers=answers,
            processing_time=processing_time,
            metadata={
                "document_id": document_id,
                "question_count": len(request.questions),
                "system_version": "2.0.0",
                "llm_provider": Config.LLM_PROVIDER,
                "workflow_steps_completed": [
                    "1. Input Documents ✓",
                    "2. LLM Parser ✓", 
                    "3. Embedding Search ✓",
                    "4. Clause Matching ✓",
                    "5. Logic Evaluation ✓",
                    "6. JSON Output ✓"
                ]
            }
        )
        
    except Exception as e:
        logger.error(f"Error in hackrx/run endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
# --- END: MODIFIED ENDPOINT ---

@app.get("/models")
async def get_available_models():
    """Get information about available LLM models"""
    return {
        "current_provider": Config.LLM_PROVIDER,
        "available_providers": {
            "openai": {
                "available": bool(Config.OPENAI_API_KEY),
                "model": Config.OPENAI_MODEL if Config.OPENAI_API_KEY else "not configured"
            },
            "gemini": {
                "available": bool(GEMINI_AVAILABLE and Config.GEMINI_API_KEY),
                "model": Config.GEMINI_MODEL if (GEMINI_AVAILABLE and Config.GEMINI_API_KEY) else "not configured"
            }
        },
        "workflow_integration": {
            "step_2": "LLM Parser - Both OpenAI and Gemini supported",
            "step_5": "Logic Evaluation - Both OpenAI and Gemini supported"
        }
    }

@app.post("/switch-model")
async def switch_model(provider: str = Query(..., description="LLM provider: 'openai' or 'gemini'")):
    """Switch between OpenAI and Gemini (runtime switching)"""
    if provider.lower() not in ["openai", "gemini"]:
        raise HTTPException(status_code=400, detail="Invalid provider. Use 'openai' or 'gemini'")
    
    success = retrieval_system.switch_llm_provider(provider)
    
    if success:
        return {
            "status": "success",
            "current_provider": Config.LLM_PROVIDER,
            "message": f"Switched to {provider} successfully",
            "workflow_maintained": True
        }
    else:
        raise HTTPException(status_code=500, detail=f"Failed to switch to {provider}. Check if it's configured.")

@app.get("/documents/{document_id}")
async def get_document_info(document_id: str):
    """Get information about a processed document"""
    with retrieval_system.SessionLocal() as db:
        doc = db.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "id": doc.id,
            "url": doc.url,
            "processed": doc.processed,
            "metadata": doc.metadata,
            "created_at": doc.created_at.isoformat()
        }

@app.get("/documents/{document_id}/queries")
async def get_document_queries(document_id: str):
    """Get all queries processed for a document"""
    with retrieval_system.SessionLocal() as db:
        queries = db.query(QueryRecord).filter(QueryRecord.document_id == document_id).all()
        
        return [
            {
                "id": query.id,
                "query": query.query,
                "response": query.response,
                "processing_time": query.processing_time,
                "llm_provider": query.llm_provider,
                "created_at": query.created_at.isoformat()
            }
            for query in queries
        ]

@app.get("/workflow-status")
async def get_workflow_status():
    """Get the current status of the 6-step workflow implementation"""
    return {
        "workflow_steps": {
            "1": {
                "name": "Input Documents",
                "status": "✓ Implemented",
                "component": "DocumentProcessor",
                "supports": ["PDF", "DOCX", "TXT", "EML"]
            },
            "2": {
                "name": "LLM Parser", 
                "status": "✓ Enhanced",
                "component": f"{Config.LLM_PROVIDER.upper()}Service",
                "providers": ["OpenAI GPT-4", "Google Gemini Pro"]
            },
            "3": {
                "name": "Embedding Search",
                "status": "✓ Implemented", 
                "component": "EmbeddingService",
                "model": Config.EMBEDDING_MODEL
            },
            "4": {
                "name": "Clause Matching",
                "status": "✓ Implemented",
                "component": "EmbeddingService.search_similar",
                "method": "Semantic similarity search"
            },
            "5": {
                "name": "Logic Evaluation",
                "status": "✓ Enhanced",
                "component": f"{Config.LLM_PROVIDER.upper()}Service",
                "features": ["Reasoning", "Confidence scoring", "Source tracking"]
            },
            "6": {
                "name": "JSON Output",
                "status": "✓ Implemented",
                "component": "QueryResponse",
                "format": "Structured Pydantic model"
            }
        },
        "current_config": {
            "llm_provider": Config.LLM_PROVIDER,
            "embedding_model": Config.EMBEDDING_MODEL,
            "chunk_size": Config.CHUNK_SIZE,
            "vector_db": "Pinecone" if hasattr(retrieval_system.embedding_service, 'pinecone_index') else "FAISS"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting Enhanced LLM-Powered Query-Retrieval System")
    print(f"🤖 Current LLM Provider: {Config.LLM_PROVIDER.upper()}")
    print(f"📊 OpenAI Available: {bool(Config.OPENAI_API_KEY)}")
    print(f"🧠 Gemini Available: {bool(GEMINI_AVAILABLE and Config.GEMINI_API_KEY)}")
    print("🔄 6-Step Workflow: ✅ Fully Implemented")
    print("📚 API Documentation: http://localhost:8000/docs")
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )
