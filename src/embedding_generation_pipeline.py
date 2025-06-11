import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Embedding model imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  sentence-transformers not installed. Run: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    print("⚠️  openai not installed. Run: pip install openai")
    OPENAI_AVAILABLE = False

# Vector database imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("⚠️  faiss not installed. Run: pip install faiss-cpu")
    FAISS_AVAILABLE = False

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    print("⚠️  chromadb not installed. Run: pip install chromadb")
    CHROMADB_AVAILABLE = False

class EmbeddingGenerator:
    """
    Generate embeddings for multilingual document chunks
    Supports multiple embedding models and vector databases
    """
    
    def __init__(self, 
                 processed_data_path: str = "data/processed",
                 embeddings_path: str = "data/embeddings",
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        
        self.processed_data_path = Path(processed_data_path)
        self.embeddings_path = Path(embeddings_path)
        self.embeddings_path.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        
        # Storage for embeddings and metadata
        self.chunks_data = []
        self.embeddings = None
        self.chunk_metadata = []
        
    def initialize_embedding_model(self, model_name: Optional[str] = None):
        """
        Initialize the embedding model
        """
        if model_name:
            self.model_name = model_name
            
        print(f"🔮 Initializing embedding model: {self.model_name}")
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required. Run: pip install sentence-transformers")
        
        # Load multilingual embedding model
        self.model = SentenceTransformer(self.model_name)
        
        # Get embedding dimension
        test_embedding = self.model.encode(["test"])
        self.embedding_dim = test_embedding.shape[1]
        
        print(f"✅ Model loaded successfully")
        print(f"📐 Embedding dimension: {self.embedding_dim}")
        
        return self.model
    
    def load_processed_documents(self) -> List[Dict]:
        """
        Load all processed documents and extract chunks
        """
        print("📚 Loading processed documents...")
        
        processed_files = list(self.processed_data_path.glob("*_processed.json"))
        
        if not processed_files:
            raise FileNotFoundError(f"No processed files found in {self.processed_data_path}")
        
        self.chunks_data = []
        doc_count = 0
        total_chunks = 0
        
        for file_path in processed_files:
            print(f"  📄 Loading: {file_path.stem}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            # Extract chunks with metadata
            chunks = doc_data.get('chunks', [])
            metadata = doc_data.get('metadata', {})
            
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'chunk_id': f"{metadata.get('doc_id', file_path.stem)}_chunk_{i}",
                    'doc_id': metadata.get('doc_id', file_path.stem),
                    'chunk_index': i,
                    'text': chunk,
                    'word_count': len(chunk.split()),
                    'source': metadata.get('source', 'Unknown'),
                    'language': metadata.get('language', 'unknown'),
                    'doc_type': metadata.get('doc_type', 'unknown'),
                    'title': metadata.get('title', ''),
                    'url': metadata.get('url', ''),
                    'topic_tags': metadata.get('topic_tags', [])
                }
                
                self.chunks_data.append(chunk_data)
                total_chunks += 1
            
            doc_count += 1
        
        print(f"✅ Loaded {doc_count} documents with {total_chunks} chunks")
        
        # Show distribution by language
        lang_dist = {}
        for chunk in self.chunks_data:
            lang = chunk['language']
            lang_dist[lang] = lang_dist.get(lang, 0) + 1
        
        print(f"🌐 Language distribution: {lang_dist}")
        
        return self.chunks_data
    
    def generate_embeddings(self, batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for all chunks
        """
        if not self.model:
            self.initialize_embedding_model()
        
        if not self.chunks_data:
            self.load_processed_documents()
        
        print(f"🔮 Generating embeddings for {len(self.chunks_data)} chunks...")
        print(f"📦 Batch size: {batch_size}")
        
        # Extract text for embedding
        texts = [chunk['text'] for chunk in self.chunks_data]
        
        # Generate embeddings in batches
        embeddings_list = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_end = min(i + batch_size, len(texts))
            
            print(f"  Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} "
                  f"(chunks {i+1}-{batch_end})")
            
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            embeddings_list.append(batch_embeddings)
        
        # Combine all embeddings
        self.embeddings = np.vstack(embeddings_list)
        
        print(f"✅ Generated embeddings: {self.embeddings.shape}")
        print(f"📐 Shape: {len(self.chunks_data)} chunks × {self.embedding_dim} dimensions")
        
        return self.embeddings
    
    def save_embeddings(self, format_type: str = "both"):
        """
        Save embeddings and metadata in various formats
        """
        print(f"💾 Saving embeddings in {format_type} format(s)...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save embeddings as numpy array
        if format_type in ["numpy", "both"]:
            embeddings_file = self.embeddings_path / f"embeddings_{timestamp}.npy"
            np.save(embeddings_file, self.embeddings)
            print(f"  ✅ Embeddings saved: {embeddings_file}")
        
        # Save metadata as JSON
        metadata_file = self.embeddings_path / f"chunks_metadata_{timestamp}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunks_data, f, ensure_ascii=False, indent=2)
        print(f"  ✅ Metadata saved: {metadata_file}")
        
        # Save combined data as pickle (for easy loading)
        if format_type in ["pickle", "both"]:
            combined_data = {
                'embeddings': self.embeddings,
                'chunks_metadata': self.chunks_data,
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'generation_date': datetime.now().isoformat(),
                'total_chunks': len(self.chunks_data)
            }
            
            pickle_file = self.embeddings_path / f"embeddings_complete_{timestamp}.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(combined_data, f)
            print(f"  ✅ Combined data saved: {pickle_file}")
        
        # Save configuration file
        config = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'total_chunks': len(self.chunks_data),
            'generation_date': datetime.now().isoformat(),
            'files': {
                'embeddings': f"embeddings_{timestamp}.npy",
                'metadata': f"chunks_metadata_{timestamp}.json",
                'combined': f"embeddings_complete_{timestamp}.pkl"
            }
        }
        
        config_file = self.embeddings_path / f"embedding_config_{timestamp}.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"  ✅ Configuration saved: {config_file}")
        
        return {
            'embeddings_file': embeddings_file if format_type in ["numpy", "both"] else None,
            'metadata_file': metadata_file,
            'pickle_file': pickle_file if format_type in ["pickle", "both"] else None,
            'config_file': config_file
        }
    
    def create_faiss_index(self, index_type: str = "flat") -> Tuple[object, str]:
        """
        Create FAISS index for fast similarity search
        """
        if not FAISS_AVAILABLE:
            raise ImportError("faiss required. Run: pip install faiss-cpu")
        
        if self.embeddings is None:
            raise ValueError("No embeddings generated. Run generate_embeddings() first.")
        
        print(f"🔍 Creating FAISS index ({index_type})...")
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        
        # Create appropriate index type
        if index_type == "flat":
            # Exact search (good for small datasets)
            index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
        elif index_type == "ivf":
            # Approximate search (faster for large datasets)
            nlist = min(100, len(self.chunks_data) // 4)  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            
            # Train the index
            index.train(normalized_embeddings.astype('float32'))
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add embeddings to index
        index.add(normalized_embeddings.astype('float32'))
        
        print(f"✅ FAISS index created with {index.ntotal} vectors")
        
        # Save FAISS index
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        index_file = self.embeddings_path / f"faiss_index_{index_type}_{timestamp}.index"
        faiss.write_index(index, str(index_file))
        
        print(f"💾 FAISS index saved: {index_file}")
        
        return index, str(index_file)
    
    def test_similarity_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Test similarity search with a sample query
        """
        if not self.model:
            raise ValueError("Model not initialized")
        
        if self.embeddings is None:
            raise ValueError("No embeddings generated")
        
        print(f"🔍 Testing similarity search for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Calculate similarities
        similarities = np.dot(
            self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True),
            query_embedding.T
        ).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            result = {
                'rank': i + 1,
                'similarity': float(similarities[idx]),
                'chunk_id': self.chunks_data[idx]['chunk_id'],
                'text': self.chunks_data[idx]['text'][:200] + "...",
                'source': self.chunks_data[idx]['source'],
                'language': self.chunks_data[idx]['language']
            }
            results.append(result)
            
            print(f"  {i+1}. Score: {similarities[idx]:.3f} | "
                  f"{self.chunks_data[idx]['language']} | "
                  f"{self.chunks_data[idx]['source']}")
            print(f"     Text: {self.chunks_data[idx]['text'][:100]}...")
        
        return results
    
    def generate_embedding_report(self) -> Dict:
        """
        Generate comprehensive report on embeddings
        """
        if not self.chunks_data or self.embeddings is None:
            raise ValueError("No data available for report")
        
        report = {
            'generation_date': datetime.now().isoformat(),
            'model_info': {
                'name': self.model_name,
                'embedding_dimension': self.embedding_dim
            },
            'data_summary': {
                'total_chunks': len(self.chunks_data),
                'total_documents': len(set(chunk['doc_id'] for chunk in self.chunks_data)),
                'average_chunk_length': np.mean([chunk['word_count'] for chunk in self.chunks_data]),
                'embedding_shape': self.embeddings.shape
            },
            'language_distribution': {},
            'source_distribution': {},
            'doc_type_distribution': {}
        }
        
        # Calculate distributions
        for chunk in self.chunks_data:
            # Language distribution
            lang = chunk['language']
            report['language_distribution'][lang] = report['language_distribution'].get(lang, 0) + 1
            
            # Source distribution
            source = chunk['source']
            report['source_distribution'][source] = report['source_distribution'].get(source, 0) + 1
            
            # Document type distribution
            doc_type = chunk['doc_type']
            report['doc_type_distribution'][doc_type] = report['doc_type_distribution'].get(doc_type, 0) + 1
        
        # Save report
        report_file = self.embeddings_path / f"embedding_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📊 Embedding report saved: {report_file}")
        
        return report

def main():
    """
    Main embedding generation pipeline
    """
    print("🚀 STARTING EMBEDDING GENERATION PIPELINE")
    print("=" * 55)
    
    # Initialize embedding generator
    embedding_gen = EmbeddingGenerator()
    
    # Step 1: Initialize model
    embedding_gen.initialize_embedding_model()
    
    # Step 2: Load processed documents
    embedding_gen.load_processed_documents()
    
    # Step 3: Generate embeddings
    embedding_gen.generate_embeddings(batch_size=32)
    
    # Step 4: Save embeddings
    saved_files = embedding_gen.save_embeddings(format_type="both")
    
    # Step 5: Create FAISS index
    if FAISS_AVAILABLE:
        try:
            index, index_file = embedding_gen.create_faiss_index("flat")
            print(f"✅ FAISS index ready for RAG retrieval")
        except Exception as e:
            print(f"⚠️  FAISS index creation failed: {e}")
    
    # Step 6: Test similarity search
    test_queries = [
        "What is Malaysia's poverty rate?",
        "Bagaimana cara mendaftar eKasih?",
        "SARA program assistance amount",
        "Kemiskinan tegar di Malaysia"
    ]
    
    print(f"\n🧪 TESTING SIMILARITY SEARCH")
    print("=" * 30)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            results = embedding_gen.test_similarity_search(query, top_k=3)
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
    
    # Step 7: Generate report
    report = embedding_gen.generate_embedding_report()
    
    print(f"\n🎉 EMBEDDING GENERATION COMPLETED!")
    print("=" * 40)
    print(f"📊 Total chunks: {len(embedding_gen.chunks_data)}")
    print(f"🔮 Embedding dimension: {embedding_gen.embedding_dim}")
    print(f"🌐 Languages: {list(report['language_distribution'].keys())}")
    print(f"📁 Files saved in: {embedding_gen.embeddings_path}")
    
    return embedding_gen

if __name__ == "__main__":
    main()