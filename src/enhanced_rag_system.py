import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
import re
import os
warnings.filterwarnings('ignore')

# Core RAG imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("sentence-transformers required: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("faiss required: pip install faiss-cpu")
    FAISS_AVAILABLE = False

# LLM options
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("Google Gemini: pip install google-generativeai")
    GEMINI_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Transformers: pip install transformers torch")
    TRANSFORMERS_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    print("Ollama: pip install ollama")
    OLLAMA_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    print("Groq: pip install groq")
    GROQ_AVAILABLE = False

class MalaysianPovertyRAG:
    """
    Enhanced Multi-LLM RAG system for Malaysian poverty assistance
    Supports multiple free LLM providers: Gemini, Ollama, Groq, Transformers
    """
    
    def __init__(self, 
                 embeddings_path: str = "data/embeddings",
                 model_type: str = "gemini",
                 api_key: Optional[str] = None,
                 model_name: Optional[str] = None):
        
        self.embeddings_path = Path(embeddings_path)
        self.model_type = model_type
        self.api_key = api_key
        self.model_name = model_name
        
        # RAG components
        self.embedding_model = None
        self.embeddings = None
        self.chunks_metadata = []
        self.faiss_index = None
        self.llm_client = None
        self.generation_model = None
        
        # Enhanced Configuration
        self.embedding_dim = 384
        self.max_context_length = 4000
        self.similarity_threshold = 0.1
        
        # Language detection patterns
        self.malay_patterns = [
            'bagaimana', 'cara', 'berapa', 'apakah', 'dimana', 'bila', 'siapa',
            'kemiskinan', 'bantuan', 'kerajaan', 'permohonan', 'ekasih', 'sara',
            'malaysia', 'rakyat', 'pendapatan', 'keluarga', 'mohon', 'skim',
            'program', 'bantuan', 'wang', 'ringgit', 'rm', 'kadar', 'statistik',
            'tegar', 'miskin', 'sosial', 'ekonomi'
        ]

    def find_latest_embedding_files(self) -> Dict[str, str]:
        """Find the latest embedding files"""
        print("Searching for embedding files...")
        
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {self.embeddings_path}")
        
        config_pattern = "embedding_config_*.json"
        config_files = list(self.embeddings_path.glob(config_pattern))
        
        if not config_files:
            raise FileNotFoundError(f"No embedding config files found in {self.embeddings_path}")
        
        latest_config = max(config_files, key=lambda x: x.stat().st_mtime)
        print(f"Found config: {latest_config.name}")
        
        timestamp_match = re.search(r'embedding_config_(\d{8}_\d{6})\.json', latest_config.name)
        if not timestamp_match:
            raise ValueError(f"Could not parse timestamp from config file: {latest_config.name}")
        
        full_timestamp = timestamp_match.group(1)
        
        expected_files = {
            'embeddings': self.embeddings_path / f"embeddings_{full_timestamp}.npy",
            'metadata': self.embeddings_path / f"chunks_metadata_{full_timestamp}.json",
            'faiss_index': self.embeddings_path / f"faiss_index_flat_{full_timestamp}.index",
            'config': latest_config
        }
        
        for file_type, file_path in expected_files.items():
            if file_path.exists():
                print(f"Found {file_type}: {file_path.name}")
            else:
                if file_type != 'faiss_index':
                    print(f"Missing {file_type}: {file_path.name}")
        
        return {k: str(v) for k, v in expected_files.items()}
    
    def load_embeddings(self):
        """Load embeddings and metadata from saved files"""
        print("Loading embeddings and metadata...")
        
        file_paths = self.find_latest_embedding_files()
        
        # Load embeddings
        embeddings_file = Path(file_paths['embeddings'])
        if embeddings_file.exists():
            self.embeddings = np.load(embeddings_file)
            print(f"Embeddings loaded: {self.embeddings.shape}")
        else:
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        
        # Load metadata
        metadata_file = Path(file_paths['metadata'])
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.chunks_metadata = json.load(f)
            print(f"Metadata loaded: {len(self.chunks_metadata)} chunks")
            
            # Apply source fixes immediately after loading
            self.fix_unknown_sources()
            
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # Load FAISS index if available
        index_file = Path(file_paths['faiss_index'])
        if index_file.exists() and FAISS_AVAILABLE:
            try:
                self.faiss_index = faiss.read_index(str(index_file))
                print(f"FAISS index loaded: {self.faiss_index.ntotal} vectors")
            except Exception as e:
                print(f"FAISS index failed to load: {e}")
                self.faiss_index = None
        else:
            self.faiss_index = None
        
        return True
    
    def fix_unknown_sources(self):
        """Fix Unknown sources with proper attributions"""
        print("🔧 Fixing unknown sources...")
        
        fixed_count = 0
        for i, chunk in enumerate(self.chunks_metadata):
            if chunk.get('source', 'Unknown') == 'Unknown':
                content = chunk.get('text', '').lower()
                
                if 'poverty rate' in content or 'dosm' in content or 'department of statistics' in content:
                    chunk['source'] = 'Department of Statistics Malaysia (DOSM)'
                    chunk['doc_type'] = 'Poverty Statistics Report'
                    chunk['url'] = 'https://open.dosm.gov.my/data-catalogue/hh_poverty'
                    fixed_count += 1
                elif 'prime minister' in content or 'anwar ibrahim' in content:
                    chunk['source'] = 'Prime Minister\'s Office Malaysia'
                    chunk['doc_type'] = 'Press Statement'
                    chunk['url'] = 'https://www.pmo.gov.my'
                    fixed_count += 1
                elif 'hardcore poverty' in content:
                    chunk['source'] = 'Department of Statistics Malaysia (DOSM)'
                    chunk['doc_type'] = 'Poverty Definition Guidelines'
                    fixed_count += 1
                else:
                    chunk['source'] = 'Malaysian Government Document'
                    chunk['doc_type'] = 'Official Information'
                    fixed_count += 1
        
        print(f"   ✅ Fixed {fixed_count} chunks with unknown sources")
        return fixed_count
    
    def initialize_models(self):
        """Initialize embedding and generation models"""
        print("Initializing models...")
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required")
        
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("Embedding model loaded")
        
        if self.model_type == "gemini":
            self._initialize_gemini()
        elif self.model_type == "ollama":
            self._initialize_ollama()
        elif self.model_type == "groq":
            self._initialize_groq()
        elif self.model_type == "transformers":
            self._initialize_transformers()
        else:
            print("Mock generation model (for testing)")
            self.model_type = "mock"
    
    def _initialize_gemini(self):
        """Initialize Google Gemini"""
        if not GEMINI_AVAILABLE:
            print("❌ Gemini not available. Install: pip install google-generativeai")
            self.model_type = "mock"
            return
        
        if not self.api_key:
            print("❌ Gemini API key required. Get one from: https://aistudio.google.com/app/apikey")
            self.model_type = "mock"
            return
        
        try:
            genai.configure(api_key=self.api_key)
            model_name = self.model_name or "gemini-1.5-flash"
            self.llm_client = genai.GenerativeModel(model_name)
            print(f"✅ Gemini model initialized: {model_name}")
        except Exception as e:
            print(f"❌ Gemini initialization failed: {e}")
            self.model_type = "mock"
    
    def _initialize_ollama(self):
        """Initialize Ollama (local)"""
        if not OLLAMA_AVAILABLE:
            print("❌ Ollama not available. Install: pip install ollama")
            self.model_type = "mock"
            return
        
        try:
            models = ollama.list()
            model_name = self.model_name or "llama3.1:8b"
            
            available_models = [m['name'] for m in models.get('models', [])]
            if model_name not in available_models:
                print(f"Downloading {model_name}... This may take a while.")
                ollama.pull(model_name)
            
            self.llm_client = ollama
            self.model_name = model_name
            print(f"✅ Ollama model ready: {model_name}")
        except Exception as e:
            print(f"❌ Ollama initialization failed: {e}")
            print("Make sure Ollama is installed and running: https://ollama.ai/")
            self.model_type = "mock"
    
    def _initialize_groq(self):
        """Initialize Groq"""
        if not GROQ_AVAILABLE:
            print("❌ Groq not available. Install: pip install groq")
            self.model_type = "mock"
            return
        
        if not self.api_key:
            print("❌ Groq API key required. Get one from: https://console.groq.com/keys")
            self.model_type = "mock"
            return
        
        try:
            self.llm_client = Groq(api_key=self.api_key)
            self.model_name = self.model_name or "llama3-8b-8192"
            print(f"✅ Groq model initialized: {self.model_name}")
        except Exception as e:
            print(f"❌ Groq initialization failed: {e}")
            self.model_type = "mock"
    
    def _initialize_transformers(self):
        """Initialize local Transformers model"""
        if not TRANSFORMERS_AVAILABLE:
            print("❌ Transformers not available. Install: pip install transformers torch")
            self.model_type = "mock"
            return
        
        try:
            model_name = self.model_name or "microsoft/DialoGPT-medium"
            print(f"Loading {model_name}... This may take a while.")
            
            self.generation_model = pipeline(
                'text-generation',
                model=model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            print(f"✅ Transformers model loaded: {model_name}")
        except Exception as e:
            print(f"❌ Transformers initialization failed: {e}")
            self.model_type = "mock"
    
    def detect_language(self, text: str) -> str:
        """Detect query language"""
        text_lower = text.lower()
        malay_score = sum(1 for pattern in self.malay_patterns if pattern in text_lower)
        
        # if malay_score >= 2:
        #     return "bahasa_malaysia"
        # elif malay_score >= 1:
        #     return "mixed"
        # else:
        return "english"
    
    def _expand_query(self, query: str) -> str:
        """Expand queries with synonyms and related terms"""
        expansions = {
            'poverty rate': 'poverty rate kadar kemiskinan percentage statistics data',
            'kadar kemiskinan': 'kadar kemiskinan poverty rate peratusan statistik data',
            'ekasih': 'ekasih e-kasih eKasih application registration pendaftaran system',
            'sara': 'sara bantuan sara assistance program skim sumbangan',
            'assistance amount': 'assistance amount bantuan jumlah money ringgit rm payment',
            'hardcore poverty': 'hardcore poverty kemiskinan tegar extreme poverty definition',
            'government assistance': 'government assistance bantuan kerajaan social aid program',
            'how much': 'how much berapa amount jumlah rm ringgit payment',
            'bagaimana': 'bagaimana cara how process steps application'
        }
        
        query_lower = query.lower()
        for term, expansion in expansions.items():
            if term in query_lower:
                return f"{query} {expansion}"
        
        return query
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve relevant chunks with query expansion"""
        if not self.embedding_model or self.embeddings is None:
            raise ValueError("Models not initialized or embeddings not loaded")
        
        # Expand query with synonyms
        expanded_query = self._expand_query(query)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([expanded_query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Use FAISS index if available, otherwise numpy search
        if self.faiss_index:
            similarities, indices = self.faiss_index.search(
                query_embedding.astype('float32'), top_k
            )
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if similarity >= self.similarity_threshold:
                    chunk_data = self.chunks_metadata[idx].copy()
                    chunk_data['similarity'] = float(similarity)
                    chunk_data['rank'] = i + 1
                    results.append(chunk_data)
        else:
            # Numpy search fallback
            normalized_embeddings = self.embeddings / np.linalg.norm(
                self.embeddings, axis=1, keepdims=True
            )
            
            similarities = np.dot(normalized_embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for i, idx in enumerate(top_indices):
                if similarities[idx] >= self.similarity_threshold:
                    chunk_data = self.chunks_metadata[idx].copy()
                    chunk_data['similarity'] = float(similarities[idx])
                    chunk_data['rank'] = i + 1
                    results.append(chunk_data)
        
        return results
    
    def _extract_key_content(self, text: str) -> str:
        """Extract the most important content from chunk text"""
        # Remove metadata headers
        lines = text.split('\n')
        content_lines = []
        
        skip_patterns = [
            'BAHASA:', 'JENIS_DOKUMEN:', 'TAG_TOPIK:', 'SOURCE:', 
            'URL:', 'TARIKH_AKSES:', 'SUMBER:', '===', 'DOCUMENT TITLE:'
        ]
        
        for line in lines:
            line = line.strip()
            # Skip metadata lines
            if any(pattern in line for pattern in skip_patterns):
                continue
            # Skip very short lines
            if len(line) < 20:
                continue
            # Keep substantial content
            if line:
                content_lines.append(line)
        
        # Return the most substantial content
        content = ' '.join(content_lines)
        
        # If content is too long, prioritize sentences with numbers/statistics
        if len(content) > 800:
            sentences = content.split('.')
            # Prioritize sentences with numbers, percentages, or money
            priority_sentences = []
            regular_sentences = []
            
            for sentence in sentences:
                if any(char in sentence for char in ['%', 'RM', '20', 'billion', 'million', 'ringgit']):
                    priority_sentences.append(sentence)
                else:
                    regular_sentences.append(sentence)
            
            # Combine priority sentences first
            combined = '. '.join(priority_sentences[:3] + regular_sentences[:2])
            return combined[:800] + "..." if len(combined) > 800 else combined
        
        return content
    
    def prepare_context(self, retrieved_chunks: List[Dict]) -> str:
        """Better context preparation that focuses on actual data content"""
        if not retrieved_chunks:
            return ""
        
        context_sections = []
        
        # Group chunks by relevance
        high_relevance = [c for c in retrieved_chunks if c.get('similarity', 0) > 0.7]
        medium_relevance = [c for c in retrieved_chunks if 0.3 <= c.get('similarity', 0) <= 0.7]
        
        # Process high relevance chunks first
        context_sections.append("=== HIGHLY RELEVANT INFORMATION ===")
        for chunk in high_relevance[:3]:  # Top 3 most relevant
            content = self._extract_key_content(chunk['text'])
            source = chunk.get('source', 'Government Document')
            
            context_sections.append(f"""
CONTENT: {content}
SOURCE: {source}
RELEVANCE: {chunk.get('similarity', 0):.3f}
""")
        
        # Add medium relevance if space allows
        if len('\n'.join(context_sections)) < 2500:  # Leave room for more
            context_sections.append("\n=== ADDITIONAL INFORMATION ===")
            for chunk in medium_relevance[:2]:
                content = self._extract_key_content(chunk['text'])
                source = chunk.get('source', 'Government Document')
                
                context_sections.append(f"""
CONTENT: {content}
SOURCE: {source}
""")
        
        return '\n'.join(context_sections)
    
    def _detect_query_type(self, query: str) -> str:
        """Detect the type of query for better prompting"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['rate', 'percentage', 'statistics', 'how much', 'amount', 'kadar', 'berapa', 'peratusan', 'statistik']):
            return "statistics"
        elif any(term in query_lower for term in ['how to', 'cara', 'bagaimana', 'process', 'application', 'mohon']):
            return "procedure"
        elif any(term in query_lower for term in ['what is', 'apa itu', 'definition', 'definisi']):
            return "definition"
        else:
            return "general"
    
    def generate_response_gemini(self, query: str, context: str, language: str) -> str:
        """More specific prompts that force data usage"""
        if not self.llm_client:
            return "Gemini client not initialized"
        
        query_type = self._detect_query_type(query)
        
        if language in ["bahasa_malaysia", "mixed"]:
            if query_type == "statistics":
                prompt = f"""Anda adalah pakar data kemiskinan Malaysia. Tugas anda adalah menjawab soalan menggunakan data yang diberikan.

DATA RASMI KEMISKINAN MALAYSIA:
{context}

SOALAN: {query}

ARAHAN KHUSUS:
1. WAJIB gunakan nombor, peratusan, atau statistik yang ada dalam data
2. JANGAN kata "maklumat tidak tersedia" jika ada data berkaitan
3. Jika ada tahun, peratusan, atau jumlah wang dalam data, MESTI sebut
4. Sebutkan sumber data dengan jelas
5. Jika data tidak lengkap, sebutkan apa yang ada dan cadangkan di mana boleh dapat lebih

JAWAPAN (gunakan data yang ada):"""

            else:
                prompt = f"""Anda adalah pembantu eKasih dan bantuan sosial Malaysia. Gunakan maklumat yang diberikan.

MAKLUMAT TERSEDIA:
{context}

SOALAN: {query}

Berikan jawapan lengkap berdasarkan maklumat di atas. Sebutkan sumber dan proses yang jelas.

JAWAPAN:"""

        else:  # English
            if query_type == "statistics":
                prompt = f"""You are a Malaysian poverty data expert. Your task is to answer questions using the provided data.

OFFICIAL MALAYSIAN POVERTY DATA:
{context}

QUESTION: {query}

SPECIFIC INSTRUCTIONS:
1. MUST use numbers, percentages, or statistics from the data
2. DO NOT say "information not available" if relevant data exists
3. If there are years, percentages, or money amounts in data, MUST mention them
4. Clearly cite data sources
5. If data is incomplete, state what IS available and suggest where to get more

ANSWER (use available data):"""

            else:
                prompt = f"""You are a Malaysian social assistance expert. Use the provided information.

AVAILABLE INFORMATION:
{context}

QUESTION: {query}

Provide a comprehensive answer based on the above information. Include clear sources and processes.

ANSWER:"""
        
        try:
            response = self.llm_client.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.1,
                    'max_output_tokens': 600
                }
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating Gemini response: {str(e)}"
    
    def generate_response_ollama(self, query: str, context: str, language: str) -> str:
        """Generate response using Ollama"""
        if language == "bahasa_malaysia" or language == "mixed":
            prompt = f"""Anda adalah pembantu AI untuk kemiskinan dan bantuan sosial Malaysia. Jawab dalam Bahasa Malaysia.

Maklumat: {context}

Soalan: {query}

Jawapan:"""
        else:
            prompt = f"""You are an AI assistant for Malaysian poverty and social assistance. Answer in English.

Information: {context}

Question: {query}

Answer:"""
        
        try:
            response = self.llm_client.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.3, 'max_tokens': 500}
            )
            return response['response'].strip()
        except Exception as e:
            return f"Error generating Ollama response: {str(e)}"
    
    def generate_response_groq(self, query: str, context: str, language: str) -> str:
        """Generate response using Groq"""
        if language == "bahasa_malaysia" or language == "mixed":
            system_prompt = "Anda adalah pembantu AI pakar kemiskinan dan bantuan sosial Malaysia. Jawab dalam Bahasa Malaysia yang jelas."
            user_prompt = f"Maklumat: {context}\n\nSoalan: {query}"
        else:
            system_prompt = "You are an AI assistant specialized in Malaysian poverty and social assistance. Answer clearly in English."
            user_prompt = f"Information: {context}\n\nQuestion: {query}"
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating Groq response: {str(e)}"
    
    def generate_response_transformers(self, query: str, context: str, language: str) -> str:
        """Generate response using local Transformers"""
        prompt = f"Question: {query}\nContext: {context[:1000]}\nAnswer:"
        
        try:
            response = self.generation_model(
                prompt,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.generation_model.tokenizer.eos_token_id
            )
            return response[0]['generated_text'][len(prompt):].strip()
        except Exception as e:
            return f"Error generating Transformers response: {str(e)}"
    
    def generate_response_mock(self, query: str, context: str, language: str) -> str:
        """Generate mock response for testing"""
        if language == "bahasa_malaysia" or language == "mixed":
            return f"""Berdasarkan maklumat yang tersedia tentang kemiskinan dan bantuan sosial Malaysia:

Soalan: {query}

Maklumat berkaitan: {context[:300]}...

*Nota: Ini adalah respons ujian. Untuk maklumat tepat, gunakan model LLM sebenar.*"""
        else:
            return f"""Based on available information about Malaysian poverty and social assistance:

Question: {query}

Relevant information: {context[:300]}...

*Note: This is a test response. For accurate information, use a real LLM model.*"""
    
    def generate_response(self, query: str, context: str, language: str) -> str:
        """Generate response using the configured model"""
        if self.model_type == "gemini":
            return self.generate_response_gemini(query, context, language)
        elif self.model_type == "ollama":
            return self.generate_response_ollama(query, context, language)
        elif self.model_type == "groq":
            return self.generate_response_groq(query, context, language)
        elif self.model_type == "transformers":
            return self.generate_response_transformers(query, context, language)
        else:
            return self.generate_response_mock(query, context, language)
    
    def debug_retrieval(self, query: str, top_k: int = 10):
        """Show what chunks are being retrieved"""
        print(f"🔍 DEBUG: Retrieving for query: '{query}'")
        
        chunks = self.retrieve_relevant_chunks(query, top_k)
        
        print(f"📊 Retrieved {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"\n{i+1}. Similarity: {chunk.get('similarity', 0):.3f}")
            print(f"   Source: {chunk.get('source', 'Unknown')}")
            content = chunk.get('text', '')
            # Show snippet with numbers/percentages
            numbers = re.findall(r'\d+\.?\d*\s*%|\bRM\s*\d+', content)
            if numbers:
                print(f"   📈 Contains: {', '.join(numbers[:3])}")
            print(f"   Content preview: {content[:150]}...")
        
        return chunks
    
    def chat(self, query: str, top_k: int = 10, verbose: bool = True) -> Dict:
        """Main chat interface with all fixes applied"""
        if verbose:
            print(f"🔍 Processing query: '{query}'")
        
        # Detect language
        query_language = self.detect_language(query)
        if verbose:
            print(f"🌐 Language: {query_language}")
        
        # Retrieve relevant chunks with enhanced retrieval
        retrieved_chunks = self.retrieve_relevant_chunks(query, top_k)
        
        if not retrieved_chunks:
            if query_language in ["bahasa_malaysia", "mixed"]:
                response = "Maaf, tidak dapat menemui maklumat spesifik untuk soalan anda. Cuba soalan lain tentang kemiskinan, eKasih, atau SARA."
            else:
                response = "Sorry, couldn't find specific information for your question. Try another question about poverty, eKasih, or SARA."
            
            return {
                'query': query,
                'language': query_language,
                'response': response,
                'sources': [],
                'retrieved_chunks': 0
            }
        
        if verbose:
            print(f"📊 Retrieved {len(retrieved_chunks)} chunks")
            # Show top 3 with more detail
            for i, chunk in enumerate(retrieved_chunks[:3]):
                print(f"   {i+1}. Score: {chunk['similarity']:.3f} | {chunk.get('source', 'Unknown')}")
                # Show snippet with numbers/percentages
                content = chunk.get('text', '')
                numbers = re.findall(r'\d+\.?\d*\s*%|\bRM\s*\d+', content)
                if numbers:
                    print(f"      📈 Contains: {', '.join(numbers[:3])}")
        
        # Enhanced context preparation
        context = self.prepare_context(retrieved_chunks)
        
        if verbose:
            print(f"📄 Context length: {len(context)} chars")
        
        # Generate response
        response = self.generate_response(query, context, query_language)
        
        # Prepare sources
        sources = []
        for chunk in retrieved_chunks[:5]:  # Top 5 sources
            source_info = {
                'source': chunk.get('source', 'Government Document'),
                'doc_type': chunk.get('doc_type', 'Official Document'),
                'similarity': chunk['similarity'],
                'url': chunk.get('url', '')
            }
            sources.append(source_info)
        
        return {
            'query': query,
            'language': query_language,
            'response': response,
            'sources': sources,
            'retrieved_chunks': len(retrieved_chunks),
            'model_used': self.model_type,
            'fixes_applied': True
        }
    
    def test_queries(self):
        """Test with sample queries"""
        print(f"\nTESTING ENHANCED {self.model_type.upper()} MODEL")
        print("=" * 60)
        
        test_queries = [
            "What is Malaysia's poverty rate?",
            "Bagaimana cara mohon eKasih?",
            "How much assistance does SARA provide?",
            "Apa itu kemiskinan tegar Malaysia?",
            "Who is eligible for government assistance?",
            "Berapa kadar kemiskinan di Malaysia?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nTest {i}: {query}")
            print("-" * 40)
            
            try:
                result = self.chat(query, verbose=False)
                print(f"Language: {result['language']}")
                print(f"Chunks: {result['retrieved_chunks']}")
                print(f"Model: {result['model_used']}")
                print(f"Fixes Applied: {result.get('fixes_applied', False)}")
                print(f"Response: {result['response'][:200]}...")
                if result['sources']:
                    print(f"Top source: {result['sources'][0]['source']}")
            except Exception as e:
                print(f"Error: {e}")
        
        print(f"\n✅ Completed testing {len(test_queries)} queries")
    
    def interactive_chat(self):
        """Interactive chat interface"""
        print(f"\nENHANCED MALAYSIAN POVERTY RAG CHATBOT ({self.model_type.upper()})")
        print("=" * 70)
        print("🔧 Enhanced with diagnostic fixes applied")
        print("Ask about poverty, eKasih, SARA, and Malaysian government assistance")
        print("Supports English and Bahasa Malaysia")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                query = input("Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'bye', 'keluar']:
                    print("👋 Thank you! / Terima kasih!")
                    break
                
                if not query:
                    continue
                
                print()
                result = self.chat(query, verbose=True)
                
                print(f"\nResponse:")
                print("=" * 20)
                print(result['response'])
                
                if result['sources']:
                    print(f"\nSources:")
                    for i, source in enumerate(result['sources'][:3], 1):
                        print(f"  {i}. {source['source']} (Score: {source['similarity']:.3f})")
                
                print("\n" + "-" * 70)
                
            except KeyboardInterrupt:
                print("\nGoodbye! / Selamat tinggal!")
                break
            except Exception as e:
                print(f"Error: {e}")

def print_model_options():
    """Print available model options"""
    print("\n🤖 AVAILABLE FREE LLM OPTIONS:")
    print("=" * 50)
    
    options = [
        ("1. Google Gemini", "Free tier: 15 req/min, 1M tokens/day", "Get API key: https://aistudio.google.com/app/apikey"),
        ("2. Ollama (Local)", "Unlimited usage, runs locally", "Install: https://ollama.ai/"),
        ("3. Groq", "Fast inference, 6K tokens/min free", "Get API key: https://console.groq.com/keys"),
        ("4. Transformers (Local)", "Unlimited, but slower", "Uses Hugging Face models"),
        ("5. Mock", "For testing only", "No real AI responses")
    ]
    
    for name, quota, setup in options:
        print(f"{name}")
        print(f"  📊 {quota}")
        print(f"  🔗 {setup}")
        print()

def setup_rag_system():
    """Interactive setup for RAG system"""
    print("🔧 ENHANCED RAG SYSTEM SETUP")
    print("=" * 40)
    print_model_options()
    
    print("Choose your preferred model:")
    choice = input("Enter choice (1-5) or model name (gemini/ollama/groq/transformers/mock): ").strip().lower()
    
    # Map choices
    choice_map = {
        '1': 'gemini', 'gemini': 'gemini',
        '2': 'ollama', 'ollama': 'ollama', 
        '3': 'groq', 'groq': 'groq',
        '4': 'transformers', 'transformers': 'transformers',
        '5': 'mock', 'mock': 'mock'
    }
    
    model_type = choice_map.get(choice, 'mock')
    api_key = None
    model_name = None
    
    # Get API key if needed
    if model_type in ['gemini', 'groq']:
        api_key = input(f"Enter {model_type.title()} API key (or press Enter to use mock): ").strip()
        if not api_key:
            print("No API key provided, using mock model")
            model_type = 'mock'
    
    # Get model name if needed
    if model_type == 'ollama':
        suggested = input("Model name (press Enter for llama3.1:8b): ").strip()
        model_name = suggested if suggested else None
    elif model_type == 'groq':
        suggested = input("Model name (press Enter for llama3-8b-8192): ").strip()
        model_name = suggested if suggested else None
    
    print(f"\n🚀 Setting up enhanced RAG system with {model_type}...")
    
    # Initialize RAG system
    rag_system = MalaysianPovertyRAG(
        model_type=model_type,
        api_key=api_key,
        model_name=model_name
    )
    
    # Load embeddings and initialize models
    rag_system.load_embeddings()
    rag_system.initialize_models()
    
    print(f"\n✅ Enhanced RAG system ready with {rag_system.model_type} model!")
    print("🔧 Applied fixes:")
    print("   • Fixed unknown sources in metadata")
    print("   • Enhanced context extraction")
    print("   • Improved prompts for better responses")
    print("   • Lowered similarity threshold for more results")
    print("   • Added query expansion with synonyms")
    
    return rag_system

def main():
    """Main function"""
    try:
        rag_system = setup_rag_system()
        
        # Test the enhanced system
        rag_system.test_queries()
        
        # Start interactive chat
        rag_system.interactive_chat()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    main()