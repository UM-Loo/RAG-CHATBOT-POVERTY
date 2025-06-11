import os
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import json
from datetime import datetime

# PDF processing imports
try:
    import PyPDF2
    import fitz # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    print("PDF libraries not installed. Run: pip install PyPDF2 PyMuPDF")
    PDF_SUPPORT = False

class DocumentPreprocessor:
    """
    Preprocesses collected documents for RAG pipeline
    Handles multilingual content (English + Bahasa Malaysia)
    - Extracts structured metadata from your document headers
    - Maps both English and Bahasa Malaysia field names
    - Creates unique document IDs
    - Tracks source information and dates
    """
    
    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Language detection patterns
        self.malay_patterns = [
            r'\b(kerajaan|bantuan|kemiskinan|pendapatan|isi rumah|permohonan)\b',
            r'\b(yang|dan|untuk|dengan|dalam|kepada|pada)\b',
            r'\b(malaysia|negara|rakyat|masyarakat)\b'
        ]
    
    def detect_language(self, text: str) -> str:
        """
        Detect document language based on content
        """
        text_lower = text.lower()
        
        malay_score = 0
        for pattern in self.malay_patterns:
            malay_score += len(re.findall(pattern, text_lower))
        
        total_words = len(text.split())
        malay_ratio = malay_score / max(total_words, 1)
        
        if malay_ratio > 0.15:
            return 'bahasa_malaysia'
        elif malay_ratio > 0.05:
            return 'mixed'
        else:
            return 'english'
    
    def extract_metadata_from_txt(self, file_path: Path) -> Dict:
        """
        FIXED: Extract metadata from TXT document header with better parsing
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {}
        metadata['doc_id'] = file_path.stem
        metadata['file_path'] = str(file_path)
        metadata['file_type'] = 'txt'
        
        # Look for metadata sections with different patterns
        metadata_patterns = [
            'METADATA DOKUMEN',
            'DOCUMENT METADATA', 
            'TAJUK:',
            'TITLE:'
        ]
        
        for pattern in metadata_patterns:
            if pattern in content:
                # Find the metadata section
                if '==================================================' in content:
                    parts = content.split('==================================================')
                    for i, part in enumerate(parts):
                        if pattern in part:
                            metadata_section = part
                            break
                    else:
                        continue
                else:
                    # Fallback: find the section after the pattern
                    start_idx = content.find(pattern)
                    if start_idx != -1:
                        # Find next major section or end
                        end_patterns = ['KANDUNGAN DOKUMEN', 'DOCUMENT CONTENT', '=====']
                        end_idx = len(content)
                        for end_pattern in end_patterns:
                            pattern_idx = content.find(end_pattern, start_idx + 100)
                            if pattern_idx != -1:
                                end_idx = min(end_idx, pattern_idx)
                        metadata_section = content[start_idx:end_idx]
                
                # Parse metadata fields
                for line in metadata_section.split('\n'):
                    line = line.strip()
                    if ':' in line and not line.startswith('='):
                        key, value = line.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        
                        # Map metadata keys
                        key_mapping = {
                            'tajuk': 'title',
                            'title': 'title',
                            'sumber': 'source', 
                            'source': 'source',
                            'url': 'url',
                            'bahasa': 'language',
                            'language': 'language',
                            'jenis_dokumen': 'doc_type',
                            'document_type': 'doc_type',
                            'tarikh_terbit': 'publish_date',
                            'publish_date': 'publish_date',
                            'tag_topik': 'topic_tags',
                            'topic_tags': 'topic_tags',
                            'relevan': 'relevance',
                            'relevance': 'relevance'
                        }
                        
                        if key in key_mapping:
                            if key in ['tag_topik', 'topic_tags']:
                                metadata[key_mapping[key]] = [tag.strip() for tag in value.split(',')]
                            else:
                                metadata[key_mapping[key]] = value
                break
        
        return metadata
    
    def extract_content_from_txt(self, file_path: Path) -> str:
        """
        Extract main document content with correct priority based on diagnostics
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"File size: {len(content)} characters")
        
        # DOCUMENT CONTENT (works for your EN files)
        if 'DOCUMENT CONTENT' in content:
            parts = content.split('DOCUMENT CONTENT')
            if len(parts) > 1:
                content_part = parts[1]
                # Remove end markers
                for end_marker in ['END OF DOCUMENT', '====']:
                    if end_marker in content_part:
                        content_part = content_part.split(end_marker)[0]
                
                extracted_content = content_part.strip()
                if len(extracted_content) > 500:  # Good content length
                    print(f"Extracted {len(extracted_content)} characters using DOCUMENT CONTENT")
                    return extracted_content
        
        # KANDUNGAN DOKUMEN (works for your BM files)
        if 'KANDUNGAN DOKUMEN' in content:
            parts = content.split('KANDUNGAN DOKUMEN')
            if len(parts) > 1:
                content_part = parts[1]
                # Remove end markers
                for end_marker in ['TAMAT DOKUMEN', '====']:
                    if end_marker in content_part:
                        content_part = content_part.split(end_marker)[0]
                
                extracted_content = content_part.strip()
                if len(extracted_content) > 500:  # Good content length
                    print(f"Extracted {len(extracted_content)} characters using KANDUNGAN DOKUMEN")
                    return extracted_content
        
        # Content detection method (fallback)
        lines = content.split('\n')
        content_start = 0
        
        # Skip initial metadata/header lines
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['kuala lumpur', 'putrajaya', 'malaysia', 'government']):
                content_start = i
                break
        
        if content_start > 0:
            extracted_content = '\n'.join(lines[content_start:]).strip()
            if len(extracted_content) > 100:
                print(f"Extracted {len(extracted_content)} characters using content detection")
                return extracted_content
        
        if '==================================================' in content:
            parts = content.split('==================================================')
            if len(parts) >= 3:
                content_part = parts[2].strip()
                if len(content_part) > 100:
                    print(f"Using fallback section method: {len(content_part)} characters")
                    return content_part
        
        # Absolute fallback
        print(f" Using full content as fallback")
        return content.strip()
    
    def extract_content_from_pdf(self, file_path: Path) -> Tuple[str, Dict]:
        """
        Extract content and metadata from PDF files
        """
        if not PDF_SUPPORT:
            return "", {'error': 'PDF support not available'}
        
        metadata = {
            'doc_id': file_path.stem,
            'file_path': str(file_path),
            'file_type': 'pdf',
            'title': file_path.stem.replace('_', ' ').title(),
            'source': 'PDF Document'
        }
        
        content = ""
        
        try:
            # Try PyMuPDF first (better text extraction)
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                content += page_text + "\n"
            
            doc.close()
            print(f"PDF extracted: {len(content)} characters using PyMuPDF")
            
        except Exception as e:
            print(f" PyMuPDF failed, trying PyPDF2: {e}")
            
            try:
                # Fallback to PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        content += page.extract_text() + "\n"
                
                print(f"PDF extracted: {len(content)} characters using PyPDF2")
                
            except Exception as e2:
                print(f"PDF extraction failed: {e2}")
                return "", metadata
        
        return content.strip(), metadata
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content
        - Removes formatting artifacts and excessive whitespace
        - Fixes common encoding issues
        - Normalizes Malaysian currency format (RM)
        - Standardizes bullet points and lists
        """
        if not text or len(text.strip()) == 0:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Fix common OCR/encoding issues
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€�', '"')
        
        # Normalize Malaysian currency
        text = re.sub(r'RM\s*(\d)', r'RM\1', text)
        
        # Clean up bullet points
        text = re.sub(r'^[-•]\s*', '• ', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def create_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into chunks for RAG processing
        Uses sentence-aware chunking with overlap
        - Creates 500-word chunks with 50-word overlap
        - Preserves sentence boundaries (doesn't cut mid-sentence)
        - Maintains context with overlapping content
        - Optimized for RAG retrieval performance
        """
        # Split into sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_size + sentence_words > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_sentences = []
                temp_size = 0
                
                # Get last few sentences for overlap
                for prev_sentence in reversed(current_chunk.split('. ')):
                    if temp_size + len(prev_sentence.split()) <= overlap:
                        overlap_sentences.insert(0, prev_sentence)
                        temp_size += len(prev_sentence.split())
                    else:
                        break
                
                current_chunk = '. '.join(overlap_sentences)
                if current_chunk and not current_chunk.endswith('.'):
                    current_chunk += '. '
                current_size = temp_size
            
            current_chunk += sentence + " "
            current_size += sentence_words
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_document(self, file_path: Path) -> Dict:
        """
        Process a single document through the full pipeline
        """
        print(f"Processing: {file_path.name}")
        
        # Extract metadata and content
        if file_path.suffix.lower() == '.pdf':
            # Handle PDF files
            content, metadata = self.extract_content_from_pdf(file_path)
        else:
            # Handle TXT files
            metadata = self.extract_metadata_from_txt(file_path)
            content = self.extract_content_from_txt(file_path)
        
        # Clean content
        cleaned_content = self.clean_text(content)
        
        # Detect language if not specified
        if 'language' not in metadata or not metadata['language']:
            detected_lang = self.detect_language(cleaned_content)
            metadata['language'] = detected_lang
            print(f"Detected language: {detected_lang}")
        
        # Create chunks
        chunks = self.create_chunks(cleaned_content)
        
        # Prepare processed document
        processed_doc = {
            'metadata': metadata,
            'full_content': cleaned_content,
            'chunks': chunks,
            'chunk_count': len(chunks),
            'word_count': len(cleaned_content.split()),
            'processing_date': datetime.now().isoformat()
        }
        
        print(f"{len(chunks)} chunks created, {len(cleaned_content.split())} words")
        
        return processed_doc
    
    def process_all_documents(self) -> List[Dict]:
        """
        Process all documents in the raw data directory
        """
        print("Starting Document Preprocessing Pipeline")
        print("=" * 50)
        
        all_processed = []
        
        # Find all text and PDF files
        text_files = list(self.raw_data_path.rglob("*.txt"))
        pdf_files = list(self.raw_data_path.rglob("*.pdf"))
        all_files = text_files + pdf_files
        
        print(f"Found {len(text_files)} TXT files and {len(pdf_files)} PDF files")
        print(f"Total: {len(all_files)} documents to process")
        print()
        
        for file_path in text_files:
            try:
                processed_doc = self.process_document(file_path)
                all_processed.append(processed_doc)
                
                # Save individual processed document
                output_file = self.processed_data_path / f"{file_path.stem}_processed.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_doc, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                import traceback
                traceback.print_exc()
        
        print()
        print("Processing Summary:")
        print("=" * 30)
        
        # Create summary statistics
        total_chunks = sum(doc['chunk_count'] for doc in all_processed)
        total_words = sum(doc['word_count'] for doc in all_processed)
        
        # Language distribution
        lang_dist = {}
        for doc in all_processed:
            lang = doc['metadata'].get('language', 'unknown')
            lang_dist[lang] = lang_dist.get(lang, 0) + 1
        
        print(f"Documents processed: {len(all_processed)}")
        print(f"Total chunks created: {total_chunks}")
        print(f"Total words: {total_words:,}")
        print(f"Language distribution: {lang_dist}")
        
        # Save master index
        master_index = {
            'processing_date': datetime.now().isoformat(),
            'total_documents': len(all_processed),
            'total_chunks': total_chunks,
            'total_words': total_words,
            'language_distribution': lang_dist,
            'documents': [doc['metadata'] for doc in all_processed]
        }
        
        with open(self.processed_data_path / 'master_index.json', 'w', encoding='utf-8') as f:
            json.dump(master_index, f, ensure_ascii=False, indent=2)
        
        print(f"All processed documents saved to: {self.processed_data_path}")
        
        return all_processed

# Usage Example
def main():
    """
    Main preprocessing pipeline execution
    """
    # Install PDF dependencies if needed
    try:
        import PyPDF2
        import fitz
    except ImportError:
        print("Installing PDF dependencies...")
        os.system("pip install PyPDF2 PyMuPDF")
    
    # Set up paths
    raw_data_path = "data/raw_documents"
    processed_data_path = "data/processed"
    
    # Initialize preprocessor
    preprocessor = DocumentPreprocessor(raw_data_path, processed_data_path)
    
    # Process all documents
    processed_documents = preprocessor.process_all_documents()
    
    print("\n🎉 FIXED preprocessing pipeline completed!")
    print(f"Ready for RAG implementation with {len(processed_documents)} documents")

if __name__ == "__main__":
    main()