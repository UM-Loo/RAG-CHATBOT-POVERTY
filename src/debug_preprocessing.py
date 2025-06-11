# PDF Processing Debug & Fix
# This will help identify and fix the PDF processing issue

from pathlib import Path
import json
from datetime import datetime
import traceback

def debug_pdf_processing():
    """
    Debug PDF processing step by step
    """
    print("DEBUGGING PDF PROCESSING")
    print("=" * 40)
    
    # Find the PDF file
    raw_data_path = Path("raw_documents")
    pdf_files = list(raw_data_path.rglob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found")
        return
    
    pdf_file = pdf_files[0]  # BNM_annual_report_2023_EN.pdf
    print(f"Processing: {pdf_file.name}")
    print(f"Size: {pdf_file.stat().st_size} bytes")
    
    # Step 1: Test PDF content extraction
    try:
        content, metadata = extract_pdf_content_debug(pdf_file)
        print(f"Content extracted: {len(content)} characters")
        print(f"Metadata: {metadata}")
    except Exception as e:
        print(f"Content extraction failed: {e}")
        traceback.print_exc()
        return
    
    # Step 2: Test text cleaning
    try:
        cleaned_content = clean_text_debug(content)
        print(f"Text cleaned: {len(cleaned_content)} characters")
    except Exception as e:
        print(f"Text cleaning failed: {e}")
        traceback.print_exc()
        return
    
    # Step 3: Test language detection
    try:
        language = detect_language_debug(cleaned_content)
        print(f"Language detected: {language}")
    except Exception as e:
        print(f"Language detection failed: {e}")
        traceback.print_exc()
        return
    
    # Step 4: Test chunking
    try:
        chunks = create_chunks_debug(cleaned_content)
        print(f"Chunks created: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
            print(f"  Chunk {i+1}: {len(chunk.split())} words - {chunk[:100]}...")
    except Exception as e:
        print(f"Chunking failed: {e}")
        traceback.print_exc()
        return
    
    # Step 5: Test complete document creation
    try:
        processed_doc = create_processed_doc_debug(pdf_file, metadata, cleaned_content, chunks, language)
        print(f"Processed document created successfully")
        print(f"  - Chunks: {processed_doc['chunk_count']}")
        print(f"  - Words: {processed_doc['word_count']}")
        print(f"  - Language: {processed_doc['metadata']['language']}")
    except Exception as e:
        print(f"Document creation failed: {e}")
        traceback.print_exc()
        return
    
    # Step 6: Test file saving
    try:
        output_path = Path("data/processed")
        output_path.mkdir(exist_ok=True)
        
        output_file = output_path / f"{pdf_file.stem}_processed.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_doc, f, ensure_ascii=False, indent=2)
        
        print(f"File saved successfully: {output_file}")
        print(f"  File size: {output_file.stat().st_size} bytes")
        
    except Exception as e:
        print(f"File saving failed: {e}")
        traceback.print_exc()
        return
    
    print("\n🎉 PDF processing completed successfully!")

def extract_pdf_content_debug(file_path):
    """
    Debug PDF content extraction
    """
    import fitz  # PyMuPDF
    
    metadata = {
        'doc_id': file_path.stem,
        'file_path': str(file_path),
        'file_type': 'pdf',
        'title': file_path.stem.replace('_', ' ').title(),
        'source': 'Bank Negara Malaysia',
        'language': 'English',  # Default for BNM report
        'doc_type': 'Annual Report'
    }
    
    content = ""
    
    # Extract text from PDF
    doc = fitz.open(file_path)
    print(f"  PDF has {len(doc)} pages")
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        content += page_text + "\n"
        
        if page_num == 0:  # Debug first page
            print(f"  Page 0 content: {len(page_text)} chars")
    
    doc.close()
    
    return content.strip(), metadata

def clean_text_debug(text):
    """
    Debug text cleaning
    """
    import re
    
    if not text or len(text.strip()) == 0:
        raise ValueError("Empty text content")
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Fix common encoding issues
    text = text.replace('â€™', "'")
    text = text.replace('â€œ', '"')
    text = text.replace('â€�', '"')
    
    # Normalize Malaysian currency
    text = re.sub(r'RM\s*(\d)', r'RM\1', text)
    
    return text.strip()

def detect_language_debug(text):
    """
    Debug language detection
    """
    malay_patterns = [
        r'\b(kerajaan|bantuan|kemiskinan|pendapatan|isi rumah|permohonan)\b',
        r'\b(yang|dan|untuk|dengan|dalam|kepada|pada)\b',
        r'\b(malaysia|negara|rakyat|masyarakat)\b'
    ]
    
    text_lower = text.lower()
    
    malay_score = 0
    for pattern in malay_patterns:
        import re
        malay_score += len(re.findall(pattern, text_lower))
    
    total_words = len(text.split())
    malay_ratio = malay_score / max(total_words, 1)
    
    if malay_ratio > 0.15:
        return 'bahasa_malaysia'
    elif malay_ratio > 0.05:
        return 'mixed'
    else:
        return 'english'

def create_chunks_debug(text, chunk_size=500, overlap=50):
    """
    Debug chunking
    """
    import re
    
    if not text or len(text.strip()) == 0:
        raise ValueError("Empty text for chunking")
    
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
            chunk_sentences = current_chunk.split('. ')
            overlap_sentences = []
            temp_size = 0
            
            for prev_sentence in reversed(chunk_sentences[-3:]):
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

def create_processed_doc_debug(file_path, metadata, content, chunks, language):
    """
    Debug processed document creation
    """
    metadata['language'] = language
    
    processed_doc = {
        'metadata': metadata,
        'full_content': content,
        'chunks': chunks,
        'chunk_count': len(chunks),
        'word_count': len(content.split()) if content else 0,
        'processing_date': datetime.now().isoformat()
    }
    
    return processed_doc

def fix_preprocessing_pipeline():
    """
    Apply fix to the main preprocessing pipeline
    """
    print("\nAPPLYING FIX TO PREPROCESSING PIPELINE")
    print("=" * 50)
    
    # Import the main preprocessor
    try:
        from preprocessing_pipeline import DocumentPreprocessor
        
        # Process only the PDF file to test
        preprocessor = DocumentPreprocessor("raw_documents", "data/processed")
        
        # Find PDF file
        pdf_files = list(Path("raw_documents").rglob("*.pdf"))
        if pdf_files:
            pdf_file = pdf_files[0]
            print(f"Processing PDF: {pdf_file.name}")
            
            try:
                processed_doc = preprocessor.process_document(pdf_file)
                
                # Save the processed document
                output_file = Path("data/processed") / f"{pdf_file.stem}_processed.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_doc, f, ensure_ascii=False, indent=2)
                
                print(f"PDF processed successfully!")
                print(f"  Chunks: {processed_doc['chunk_count']}")
                print(f"  Words: {processed_doc['word_count']}")
                print(f"  File saved: {output_file}")
                
                return True
                
            except Exception as e:
                print(f"Error in main pipeline: {e}")
                traceback.print_exc()
                return False
        else:
            print("No PDF files found")
            return False
            
    except Exception as e:
        print(f"Error importing pipeline: {e}")
        return False

def main():
    """
    Main debug and fix function
    """
    print("PDF PROCESSING DEBUG & FIX")
    print("=" * 50)
    
    # Step 1: Debug step by step
    debug_pdf_processing()
    
    # Step 2: Try with main pipeline
    print("\n" + "=" * 50)
    fix_preprocessing_pipeline()
    
    # Step 3: Verify results
    print("\nVERIFICATION")
    print("=" * 20)
    
    processed_path = Path("data/processed")
    bnm_file = processed_path / "BNM_annual_report_2023_EN_processed.json"
    
    if bnm_file.exists():
        print("BNM_annual_report_2023_EN_processed.json created successfully!")
        
        with open(bnm_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  Chunks: {data['chunk_count']}")
        print(f"  Words: {data['word_count']}")
        print(f"  Language: {data['metadata']['language']}")
    else:
        print("BNM_annual_report_2023_EN_processed.json still missing")
    
    # Check master index
    master_file = processed_path / "master_index.json"
    if master_file.exists():
        with open(master_file, 'r', encoding='utf-8') as f:
            master_data = json.load(f)
        print(f"  Master index documents: {master_data.get('total_documents', 0)}")

if __name__ == "__main__":
    main()