import numpy as np
import json
from pathlib import Path
from typing import List, Dict
import re
from collections import Counter

class RAGDiagnosticTool:
    """
    Diagnostic tool to analyze and improve RAG system performance
    """
    
    def __init__(self, embeddings_path: str = "data/embeddings"):
        self.embeddings_path = Path(embeddings_path)
        self.chunks_metadata = []
        self.embeddings = None
        
    def load_data(self):
        """Load metadata and embeddings for analysis"""
        print("Loading RAG data for analysis...")
        
        # Find latest metadata file
        config_files = list(self.embeddings_path.glob("embedding_config_*.json"))
        if not config_files:
            raise FileNotFoundError("No embedding config files found")
        
        latest_config = max(config_files, key=lambda x: x.stat().st_mtime)
        timestamp_match = re.search(r'embedding_config_(\d{8}_\d{6})\.json', latest_config.name)
        
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            
            # Load metadata
            metadata_file = self.embeddings_path / f"chunks_metadata_{timestamp}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.chunks_metadata = json.load(f)
                print(f"✅ Loaded {len(self.chunks_metadata)} chunks")
            
            # Load embeddings
            embeddings_file = self.embeddings_path / f"embeddings_{timestamp}.npy"
            if embeddings_file.exists():
                self.embeddings = np.load(embeddings_file)
                print(f"✅ Loaded embeddings: {self.embeddings.shape}")
    
    def analyze_chunk_content(self):
        """Analyze what's actually in the chunks"""
        print("\n📊 CHUNK CONTENT ANALYSIS")
        print("=" * 50)
        
        # Source distribution
        sources = [chunk.get('source', 'Unknown') for chunk in self.chunks_metadata]
        source_counts = Counter(sources)
        
        print("📁 Sources distribution:")
        for source, count in source_counts.most_common():
            print(f"  • {source}: {count} chunks")
        
        # Language distribution
        languages = [chunk.get('language', 'unknown') for chunk in self.chunks_metadata]
        lang_counts = Counter(languages)
        
        print("\n🌐 Language distribution:")
        for lang, count in lang_counts.most_common():
            print(f"  • {lang}: {count} chunks")
        
        # Document types
        doc_types = [chunk.get('doc_type', 'Unknown') for chunk in self.chunks_metadata]
        type_counts = Counter(doc_types)
        
        print("\n📋 Document types:")
        for doc_type, count in type_counts.most_common():
            print(f"  • {doc_type}: {count} chunks")
        
        # Content length analysis
        lengths = [len(chunk.get('text', '')) for chunk in self.chunks_metadata]
        print(f"\n📏 Content length stats:")
        print(f"  • Average: {np.mean(lengths):.0f} characters")
        print(f"  • Min: {min(lengths)} characters")
        print(f"  • Max: {max(lengths)} characters")
        
        return source_counts, lang_counts, type_counts
    
    def search_for_keywords(self, keywords: List[str]):
        """Search for specific keywords in chunks"""
        print(f"\n🔍 KEYWORD SEARCH")
        print("=" * 50)
        
        for keyword in keywords:
            print(f"\nSearching for: '{keyword}'")
            found_chunks = []
            
            for i, chunk in enumerate(self.chunks_metadata):
                text = chunk.get('text', '').lower()
                if keyword.lower() in text:
                    found_chunks.append({
                        'index': i,
                        'source': chunk.get('source', 'Unknown'),
                        'snippet': self._get_keyword_snippet(text, keyword.lower())
                    })
            
            if found_chunks:
                print(f"  ✅ Found in {len(found_chunks)} chunks:")
                for match in found_chunks[:3]:  # Show top 3
                    print(f"    • Index {match['index']} ({match['source']})")
                    print(f"      \"{match['snippet']}\"")
            else:
                print(f"  ❌ Not found in any chunks")
    
    def _get_keyword_snippet(self, text: str, keyword: str, context_length: int = 100) -> str:
        """Get a snippet around the keyword"""
        pos = text.find(keyword)
        if pos == -1:
            return ""
        
        start = max(0, pos - context_length)
        end = min(len(text), pos + len(keyword) + context_length)
        snippet = text[start:end]
        
        # Clean up
        snippet = snippet.replace('\n', ' ').strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
            
        return snippet
    
    def analyze_poverty_specific_content(self):
        """Look for poverty-specific content"""
        print(f"\n🏚️ POVERTY DATA ANALYSIS")
        print("=" * 50)
        
        # Key poverty terms to search for
        poverty_terms = [
            "poverty rate", "kadar kemiskinan", 
            "ekasih", "e-kasih",
            "sara", "bantuan sara",
            "poverty line", "garis kemiskinan",
            "hardcore poverty", "kemiskinan tegar",
            "B40", "income",
            "rm", "ringgit", "assistance", "bantuan"
        ]
        
        self.search_for_keywords(poverty_terms)
    
    def find_statistical_data(self):
        """Look for numbers, percentages, and statistical data"""
        print(f"\n📈 STATISTICAL DATA ANALYSIS")
        print("=" * 50)
        
        stats_found = []
        
        for i, chunk in enumerate(self.chunks_metadata):
            text = chunk.get('text', '')
            
            # Look for percentages
            percentages = re.findall(r'\d+\.?\d*\s*%', text)
            
            # Look for monetary amounts
            money_amounts = re.findall(r'RM\s*\d+(?:,\d{3})*(?:\.\d{2})?', text, re.IGNORECASE)
            
            # Look for years
            years = re.findall(r'\b20\d{2}\b', text)
            
            if percentages or money_amounts or years:
                stats_found.append({
                    'index': i,
                    'source': chunk.get('source', 'Unknown'),
                    'percentages': percentages,
                    'money': money_amounts,
                    'years': years,
                    'snippet': text[:200] + "..." if len(text) > 200 else text
                })
        
        print(f"Found statistical data in {len(stats_found)} chunks:")
        for stat in stats_found[:5]:  # Show top 5
            print(f"\n  📊 Index {stat['index']} ({stat['source']})")
            if stat['percentages']:
                print(f"    📈 Percentages: {', '.join(stat['percentages'])}")
            if stat['money']:
                print(f"    💰 Money: {', '.join(stat['money'])}")
            if stat['years']:
                print(f"    📅 Years: {', '.join(stat['years'])}")
            print(f"    📝 \"{stat['snippet']}\"")
    
    def test_specific_queries(self):
        """Test how well chunks match specific queries"""
        print(f"\n🎯 QUERY MATCHING TEST")
        print("=" * 50)
        
        test_queries = [
            "Malaysia poverty rate percentage",
            "eKasih application process",
            "SARA assistance amount money",
            "hardcore poverty definition Malaysia"
        ]
        
        if self.embeddings is None:
            print("❌ Embeddings not loaded, skipping similarity test")
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            for query in test_queries:
                print(f"\n🔍 Query: '{query}'")
                
                # Generate query embedding
                query_embedding = model.encode([query])
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
                
                # Calculate similarities
                normalized_embeddings = self.embeddings / np.linalg.norm(
                    self.embeddings, axis=1, keepdims=True
                )
                similarities = np.dot(normalized_embeddings, query_embedding.T).flatten()
                top_indices = np.argsort(similarities)[::-1][:3]
                
                print("  📊 Top matches:")
                for i, idx in enumerate(top_indices):
                    chunk = self.chunks_metadata[idx]
                    similarity = similarities[idx]
                    print(f"    {i+1}. Score: {similarity:.3f} | {chunk.get('source', 'Unknown')}")
                    print(f"       \"{chunk.get('text', '')[:100]}...\"")
                    
        except ImportError:
            print("❌ sentence-transformers not available for similarity testing")
    
    def generate_improvement_recommendations(self):
        """Generate specific recommendations for improvement"""
        print(f"\n💡 IMPROVEMENT RECOMMENDATIONS")
        print("=" * 50)
        
        recommendations = []
        
        # Check for source issues
        unknown_sources = sum(1 for chunk in self.chunks_metadata if chunk.get('source', 'Unknown') == 'Unknown')
        if unknown_sources > 0:
            recommendations.append(f"🔧 Fix {unknown_sources} chunks with 'Unknown' sources in metadata")
        
        # Check for content issues
        empty_chunks = sum(1 for chunk in self.chunks_metadata if len(chunk.get('text', '')) < 50)
        if empty_chunks > 0:
            recommendations.append(f"🔧 Remove or merge {empty_chunks} chunks with very little content")
        
        # Check language detection
        mixed_lang = sum(1 for chunk in self.chunks_metadata if chunk.get('language') not in ['english', 'bahasa_malaysia'])
        if mixed_lang > 0:
            recommendations.append(f"🔧 Improve language detection for {mixed_lang} chunks")
        
        # Content recommendations
        recommendations.extend([
            "🔧 Lower similarity threshold from 0.2 to 0.1 for more results",
            "🔧 Increase context length to include more relevant information",
            "🔧 Add more specific prompts for different types of questions",
            "🔧 Implement query preprocessing to handle variations",
            "🔧 Add fallback responses when specific data isn't found"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        return recommendations
    
    def run_full_diagnostic(self):
        """Run complete diagnostic analysis"""
        print("🩺 RAG SYSTEM DIAGNOSTIC")
        print("=" * 60)
        
        try:
            self.load_data()
            self.analyze_chunk_content()
            self.analyze_poverty_specific_content()
            self.find_statistical_data()
            self.test_specific_queries()
            self.generate_improvement_recommendations()
            
            print(f"\n✅ Diagnostic complete!")
            
        except Exception as e:
            print(f"❌ Diagnostic failed: {e}")

def main():
    """Run the diagnostic tool"""
    diagnostic = RAGDiagnosticTool()
    diagnostic.run_full_diagnostic()

if __name__ == "__main__":
    main()