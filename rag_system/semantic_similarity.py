"""
Enhanced Semantic Similarity Module for RAG Evaluation Framework.

This module provides sophisticated semantic similarity calculations moving beyond
simple token overlap to include:
- Sentence transformer embeddings
- Cosine similarity with TF-IDF
- Semantic concept matching
- Hybrid weighted approaches
- Caching for performance optimization
"""

import re
import logging
import hashlib
from functools import lru_cache
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

from .advanced_config import SemanticSimilarityConfig, SemanticSimilarityMethod
from .models import SourceSnippet


logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Result of semantic similarity calculation."""
    score: float
    method: str
    details: Dict[str, float]
    confidence: float
    explanation: str


class SemanticSimilarityCalculator:
    """Advanced semantic similarity calculator with multiple methods."""
    
    def __init__(self, method_or_config):
        """
        Initialize calculator with method or config.
        
        Args:
            method_or_config: Either SemanticSimilarityMethod enum or SemanticSimilarityConfig object
        """
        if isinstance(method_or_config, SemanticSimilarityMethod):
            # Create default config from method
            self.config = SemanticSimilarityConfig(method=method_or_config)
        else:
            # Use provided config
            self.config = method_or_config
            
        self.method = self.config.method
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        self._sentence_model = None
        self._embedding_cache = {}
        
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.config.method in [
            SemanticSimilarityMethod.SENTENCE_TRANSFORMERS,
            SemanticSimilarityMethod.HYBRID_WEIGHTED
        ]:
            try:
                self._sentence_model = SentenceTransformer(self.config.model_name)
                logger.info(f"Loaded sentence transformer model: {self.config.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self._sentence_model = None
    
    def calculate_grounding_score(self, answer: str, snippets: List[SourceSnippet]) -> SimilarityResult:
        """
        Calculate comprehensive grounding score using configured method.
        
        Args:
            answer: Generated answer text
            snippets: List of source snippets
            
        Returns:
            SimilarityResult with detailed scoring information
        """
        if not snippets or not answer:
            return SimilarityResult(
                score=0.0,
                method=self.config.method.value,
                details={"reason": "no_snippets_or_answer"},
                confidence=1.0,
                explanation="No snippets or answer provided"
            )
        
        # Dispatch to appropriate method
        if self.config.method == SemanticSimilarityMethod.TOKEN_OVERLAP:
            return self._calculate_token_overlap(answer, snippets)
        elif self.config.method == SemanticSimilarityMethod.COSINE_SIMILARITY:
            return self._calculate_cosine_similarity(answer, snippets)
        elif self.config.method == SemanticSimilarityMethod.SENTENCE_TRANSFORMERS:
            return self._calculate_sentence_similarity(answer, snippets)
        elif self.config.method == SemanticSimilarityMethod.HYBRID_WEIGHTED:
            return self._calculate_hybrid_similarity(answer, snippets)
        else:
            raise ValueError(f"Unknown similarity method: {self.config.method}")
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using configured method.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0
        
        # Create dummy snippet for compatibility with grounding_score method
        dummy_snippet = SourceSnippet(content=text2, relevance_score=1.0)
        result = self.calculate_grounding_score(text1, [dummy_snippet])
        return result.score
    
    def _calculate_token_overlap(self, answer: str, snippets: List[SourceSnippet]) -> SimilarityResult:
        """Calculate enhanced token overlap with academic weighting."""
        answer_tokens = self._tokenize_academic(answer)
        total_overlap = 0
        total_snippet_tokens = 0
        snippet_scores = []
        
        for snippet in snippets:
            snippet_tokens = self._tokenize_academic(snippet.content)
            overlap = len(answer_tokens.intersection(snippet_tokens))
            total_overlap += overlap
            total_snippet_tokens += len(snippet_tokens)
            
            # Per-snippet score
            snippet_score = overlap / max(len(snippet_tokens), 1) if snippet_tokens else 0
            snippet_scores.append(snippet_score)
        
        # Calculate multiple overlap metrics
        precision = total_overlap / max(len(answer_tokens), 1)
        recall = total_overlap / max(total_snippet_tokens, 1)
        f1_score = 2 * (precision * recall) / max(precision + recall, 1e-6)
        
        # Final score using F1 with snippet quality weighting
        max_snippet_score = max(snippet_scores) if snippet_scores else 0
        average_snippet_score = np.mean(snippet_scores) if snippet_scores else 0
        
        final_score = (f1_score * 0.6) + (max_snippet_score * 0.25) + (average_snippet_score * 0.15)
        
        return SimilarityResult(
            score=min(1.0, final_score),
            method="enhanced_token_overlap",
            details={
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "max_snippet_score": max_snippet_score,
                "avg_snippet_score": average_snippet_score,
                "total_overlap": total_overlap,
                "answer_tokens": len(answer_tokens),
                "snippet_tokens": total_snippet_tokens
            },
            confidence=0.7,  # Medium confidence for token-based methods
            explanation=f"Enhanced token overlap: F1={f1_score:.3f}, best snippet match={max_snippet_score:.3f}"
        )
    
    def _calculate_cosine_similarity(self, answer: str, snippets: List[SourceSnippet]) -> SimilarityResult:
        """Calculate TF-IDF cosine similarity."""
        try:
            # Prepare documents
            documents = [answer] + [snippet.content for snippet in snippets]
            
            # Fit TF-IDF vectorizer
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            
            # Calculate cosine similarities
            answer_vector = tfidf_matrix[0:1]  # First document is the answer
            snippet_vectors = tfidf_matrix[1:]  # Rest are snippets
            
            similarities = cosine_similarity(answer_vector, snippet_vectors)[0]
            
            # Calculate aggregate scores
            max_similarity = np.max(similarities)
            avg_similarity = np.mean(similarities)
            weighted_similarity = np.average(similarities, weights=np.exp(similarities))  # Weight by similarity
            
            # Final score combining different similarity metrics
            final_score = (max_similarity * 0.4) + (avg_similarity * 0.3) + (weighted_similarity * 0.3)
            
            return SimilarityResult(
                score=min(1.0, final_score),
                method="tfidf_cosine_similarity",
                details={
                    "max_similarity": float(max_similarity),
                    "avg_similarity": float(avg_similarity),
                    "weighted_similarity": float(weighted_similarity),
                    "individual_similarities": similarities.tolist(),
                    "vocabulary_size": len(self.tfidf_vectorizer.vocabulary_)
                },
                confidence=0.8,  # Higher confidence for TF-IDF
                explanation=f"TF-IDF cosine similarity: max={max_similarity:.3f}, avg={avg_similarity:.3f}"
            )
            
        except Exception as e:
            logger.warning(f"TF-IDF calculation failed: {e}")
            # Fallback to token overlap
            return self._calculate_token_overlap(answer, snippets)
    
    def _calculate_sentence_similarity(self, answer: str, snippets: List[SourceSnippet]) -> SimilarityResult:
        """Calculate sentence transformer-based semantic similarity."""
        if not self._sentence_model:
            logger.warning("Sentence transformer not available, falling back to cosine similarity")
            return self._calculate_cosine_similarity(answer, snippets)
        
        try:
            # Get embeddings with caching
            answer_embedding = self._get_cached_embedding(answer)
            snippet_embeddings = [self._get_cached_embedding(snippet.content) for snippet in snippets]
            
            # Calculate semantic similarities
            similarities = []
            for snippet_embedding in snippet_embeddings:
                similarity = np.dot(answer_embedding, snippet_embedding) / (
                    np.linalg.norm(answer_embedding) * np.linalg.norm(snippet_embedding)
                )
                similarities.append(similarity)
            
            similarities = np.array(similarities)
            
            # Aggregate semantic scores
            max_semantic = np.max(similarities)
            avg_semantic = np.mean(similarities)
            
            # Apply threshold to reduce noise
            above_threshold = similarities[similarities > self.config.similarity_threshold]
            relevant_semantic = np.mean(above_threshold) if len(above_threshold) > 0 else 0
            
            # Final score emphasizing relevant matches
            final_score = (max_semantic * 0.4) + (relevant_semantic * 0.4) + (avg_semantic * 0.2)
            
            # Confidence based on consistency of similarities
            confidence = 1.0 - np.std(similarities) if len(similarities) > 1 else 0.9
            
            return SimilarityResult(
                score=min(1.0, final_score),
                method="sentence_transformers",
                details={
                    "max_semantic": float(max_semantic),
                    "avg_semantic": float(avg_semantic),
                    "relevant_semantic": float(relevant_semantic),
                    "above_threshold_count": int(len(above_threshold)),
                    "individual_similarities": similarities.tolist(),
                    "similarity_std": float(np.std(similarities))
                },
                confidence=float(confidence),
                explanation=f"Semantic similarity: max={max_semantic:.3f}, relevant={relevant_semantic:.3f}"
            )
            
        except Exception as e:
            logger.warning(f"Sentence transformer calculation failed: {e}")
            return self._calculate_cosine_similarity(answer, snippets)
    
    def _calculate_hybrid_similarity(self, answer: str, snippets: List[SourceSnippet]) -> SimilarityResult:
        """Calculate hybrid similarity combining multiple methods."""
        # Get results from different methods
        token_result = self._calculate_token_overlap(answer, snippets)
        
        if self._sentence_model:
            semantic_result = self._calculate_sentence_similarity(answer, snippets)
        else:
            semantic_result = self._calculate_cosine_similarity(answer, snippets)
        
        # Combine scores with configured weights
        token_weight = self.config.weight_token_overlap
        semantic_weight = self.config.weight_semantic_similarity
        
        combined_score = (token_result.score * token_weight) + (semantic_result.score * semantic_weight)
        
        # Combine confidence scores
        combined_confidence = (token_result.confidence * token_weight) + (semantic_result.confidence * semantic_weight)
        
        return SimilarityResult(
            score=min(1.0, combined_score),
            method="hybrid_weighted",
            details={
                "token_score": token_result.score,
                "semantic_score": semantic_result.score,
                "token_weight": token_weight,
                "semantic_weight": semantic_weight,
                "token_details": token_result.details,
                "semantic_details": semantic_result.details
            },
            confidence=combined_confidence,
            explanation=f"Hybrid: token={token_result.score:.3f}×{token_weight}, semantic={semantic_result.score:.3f}×{semantic_weight}"
        )
    
    def _get_cached_embedding(self, text: str) -> np.ndarray:
        """Get sentence embedding with caching."""
        if not self.config.enable_caching:
            return self._sentence_model.encode(text)
        
        # Create cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_key = f"{self.config.model_name}:{text_hash}"
        
        if cache_key not in self._embedding_cache:
            if len(self._embedding_cache) >= self.config.cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]
            
            self._embedding_cache[cache_key] = self._sentence_model.encode(text)
        
        return self._embedding_cache[cache_key]
    
    def _tokenize_academic(self, text: str) -> Set[str]:
        """Enhanced academic tokenization with concept extraction."""
        # Basic cleaning
        text = re.sub(r'[^\w\s-]', ' ', text.lower())
        
        # Extract tokens
        tokens = set()
        
        # Word tokens (filter short words)
        words = [word.strip() for word in text.split() if len(word.strip()) > 2]
        tokens.update(words)
        
        # Bigrams for concept matching
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            tokens.add(bigram)
        
        # Academic concept patterns (domain-specific enhancement possible)
        concept_patterns = [
            r'\b\d+\s*(?:spaces?|minutes?|percent|%|degrees?)\b',
            r'\b(?:app|application|system|platform|service)\b',
            r'\b(?:parking|garage|building|floor|level)\b'
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, text)
            tokens.update(f"concept_{match.replace(' ', '_')}" for match in matches)
        
        return tokens
    
    def calculate_citation_alignment(self, answer: str, snippet: SourceSnippet) -> float:
        """Calculate how well a specific snippet aligns with answer content."""
        result = self.calculate_grounding_score(answer, [snippet])
        return result.score
    
    def get_similarity_explanation(self, answer: str, snippets: List[SourceSnippet]) -> str:
        """Get detailed explanation of similarity calculation."""
        result = self.calculate_grounding_score(answer, snippets)
        
        explanation = f"""
Semantic Similarity Analysis ({result.method}):
- Overall Score: {result.score:.3f}
- Confidence: {result.confidence:.3f}
- Method: {result.explanation}

Detailed Breakdown:
"""
        
        for key, value in result.details.items():
            if isinstance(value, (int, float)):
                explanation += f"  {key}: {value:.3f}\n"
            elif isinstance(value, list) and len(value) <= 10:
                explanation += f"  {key}: {[f'{v:.3f}' if isinstance(v, float) else v for v in value]}\n"
            else:
                explanation += f"  {key}: {str(value)[:100]}...\n"
        
        return explanation


# Factory function for easy integration
def create_similarity_calculator(config: Optional[SemanticSimilarityConfig] = None) -> SemanticSimilarityCalculator:
    """Create semantic similarity calculator with optional configuration."""
    if config is None:
        config = SemanticSimilarityConfig()
    
    return SemanticSimilarityCalculator(config)


# Legacy compatibility function
def calculate_snippet_grounding_score_enhanced(
    answer: str, 
    snippets: List[SourceSnippet],
    method: SemanticSimilarityMethod = SemanticSimilarityMethod.HYBRID_WEIGHTED
) -> float:
    """Enhanced grounding score calculation with backward compatibility."""
    config = SemanticSimilarityConfig(method=method)
    calculator = SemanticSimilarityCalculator(config)
    result = calculator.calculate_grounding_score(answer, snippets)
    return result.score
