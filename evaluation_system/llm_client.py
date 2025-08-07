"""
LLM client for the RAG vs Agentic AI Evaluation Framework.

This module encapsulates all interactions with LLM APIs (OpenAI and Azure OpenAI)
including retry logic, JSON parsing, and response validation.
"""

import os
import json
import statistics
from datetime import datetime
from typing import Optional, Dict, Any

import openai
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, ClientSecretCredential, get_bearer_token_provider
from dotenv import load_dotenv

from .config import logger, EvalConfig
from .enums import EvaluationDimension, SystemType
from .models import AnswerEvaluation, EvaluationRubric, SourceSnippet
from .templates import load_system_prompt, render_template, SNIPPET_EVALUATION_PROMPT, STANDARD_EVALUATION_PROMPT


def create_openai_client() -> openai.OpenAI:
    """
    Create OpenAI client based on available environment variables.
    
    Supports both Azure OpenAI and standard OpenAI APIs.
    Prioritizes Azure OpenAI if endpoint is configured.
    
    Returns:
        OpenAI client instance
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Set NO_PROXY for local development (matching your working config)
    os.environ["NO_PROXY"] = "localhost,127.0.1,::1"
    
    # Check for Azure OpenAI configuration (using exact variable names from your working code)
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://begobaiatest.openai.azure.com/")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    
    # Service principal credentials
    tenant_id = os.getenv("AZURE_TENANT_ID")
    client_id = os.getenv("AZURE_CLIENT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")
    
    # Validate required environment variables
    if not all([tenant_id, client_id, client_secret]):
        logger.warning("Missing required Azure environment variables: AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET")
        logger.info("Falling back to standard OpenAI client")
        
        # Fall back to standard OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Neither Azure OpenAI credentials nor OPENAI_API_KEY is configured")
        
        logger.info("Using standard OpenAI API")
        return openai.OpenAI(api_key=api_key)
    
    try:
        logger.info(f"Using Azure OpenAI with endpoint: {azure_endpoint}")
        logger.info(f"Deployment: {deployment_name}")
        logger.info(f"API Version: {api_version}")
        logger.info(f"Tenant ID: {tenant_id}")
        
        # Initialize Azure credential with service principal (exactly matching your working code)
        credential = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret
        )
        
        # Create token provider for Azure OpenAI
        token_provider = get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default"
        )
        
        # Initialize Azure OpenAI client with service principal authentication
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )
        
        logger.info("Azure OpenAI client created successfully")
        return client
        
    except Exception as e:
        logger.warning(f"Failed to create Azure OpenAI client: {e}")
        logger.info("Falling back to standard OpenAI client")
        
        # Fall back to standard OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(f"Azure OpenAI failed and no OPENAI_API_KEY available: {e}")
        
        logger.info("Using standard OpenAI API")
        return openai.OpenAI(api_key=api_key)


def get_model_name(config: EvalConfig) -> str:
    """
    Get the appropriate model name based on the client type.
    
    For Azure OpenAI, uses the deployment name.
    For standard OpenAI, uses the model name from config.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Model or deployment name to use
    """
    # Check if using Azure OpenAI (using exact variable names from your working code)
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    
    if azure_endpoint and deployment_name:
        logger.debug(f"Using Azure deployment: {deployment_name}")
        return deployment_name
    
    logger.debug(f"Using OpenAI model: {config.model_name}")
    return config.model_name


def call_llm_evaluator(
    query: str, 
    answer: str, 
    config: EvalConfig,
    system_type: Optional[SystemType] = None,
    snippets: Optional[list[SourceSnippet]] = None
) -> Optional[AnswerEvaluation]:
    """
    Enhanced LLM evaluation with academic rigor and snippet support.
    
    Args:
        query: The original user query
        answer: The system response to evaluate
        config: Evaluation configuration with rubrics
        system_type: Optional system type annotation
        snippets: Optional list of source snippets for RAG evaluation
    
    Returns:
        AnswerEvaluation object or None if evaluation fails
    """
    timestamp = datetime.now().isoformat()
    
    # Choose appropriate template based on whether snippets are provided
    if snippets and len(snippets) > 0:
        prompt_template = SNIPPET_EVALUATION_PROMPT
        logger.debug(f"Using snippet-enhanced evaluation template with {len(snippets)} snippets")
    else:
        prompt_template = STANDARD_EVALUATION_PROMPT
        logger.debug("Using standard evaluation template")
    
    # Render the prompt with query, answer, snippets, and configuration
    rendered_prompt = render_template(
        prompt_template,
        query=query,
        answer=answer,
        snippets=snippets or [],
        dimensions=config.dimensions,
        rubrics=config.rubrics
    )
    
    messages = [
        {"role": "system", "content": rendered_prompt}
    ]
    
    for attempt in range(1, config.max_retries + 1):
        try:
            # Create appropriate OpenAI client (Azure or standard)
            client = create_openai_client()
            model_name = get_model_name(config)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=config.temperature,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            logger.debug(f"LLM Response (attempt {attempt}): {content[:200]}...")
            
            # DEBUG: Log the full response for troubleshooting
            logger.info(f"=== FULL LLM RESPONSE (Attempt {attempt}) ===")
            logger.info(content)
            logger.info("=== END RESPONSE ===")
            
            data = json.loads(content)
            
            # Enhanced validation
            if not validate_evaluation_response(data, config.dimensions):
                raise ValueError("Invalid evaluation response format")
            
            # Convert string keys back to enum dimensions
            scores = {EvaluationDimension(k): v for k, v in data["scores"].items()}
            justifications = {EvaluationDimension(k): v for k, v in data["justifications"].items()}
            confidence_scores = {EvaluationDimension(k): v for k, v in data["confidence_scores"].items()}
            
            # Calculate weighted score
            weighted_total = calculate_weighted_score(scores, config.rubrics)
            raw_total = sum(scores.values())
            
            # Check confidence threshold
            avg_confidence = statistics.mean(confidence_scores.values())
            if avg_confidence < config.confidence_threshold:
                logger.warning(f"Low confidence evaluation: {avg_confidence:.2f}")
            
            # Build evaluation metadata
            evaluation_metadata = {
                "model_used": model_name,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "timestamp": timestamp,
                "system_type": system_type.value if system_type else None,
                "has_snippets": bool(snippets and len(snippets) > 0)
            }
            
            # Add snippet analysis if present
            if data.get("snippet_analysis"):
                evaluation_metadata["snippet_analysis"] = data["snippet_analysis"]
            
            return AnswerEvaluation(
                answer=answer,
                system_type=system_type,
                scores=scores,
                justifications=justifications,
                confidence_scores=confidence_scores,
                weighted_total=weighted_total,
                raw_total=raw_total,
                evaluation_metadata=evaluation_metadata,
                overall_assessment=data.get("overall_assessment", ""),
                comparative_notes=data.get("academic_recommendation", ""),
                source_snippets=snippets,
                snippet_grounding_score=None,  # Will be calculated separately
                citation_snippet_alignment=None  # Will be calculated separately
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt} failed - JSON decode error: {e}")
        except Exception as e:
            logger.warning(f"Attempt {attempt} failed - {type(e).__name__}: {e}")
            
        if attempt < config.max_retries:
            logger.info(f"Retrying evaluation (attempt {attempt + 1}/{config.max_retries})")
    
    logger.error("LLM evaluation failed after all retries")
    return None


def test_llm_connection(config: EvalConfig) -> bool:
    """
    Test connection to the LLM API with a simple query.
    
    Args:
        config: Evaluation configuration
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        client = create_openai_client()
        model_name = get_model_name(config)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Test connection"}],
            max_tokens=10,
            temperature=0.0
        )
        logger.info(f"LLM connection test successful with model: {model_name}")
        return True
    except Exception as e:
        logger.error(f"LLM connection test failed: {e}")
        return False

from .utils import calculate_weighted_score, validate_evaluation_response


