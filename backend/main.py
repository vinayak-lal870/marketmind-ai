import os
from typing import List, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="MarketMind AI — Business Growth Intelligence API",
    description="Autonomous Opportunity Discovery & Business DNA Engine powered by Google Gemini",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google Gemini client (expects GEMINI_API_KEY in environment)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# ---------------------------------------------------------
# 1. Structured Schemas (Zero-Hallucination Guardrails)
# ---------------------------------------------------------
class BusinessDNAInput(BaseModel):
    business_name: str = Field(..., example="Deccan Looms D2C")
    industry: str = Field(..., example="Apparel & Handloom Fashion")
    location: str = Field(..., example="Hyderabad, India")
    products_services: List[str] = Field(..., example=["Handloom cotton kurtas", "Linen office wear"])
    target_customers: List[str] = Field(..., example=["Urban professionals aged 25-40"])
    price_range: str = Field(..., example="₹1,499 - ₹3,999")
    current_goals: List[str] = Field(..., example=["Increase repeat orders", "Expand to Bengaluru"])
    known_competitors: List[str] = Field(default=[], example=["FabIndia", "Okhai"])


class GrowthOpportunity(BaseModel):
    title: str
    category: Literal["underserved_segment", "competitor_blindspot", "product_expansion", "geographic_wedge"]
    description: str
    epistemic_label: Literal["verified_fact", "sourced_evidence", "ai_hypothesis"]
    supporting_reasoning: str
    target_audience: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    suggested_experiment: str
    estimated_budget_inr: int


class IntelligenceReport(BaseModel):
    business_summary: str
    core_strengths: List[str]
    identified_constraints: List[str]
    opportunities: List[GrowthOpportunity]


# ---------------------------------------------------------
# 2. Core Endpoints
# ---------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "MarketMind AI Core Engine",
        "gemini_configured": bool(GEMINI_API_KEY),
    }


@app.post("/api/v1/analyze-dna", response_model=IntelligenceReport)
def analyze_business_dna(dna: BusinessDNAInput):
    """
    Ingests Business DNA and generates structured, evidence-labeled growth opportunities
    using Gemini Structured Outputs.
    """
    if not client:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY not configured on server. Set environment variable to run live inference.",
        )

    system_instruction = (
        "You are MarketMind AI, an autonomous Business Growth Intelligence Engine. "
        "Analyze the provided Business DNA and discover 3 high-impact, realistic growth opportunities. "
        "CRITICAL RULE: Never invent fake market statistics or pretend estimates are verified facts. "
        "Strictly label every opportunity as 'ai_hypothesis' unless directly derived from user-provided facts."
    )

    prompt = f"""
    Analyze this business profile and generate a structured growth intelligence report:
    - Business Name: {dna.business_name}
    - Industry: {dna.industry}
    - Location: {dna.location}
    - Products/Services: {', '.join(dna.products_services)}
    - Target Customers: {', '.join(dna.target_customers)}
    - Price Range: {dna.price_range}
    - Current Goals: {', '.join(dna.current_goals)}
    - Known Competitors: {', '.join(dna.known_competitors) if dna.known_competitors else 'None specified'}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=IntelligenceReport,
                temperature=0.3,
            ),
        )
        return IntelligenceReport.model_validate_json(response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(exc)}")
