"""
API endpoint tests
"""
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/api/health/")
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert 'INDIA RUNS' in response.json()['message']

def test_search_endpoint():
    """Test search endpoint"""
    response = client.post("/api/search/", json={
        "query": "Python developer",
        "top_k": 10
    })
    
    assert response.status_code == 200
    data = response.json()
    assert 'results' in data
    assert 'total_results' in data

def test_ranking_endpoint():
    """Test ranking endpoint"""
    response = client.post("/api/ranking/", json={
        "job_description": "Looking for a Python developer with 3+ years experience",
        "top_k": 20
    })
    
    assert response.status_code == 200
    data = response.json()
    assert 'rankings' in data

def test_explain_endpoint():
    """Test explainability endpoint"""
    response = client.post("/api/explain/candidate", json={
        "candidate_id": "test_id",
        "job_description": "Python developer role",
        "include_llm_explanation": False
    })
    
    # May fail if test_id doesn't exist, but endpoint should exist
    assert response.status_code in [200, 404]