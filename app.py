import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(BASE_DIR, "catalog.json")

# Load catalog once
catalog_data = []
if os.path.exists(CATALOG_PATH):
    try:
        with open(CATALOG_PATH, "r") as f:
            catalog_data = json.load(f)
    except Exception as e:
        print(f"Error loading catalog.json: {e}")

def simple_search(query, limit=5):
    if not query:
        return []
    
    query = query.lower()
    matches = []
    
    for item in catalog_data:
        doc = item.get("doc", "").lower()
        meta = item.get("meta", {})
        title = meta.get("title", "").lower()
        category = meta.get("category", "").lower()
        
        # Simple keyword match
        score = 0
        if query in title: score += 10
        if query in category: score += 5
        if query in doc: score += 2
        
        if score > 0:
            matches.append({
                "score": score,
                "doc": item.get("doc"),
                "meta": meta
            })
            
    # Sort by score
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:limit]

@app.route('/')
def catalog():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/chat')
def chat():
    return send_from_directory(BASE_DIR, 'chat.html')

@app.route('/api/skillsbuild/search', methods=['GET'])
def skillsbuild_search_get():
    query = request.args.get('q', '')
    limit = int(request.args.get('n', 5))
    
    results = simple_search(query, limit)
    
    courses = []
    for i, res in enumerate(results):
        meta = res["meta"]
        courses.append({
            "rank": i + 1,
            "title": meta.get("title", "N/A"),
            "url": meta.get("url", ""),
            "category": meta.get("category", "General"),
            "audience": meta.get("audience", "All Learners"),
            "duration": meta.get("duration", "Self-paced"),
            "description": res["doc"][:300] if res["doc"] else "",
            "similarity": 0.95 - (i * 0.05) # Mocked similarity for UI
        })
            
    return jsonify({
        "query": query,
        "total_results": len(courses),
        "courses": courses
    })

# API for the chatbot (POST /api/search)
@app.route('/api/search', methods=['POST'])
def chatbot_search():
    data = request.json or {}
    query = data.get("query", "")
    
    results = simple_search(query, limit=3)
    
    formatted_results = []
    for i, res in enumerate(results):
        meta = res["meta"]
        # Format string similar to what the chatbot regex expects: "Course: Title. Eligibility: Audience. Duration: time"
        formatted_results.append(f"{i+1}. Course: {meta.get('title')}. Eligibility: {meta.get('audience')}. Duration: {meta.get('duration')}")
            
    return jsonify({
        "results": formatted_results
    })

if __name__ == '__main__':
    app.run(port=5003, debug=True)
