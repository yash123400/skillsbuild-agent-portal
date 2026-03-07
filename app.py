import os
import json
import sys

# Override sqlite3 for ChromaDB compatibility on Vercel
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Use absolute path relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "skillsbuild_memory")

# Initialize ChromaDB client lazily
_client = None
_collection = None

def get_collection():
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
        _collection = _client.get_collection(name="courses")
    return _collection

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/api/skillsbuild/search', methods=['GET'])
def skillsbuild_search():
    query = request.args.get('q', '')
    limit = int(request.args.get('n', 5))
    
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    
    try:
        collection = get_collection()
        results = collection.query(
            query_texts=[query],
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )
        
        courses = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            courses.append({
                "rank": i + 1,
                "title": meta.get("title", "N/A"),
                "url": meta.get("url", ""),
                "category": meta.get("category", "General"),
                "audience": meta.get("audience", "All Learners"),
                "duration": meta.get("duration", "Self-paced"),
                "description": doc[:300] if doc else "",
                "similarity": round(1 - dist, 4)
            })
            
        return jsonify({
            "query": query,
            "total_results": len(courses),
            "courses": courses
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({"error": "Search failed", "details": str(e), "traceback": error_details}), 500

if __name__ == '__main__':
    # Running locally
    app.run(port=5003, debug=True)
