import requests
import time
import json

BASE_URL = "http://localhost:8000"

def print_section(title):
    print(f"\n{'='*50}\n{title}\n{'='*50}")

try:
    # 1. Health Check
    print_section("1. Health Check")
    r = requests.get(f"{BASE_URL}/health")
    print(json.dumps(r.json(), indent=2))

    # 2. Check Feature Flags (Default should be minilm)
    print_section("2. Feature Flags")
    r = requests.get(f"{BASE_URL}/flags")
    print("Current Flags:", json.dumps(r.json(), indent=2))

    # 3. Embed some documents
    print_section("3. Embedding Documents (using MiniLM Model)")
    docs = [
        "KeaBuilder is the best landing page tool for SaaS",
        "We offer a seamless drag-and-drop website builder experience",
        "Pricing plans for enterprise teams using our funnel builder",
        "How to track conversions and bounce rates on your landing page"
    ]
    
    for i, doc in enumerate(docs):
        r = requests.post(f"{BASE_URL}/embed", json={"text": doc, "payload": {"id": i, "category": "demo"}})
        print(f"Embedded [{i}]:", r.json())

    # 4. Dense Search
    print_section("4. Dense Search (Semantic)")
    r = requests.post(f"{BASE_URL}/similarity", json={"query": "cost of using the platform for a big company", "top_k": 2, "method": "dense"})
    print("Results for 'cost of using...':")
    print(json.dumps(r.json(), indent=2))

    # 5. Hybrid Search
    print_section("5. Hybrid Search (Semantic + Exact Keyword)")
    r = requests.post(f"{BASE_URL}/similarity", json={"query": "conversions and bounce rates", "top_k": 2, "method": "hybrid"})
    print("Hybrid Results:")
    print(json.dumps(r.json(), indent=2))

    # 6. Switch Model using Feature Flags
    print_section("6. Switch Model to Student via Feature Flag")
    r = requests.put(f"{BASE_URL}/flags/active-model/student")
    print("Flag Update:", r.json())
    
    # 7. Async Job
    print_section("7. Background Async Job (Celery)")
    r = requests.post(f"{BASE_URL}/predict/async", json={"query": "drag and drop builder", "top_k": 2, "method": "dense"})
    job_info = r.json()
    job_id = job_info["job_id"]
    print("Job Enqueued:", job_info)
    
    print("\nPolling Job Status...")
    for _ in range(5):
        time.sleep(1)
        r = requests.get(f"{BASE_URL}/job/{job_id}")
        status = r.json()
        print(f"Status: {status['status']}")
        if status['status'] == "done":
            print("\nJob Complete! Results:")
            print(json.dumps(status['result'], indent=2))
            break
            
except Exception as e:
    print(f"Error occurred: {e}")
