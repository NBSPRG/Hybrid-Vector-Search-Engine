import urllib.request
import urllib.parse
import json

def test_api():
    base_url = "http://localhost:8000"
    
    # 1. Embed a text
    print("--- 1. Testing POST /embed ---")
    embed_req = urllib.request.Request(
        f"{base_url}/embed",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "text": "How do I create a landing page?",
            "payload": {
                "type": "prompt",
                "text": "How do I create a landing page?"
            }
        }).encode('utf-8')
    )
    
    try:
        with urllib.request.urlopen(embed_req) as response:
            embed_res = json.loads(response.read().decode())
            print(json.dumps(embed_res, indent=2))
    except Exception as e:
        print(f"Failed to embed: {e}")
        return

    # 2. Search for similarity
    print("\n--- 2. Testing POST /similarity ---")
    sim_req = urllib.request.Request(
        f"{base_url}/similarity",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "query": "How do I make an e-commerce website?",
            "top_k": 1,
            "method": "dense"
        }).encode('utf-8')
    )
    
    try:
        with urllib.request.urlopen(sim_req) as response:
            sim_res = json.loads(response.read().decode())
            print(json.dumps(sim_res, indent=2))
    except Exception as e:
        print(f"Failed to search: {e}")

if __name__ == "__main__":
    test_api()
