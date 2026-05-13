import os
import time
import threading
from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__)

# Basic in-memory state for the simulation
state = {
    "status": "IDLE",
    "logs": [],
    "progress": 0,
    "results": None
}

def log(msg):
    state["logs"].append(msg)
    if len(state["logs"]) > 50:
        state["logs"].pop(0)

def mock_gcp_workflow(params):
    state["status"] = "PROVISIONING"
    state["progress"] = 10
    state["logs"] = []
    log("Initializing pyWarpfactory Cloud Engine...")
    time.sleep(1)
    
    log("Authenticating via Application Default Credentials (ADC)...")
    time.sleep(1)
    
    log(f"Provisioning Google Cloud VM (n2-highcpu-32) in us-central1-a...")
    time.sleep(3)
    
    state["status"] = "COMPUTING"
    state["progress"] = 35
    log("Secure IAP tunnel established. Uploading metric parameters...")
    time.sleep(2)
    
    log(f"Discretizing 3D Grid: {params.get('resolution', 100)}^3 points...")
    time.sleep(2)
    
    state["progress"] = 55
    log("Performing 3+1 ADM Decomposition of Spacetime...")
    time.sleep(3)
    
    state["progress"] = 75
    log("Solving Einstein Field Equations (Stress-Energy Tensor)...")
    log("Checking Null, Weak, and Dominant Energy Conditions...")
    time.sleep(4)
    
    state["status"] = "RETRIEVING"
    state["progress"] = 90
    log("Negative Energy Density (Violations) detected in the warp bubble annulus.")
    log("Rendering Eulerian Observer heatmaps...")
    time.sleep(2)
    
    log("Retrieving artifacts via SCP...")
    time.sleep(1)
    
    log("Destroying GCP Compute instance to optimize billing...")
    time.sleep(1)
    
    state["status"] = "COMPLETED"
    state["progress"] = 100
    log("Workflow Complete. Visualizations ready.")
    
    # Mock result data
    state["results"] = {
        "max_negative_energy": "-1.45e32 J/m^3",
        "violation_volume": "12.4 m^3",
        "metric_type": params.get('metric', 'Alcubierre')
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    return jsonify(state)

@app.route('/api/dispatch', methods=['POST'])
def dispatch():
    if state["status"] not in ["IDLE", "COMPLETED"]:
        return jsonify({"error": "A job is already running."}), 400
    
    params = request.json
    threading.Thread(target=mock_gcp_workflow, args=(params,), daemon=True).start()
    return jsonify({"success": True})

@app.route('/api/reset', methods=['POST'])
def reset():
    state["status"] = "IDLE"
    state["logs"] = []
    state["progress"] = 0
    state["results"] = None
    return jsonify({"success": True})

if __name__ == '__main__':
    print("Starting pyWarpfactory UI on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)
