import json
import os
import argparse
import time
import csv
from collections import defaultdict

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: 'transformers' or 'torch' not installed. You can still restructure the data, but running Cosmos will require these libraries.")

def load_and_restructure_data(file_path):
    """
    Loads JSON sensor data and restructures it by combining sensors 
    (Radar, LiDAR, EO, IR) across small time windows into a world state.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Sort chronologically
    data.sort(key=lambda x: x['time'])

    sensor_map = {
        1: "Radar",
        2: "LiDAR",
        3: "EO (Electro-Optical camera)",
        4: "IR (Infrared / thermal camera)"
    }

    # Group by time windows (e.g., 0.5 sec) to create a 'world state'
    world_states = defaultdict(lambda: {
        "time_window": 0,
        "ownshipPosition": [],
        "sensors": {
            "Radar": [], 
            "LiDAR": [], 
            "EO (Electro-Optical camera)": [], 
            "IR (Infrared / thermal camera)": []
        }
    })

    window_size = 0.5
    for entry in data:
        t = entry['time']
        window = round(t / window_size) * window_size
        
        state = world_states[window]
        state["time_window"] = window
        # Approximate ownship position for the window
        if not state["ownshipPosition"]:
            state["ownshipPosition"] = entry['ownshipPosition']
        
        s_name = sensor_map.get(entry['sensorID'], f"Unknown_{entry['sensorID']}")
        
        # Append measurements safely
        if 'measurement' in entry and entry['measurement']:
            if isinstance(entry['measurement'], list):
                 state["sensors"][s_name].extend(entry['measurement'])
            else:
                 state["sensors"][s_name].append(entry['measurement'])
        
        if 'measurements' in entry and entry['measurements']:
            if isinstance(entry['measurements'], list):
                 state["sensors"][s_name].extend(entry['measurements'])
            else:
                 state["sensors"][s_name].append(entry['measurements'])

    # Convert to list and sort by time order
    world_states_list = sorted(list(world_states.values()), key=lambda x: x['time_window'])
    return world_states_list

def feed_to_cosmos(world_states, model_id="embedl/Cosmos-Reason2-2B-W4A16", csv_log_file="cosmos_responses.csv"):
    """
    Feeds the restructured world state data into the Nvidia Cosmos LLM 
    to analyze fusion/tracking, and logs the performance.
    """
    if not TRANSFORMERS_AVAILABLE:
        print("Please install transformers and torch to run the model:")
        print("pip install torch transformers accelerate")
        return

    print(f"Loading model '{model_id}'...")
    import transformers
    
    # Load model exactly as per Cosmos documentation
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Load properly as Qwen3VL Processor
    processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Initialize CSV logging
    file_exists = os.path.isfile(csv_log_file)
    with open(csv_log_file, mode='a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["Timestamp", "Time Window", "Prompt", "Output", "Latency (s)"])

        # Feed the first few windows to simulate a continuous state tracking
        for i, state in enumerate(world_states[:5]): # Limit to first 5 for demonstration
            print(f"\n================ Processing Time Window {i+1} ================")
            
            user_msg = f"Time window: {state['time_window']}\nOwnship Position: {state['ownshipPosition']}\nSensor Data:\n"
            for s_name, s_data in state["sensors"].items():
                user_msg += f"- {s_name}: {len(s_data)} readings detected. Data: {s_data}\n"
            user_msg += "\nWhat is the current world state based on this fusion? Are there tracked objects?"

            # Apply chat template structure specified in docs
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are an AI tracking system analyzing sensor fusion data (Radar, LiDAR, EO, IR). Your job is to analyze the data, track targets, and determine the current world state."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_msg},
                    ],
                },
            ]

            # Process inputs via processor
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                truncation=False,
            )
            inputs = inputs.to(model.device)
            
            # Start timer
            start_time = time.time()
            
            # Run inference
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            
            # End timer
            latency = time.time() - start_time
            
            # Trim the generated ids to only contain the output sequence
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(
                    inputs.input_ids, generated_ids, strict=False
                )
            ]
            response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            print(f"Latency: {latency:.2f} seconds")
            print("Cosmos Output:\n", response)

            # Log to CSV
            csv_writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                state['time_window'],
                user_msg,
                response,
                f"{latency:.4f}"
            ])
            csvfile.flush() # Ensure it's written immediately

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process sensor data and feed to Cosmos")
    parser.add_argument("--input", default="scenario4_detections.json", help="Input JSON file with detections")
    parser.add_argument("--output", default="restructured_world_state.json", help="Output restructured JSON")
    parser.add_argument("--csv", default="cosmos_responses.csv", help="CSV file to store logs")
    parser.add_argument("--run-model", action="store_true", help="Whether to download and run the Cosmos model")
    
    args = parser.parse_args()

    input_file_path = args.input
    output_file_path = args.output
    csv_file_path = args.csv

    if not os.path.exists(input_file_path):
         print(f"Error: Input file {input_file_path} not found.")
         exit(1)

    print(f"Restructuring data from {input_file_path}...")
    restructured_data = load_and_restructure_data(input_file_path)
    
    # Save the restructured data
    with open(output_file_path, "w") as f:
        json.dump(restructured_data, f, indent=2)
    print(f"Success! Saved restructured data ({len(restructured_data)} time windows) to {output_file_path}")

    if args.run_model:
        feed_to_cosmos(restructured_data, csv_log_file=csv_file_path)
    else:
        print("\nSkipping Cosmos run. To run the model, execute:")
        print(f"python {os.path.basename(__file__)} --run-model")
