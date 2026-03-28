import json
import numpy as np
from PIL import Image
import argparse
import os

def convert_to_tiff(input_json, output_tiff, image_size=2048):
    """
    Reads the raw sensor data from a JSON file, rasterizes the 
    ownship positions and available spatial measurements (e.g., Radar/LiDAR) 
    onto a 2D image, and saves it as a TIFF file.
    """
    if not os.path.exists(input_json):
        print(f"Error: Input file {input_json} not found.")
        return

    print(f"Loading data from {input_json}...")
    with open(input_json, 'r') as f:
        data = json.load(f)

    # Collect ownship positions and 2D target measurements
    ownship_positions = []
    target_positions = []

    for item in data:
        # Extract ownship position
        if 'ownshipPosition' in item and isinstance(item['ownshipPosition'], list) and len(item['ownshipPosition']) >= 2:
            ox, oy = item['ownshipPosition'][:2]
            ownship_positions.append([ox, oy])
            
            # Extract spatial measurements (e.g., Radar:1 or LiDAR:2)
            # Assuming measurements for these sensors are relative or absolute Cartesian coordinates
            # Looking at sample data, some measurements are [x,y] or [[x,y], [x,y]]
            # Add them as target dots around/from the ownship
            sensor_id = item.get('sensorID')
            measure_data = item.get('measurement', [])
            
            if sensor_id in (1, 2) and isinstance(measure_data, list):
                if len(measure_data) > 0:
                    if isinstance(measure_data[0], list): # e.g. [[x,y], ...]
                        for pt in measure_data:
                            if len(pt) >= 2:
                                # They could be relative, so we add ox, oy. If absolute, do not add.
                                # Let's assume relative for now based on typical sensor data outputs
                                target_positions.append([ox + pt[0], oy + pt[1]])
                    elif len(measure_data) >= 2 and isinstance(measure_data[0], (int, float)):
                        # Single [x,y] measurement
                        target_positions.append([ox + measure_data[0], oy + measure_data[1]])

    if not ownship_positions:
        print("No valid ownship positions found in the data.")
        return

    # Determine bounding box
    all_points = np.array(ownship_positions + target_positions)
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)

    # Add a margin
    margin_x = (max_x - min_x) * 0.05
    margin_y = (max_y - min_y) * 0.05
    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    range_x = max_x - min_x
    range_y = max_y - min_y
    
    if range_x == 0: range_x = 1.0
    if range_y == 0: range_y = 1.0

    print("Rasterizing points onto the image grid...")
    # Create black RGB image
    img_array = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    def to_pixel(x, y):
        px = int(((x - min_x) / range_x) * (image_size - 1))
        # Invert y-axis for image format (optional, image typically puts (0,0) top left)
        py = int((1.0 - (y - min_y) / range_y) * (image_size - 1))
        return min(max(px, 0), image_size - 1), min(max(py, 0), image_size - 1)

    # Draw target positions (Red)
    for x, y in target_positions:
        px, py = to_pixel(x, y)
        radius = 1
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if 0 <= py+dy < image_size and 0 <= px+dx < image_size:
                    img_array[py+dy, px+dx] = [255, 30, 30]

    # Draw ownship positions (Cyan)
    for x, y in ownship_positions:
        px, py = to_pixel(x, y)
        radius = 2
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if 0 <= py+dy < image_size and 0 <= px+dx < image_size:
                    img_array[py+dy, px+dx] = [30, 255, 255]

    print(f"Saving to {output_tiff}...")
    try:
        img = Image.fromarray(img_array)
        img.save(output_tiff, format='TIFF')
        print("Success!")
    except ImportError:
        print("Error: Pillow is required to save TIFF files. Run 'pip install Pillow'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw sensor JSON to TIFF map")
    parser.add_argument("--input", default="scenario4_detections.json", help="Input JSON file with detections")
    parser.add_argument("--output", default="sensor_map.tiff", help="Output TIFF file")
    parser.add_argument("--size", type=int, default=2048, help="Output image size (width/height)")
    
    args = parser.parse_args()
    
    convert_to_tiff(args.input, args.output, args.size)
