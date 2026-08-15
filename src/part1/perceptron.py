import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

def perceptron_output(weights: np.ndarray, bias: float, inputs: np.ndarray) -> int:
    """
    Calculates the output of a single perceptron.
    
    Args:
        weights (np.ndarray): The weights for the inputs.
        bias (float): The bias value.
        inputs (np.ndarray): The input values.
        
    Returns:
        int: 1 if the linear combination is greater than 0, otherwise 0.
    """
    linear_combination = np.dot(weights, inputs) - bias
    return 1 if linear_combination > 0 else 0

def classify_and_plot(weights: np.ndarray, bias: float, points: np.ndarray) -> None:
    """
    Classifies a set of points using a perceptron and plots the decision boundary.
    
    Args:
        weights (np.ndarray): The weights for the inputs.
        bias (float): The bias value.
        points (np.ndarray): A 2D array of input points (x, y).
    """
    num_points = len(points)
    classified_points = np.zeros((num_points, 3))
    
    for i in range(num_points):
        predicted_class = perceptron_output(weights, bias, points[i])
        classified_points[i] = [points[i][0], points[i][1], predicted_class]

    # Handle divide by zero if weights[1] is 0
    if weights[1] == 0:
        print("Error: Weight for Y-axis cannot be zero for this plotting logic.")
        return

    x_boundary = bias / weights[0] if weights[0] != 0 else 0
    y_boundary = bias / weights[1]
    
    print("\nClassified dataset:")
    for x, y, label in classified_points:
        print(f"Point ({x}, {y}) => Predicted Class: {int(label)}")
        
    print(f"\nDecision boundary intercepts: X-axis = {x_boundary}, Y-axis = {y_boundary}")

    # Plotting
    try:
        plt.figure(figsize=(8, 6))
        for x, y, label in classified_points:
            if label == 1:
                plt.scatter(x, y, color='blue', label='Class 1' if 'Class 1' not in plt.gca().get_legend_handles_labels()[1] else "")
            else:
                plt.scatter(x, y, color='red', label='Class 0' if 'Class 0' not in plt.gca().get_legend_handles_labels()[1] else "")

        x_vals = np.linspace(min(points[:, 0]) - 1, max(points[:, 0]) + 1, 100)
        y_vals = -(weights[0] * x_vals - bias) / weights[1]
        
        plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Single Perceptron Classification")
        plt.legend()
        plt.grid()
        
        # In a headless environment, we might want to save the plot instead of showing it
        if os.getenv("HEADLESS") == "true":
            plt.savefig("perceptron_classification.png")
            print("Plot saved to perceptron_classification.png")
        else:
            plt.show()
            
    except Exception as e:
        print(f"Failed to generate plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single perceptron classification algorithm.")
    parser.add_argument("--weights", type=str, default="1.0,1.0", help="Comma-separated weights (e.g., '1.0,1.0')")
    parser.add_argument("--bias", type=float, default=1.5, help="Bias value (e.g., 1.5)")
    parser.add_argument("--points", type=str, default="0,0;0,1;1,0;1,1", help="Semicolon-separated points, where each point is comma-separated x,y (e.g., '0,0;0,1')")
    
    args = parser.parse_args()
    
    try:
        w = np.array(list(map(float, args.weights.split(','))))
        
        # Parse points safely
        p_list = []
        for p_str in args.points.split(';'):
            x_str, y_str = p_str.split(',')
            p_list.append([float(x_str), float(y_str)])
            
        p = np.array(p_list)
        
        classify_and_plot(w, args.bias, p)
        
    except ValueError as ve:
        print(f"Invalid input data format: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
