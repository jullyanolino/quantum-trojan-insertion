#!/usr/bin/env python3
"""
Example Usage: Quantum Trojan Insertion Analysis

This script demonstrates the core functionality of the quantum Trojan insertion
technique described in the paper. It shows how to:
1. Create benchmark quantum circuits
2. Insert controlled Trojans
3. Analyze the impact on circuit behavior
4. Evaluate security metrics

Author: Implementation based on research by Jayden John, Lakshman Golla, Qian Wang
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from quantum_trojan import QuantumTrojanInserter, create_benchmark_circuits
import pandas as pd

def comprehensive_trojan_analysis():
    """
    Perform comprehensive analysis of quantum Trojan insertion across multiple circuits.
    """
    print("🔬 Quantum Trojan Insertion Analysis")
    print("=" * 50)
    
    # Initialize Trojan inserter
    trojan_inserter = QuantumTrojanInserter(control_qubit_index=0)
    
    # Create benchmark circuits
    benchmark_circuits = create_benchmark_circuits()
    
    # Analysis results storage
    results = []
    
    print("\n📊 Analyzing Benchmark Circuits...")
    print("-" * 30)
    
    for circuit_name, circuit in benchmark_circuits.items():
        print(f"\n🔍 Analyzing: {circuit_name}")
        print(f"   Qubits: {circuit.num_qubits}, Depth: {circuit.depth()}, Gates: {len(circuit.data)}")
        
        # Insert Trojan with different activation probabilities
        for activation_prob in [0.1, 0.3, 0.5]:
            # Insert controlled Trojan
            trojan_circuit, trojan_info = trojan_inserter.insert_controlled_trojan(
                circuit, 
                gate_limit=5, 
                activation_prob=activation_prob
            )
            
            # Simulate both circuits
            original_counts = trojan_inserter.simulate_circuit(circuit, shots=1000)
            trojan_counts = trojan_inserter.simulate_circuit(trojan_circuit, shots=1000)
            
            # Calculate metrics
            tvd = trojan_inserter.calculate_tvd(original_counts, trojan_counts, 1000)
            
            # Collect results
            result = {
                'Circuit': circuit_name,
                'Activation_Prob': activation_prob,
                'Original_Depth': circuit.depth(),
                'Trojan_Depth': trojan_circuit.depth(),
                'Depth_Increase': trojan_circuit.depth() - circuit.depth(),
                'Original_Gates': len(circuit.data),
                'Trojan_Gates': len(trojan_circuit.data),
                'Gate_Increase': len(trojan_circuit.data) - len(circuit.data),
                'Gate_Increase_Pct': ((len(trojan_circuit.data) - len(circuit.data)) / len(circuit.data)) * 100,
                'TVD': tvd,
                'Trojans_Inserted': len(trojan_info),
                'Stealth_Score': calculate_stealth_score(tvd, trojan_circuit.depth() - circuit.depth())
            }
            results.append(result)
            
            print(f"   Activation Prob: {activation_prob:.1f} | TVD: {tvd:.3f} | Gates Added: {result['Gate_Increase']}")
    
    # Convert to DataFrame for analysis
    df_results = pd.DataFrame(results)
    
    # Display summary statistics
    print("\n📈 Summary Statistics")
    print("=" * 30)
    
    # Group by circuit
    circuit_summary = df_results.groupby('Circuit').agg({
        'TVD': ['mean', 'std'],
        'Depth_Increase': 'mean',
        'Gate_Increase_Pct': 'mean',
        'Stealth_Score': 'mean'
    }).round(3)
    
    print("\nCircuit-wise Performance:")
    print(circuit_summary)
    
    # Activation probability analysis
    print("\n🎯 Impact of Activation Probability")
    print("-" * 35)
    
    activation_analysis = df_results.groupby('Activation_Prob').agg({
        'TVD': ['mean', 'std'],
        'Gate_Increase': 'mean',
        'Stealth_Score': 'mean'
    }).round(3)
    
    print(activation_analysis)
    
    # Key findings
    print("\n🔍 Key Findings")
    print("=" * 20)
    
    avg_tvd = df_results['TVD'].mean()
    avg_depth_increase = df_results['Depth_Increase'].mean()
    avg_gate_increase = df_results['Gate_Increase_Pct'].mean()
    zero_depth_increase_pct = (df_results['Depth_Increase'] == 0).mean() * 100
    
    print(f"✅ Average TVD: {avg_tvd:.3f} (Higher is more impactful)")
    print(f"✅ Average Depth Increase: {avg_depth_increase:.1f} layers")
    print(f"✅ Average Gate Increase: {avg_gate_increase:.1f}%")
    print(f"✅ Circuits with Zero Depth Increase: {zero_depth_increase_pct:.1f}%")
    
    # Security assessment
    print("\n🛡️ Security Assessment")
    print("=" * 25)
    
    high_impact