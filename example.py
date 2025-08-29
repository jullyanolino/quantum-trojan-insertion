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
    
    high_impact_circuits = df_results[df_results['TVD'] > 0.3]['Circuit'].nunique()
    stealth_circuits = df_results[df_results['Depth_Increase'] == 0]['Circuit'].nunique()
    
    print(f"🔴 High Impact Circuits (TVD > 0.3): {high_impact_circuits}/{len(benchmark_circuits)}")
    print(f"🕵️ Perfectly Stealthy Circuits (Zero Depth Increase): {stealth_circuits}/{len(benchmark_circuits)}")
    
    if avg_tvd > 0.3 and zero_depth_increase_pct > 80:
        threat_level = "🔴 CRITICAL"
    elif avg_tvd > 0.2 and zero_depth_increase_pct > 60:
        threat_level = "🟡 HIGH"
    else:
        threat_level = "🟢 MODERATE"
    
    print(f"\n🎯 Overall Threat Level: {threat_level}")
    
    return df_results

def calculate_stealth_score(tvd, depth_increase):
    """
    Calculate a stealth score based on TVD and depth increase.
    Higher scores indicate more stealthy Trojans.
    """
    # Perfect stealth would be high impact (high TVD) with zero depth increase
    depth_penalty = depth_increase * 50  # Heavy penalty for depth increase
    impact_bonus = min(30, tvd * 100)    # Bonus for impact, capped at 30
    
    base_score = 100
    stealth_score = max(0, base_score - depth_penalty + impact_bonus - (tvd * 20))
    
    return stealth_score

def demonstrate_single_circuit_analysis():
    """
    Detailed analysis of a single circuit to show step-by-step process.
    """
    print("\n" + "=" * 60)
    print("🔬 DETAILED SINGLE CIRCUIT ANALYSIS")
    print("=" * 60)
    
    # Create a Bell state circuit for detailed analysis
    bell_circuit = QuantumCircuit(2, name="Bell_State_Demo")
    bell_circuit.h(0)
    bell_circuit.cx(0, 1)
    
    print(f"\n📋 Original Circuit: {bell_circuit.name}")
    print(f"   Qubits: {bell_circuit.num_qubits}")
    print(f"   Depth: {bell_circuit.depth()}")
    print(f"   Gates: {len(bell_circuit.data)}")
    
    print("\n🔗 Circuit Diagram:")
    print(bell_circuit.draw())
    
    # Initialize Trojan inserter
    trojan_inserter = QuantumTrojanInserter(control_qubit_index=0)
    
    # Analyze empty positions
    empty_positions = trojan_inserter.find_empty_positions(bell_circuit)
    print(f"\n🎯 Empty Positions Analysis:")
    for i, positions in enumerate(empty_positions):
        print(f"   Layer {i}: {positions if positions else 'No empty positions'}")
    
    # Insert Trojan with high activation probability for demonstration
    print(f"\n⚡ Inserting Trojan (Activation Probability: 0.8)...")
    trojan_circuit, trojan_info = trojan_inserter.insert_controlled_trojan(
        bell_circuit, 
        gate_limit=3, 
        activation_prob=0.8
    )
    
    print(f"\n🦠 Trojan-Infected Circuit:")
    print(f"   Qubits: {trojan_circuit.num_qubits}")
    print(f"   Depth: {trojan_circuit.depth()}")
    print(f"   Gates: {len(trojan_circuit.data)}")
    
    print("\n🔗 Modified Circuit Diagram:")
    print(trojan_circuit.draw())
    
    # Show inserted Trojan information
    if trojan_info:
        print(f"\n🎯 Inserted Trojan Gates:")
        for i, (gate_type, control, target, layer) in enumerate(trojan_info):
            if target is not None:
                print(f"   {i+1}. {gate_type} gate: Control={control}, Target={target}, Layer={layer}")
            else:
                print(f"   {i+1}. {gate_type} gate: Qubit={control}, Layer={layer}")
    
    # Simulation and comparison
    print(f"\n🔬 Simulation Analysis (1000 shots):")
    
    original_counts = trojan_inserter.simulate_circuit(bell_circuit, shots=1000)
    trojan_counts = trojan_inserter.simulate_circuit(trojan_circuit, shots=1000)
    
    print(f"\n📊 Original Circuit Results:")
    for outcome, count in sorted(original_counts.items()):
        probability = count / 1000
        print(f"   |{outcome}⟩: {count} shots ({probability:.3f})")
    
    print(f"\n📊 Trojan Circuit Results:")
    for outcome, count in sorted(trojan_counts.items()):
        probability = count / 1000
        print(f"   |{outcome}⟩: {count} shots ({probability:.3f})")
    
    # Calculate impact metrics
    tvd = trojan_inserter.calculate_tvd(original_counts, trojan_counts, 1000)
    depth_change = trojan_circuit.depth() - bell_circuit.depth()
    gate_change = len(trojan_circuit.data) - len(bell_circuit.data)
    gate_change_pct = (gate_change / len(bell_circuit.data)) * 100
    
    print(f"\n📈 Impact Metrics:")
    print(f"   Total Variation Distance: {tvd:.4f}")
    print(f"   Depth Change: {depth_change} layers")
    print(f"   Gate Count Change: {gate_change} gates ({gate_change_pct:.1f}%)")
    print(f"   Stealth Score: {calculate_stealth_score(tvd, depth_change):.1f}/100")
    
    # Interpretation
    print(f"\n🔍 Interpretation:")
    
    if tvd > 0.3:
        print("   🔴 HIGH IMPACT: Trojan significantly alters circuit behavior")
    elif tvd > 0.1:
        print("   🟡 MEDIUM IMPACT: Moderate alteration of circuit behavior")
    else:
        print("   🟢 LOW IMPACT: Minimal change in circuit behavior")
    
    if depth_change == 0:
        print("   ✅ PERFECT STEALTH: No depth increase detected")
    elif depth_change <= 2:
        print("   🟡 MODERATE STEALTH: Minor depth increase")
    else:
        print("   🔴 POOR STEALTH: Significant depth increase")
    
    return {
        'original_circuit': bell_circuit,
        'trojan_circuit': trojan_circuit,
        'trojan_info': trojan_info,
        'metrics': {
            'tvd': tvd,
            'depth_change': depth_change,
            'gate_change': gate_change,
            'gate_change_pct': gate_change_pct
        }
    }

def advanced_analysis_scenarios():
    """
    Demonstrate advanced analysis scenarios and edge cases.
    """
    print("\n" + "=" * 60)
    print("🧪 ADVANCED ANALYSIS SCENARIOS")
    print("=" * 60)
    
    trojan_inserter = QuantumTrojanInserter(control_qubit_index=0)
    
    # Scenario 1: Different control qubit positions
    print("\n📍 Scenario 1: Impact of Control Qubit Position")
    print("-" * 45)
    
    # Create a 4-qubit test circuit
    test_circuit = QuantumCircuit(4, name="Control_Position_Test")
    test_circuit.h(1)
    test_circuit.cx(1, 2)
    test_circuit.ry(np.pi/4, 3)
    
    control_results = []
    for control_pos in range(4):
        inserter = QuantumTrojanInserter(control_qubit_index=control_pos)
        trojan_circuit, trojan_info = inserter.insert_controlled_trojan(
            test_circuit, gate_limit=3, activation_prob=0.5
        )
        
        # Simulate and calculate TVD
        original_counts = inserter.simulate_circuit(test_circuit, shots=1000)
        trojan_counts = inserter.simulate_circuit(trojan_circuit, shots=1000)
        tvd = inserter.calculate_tvd(original_counts, trojan_counts, 1000)
        
        control_results.append({
            'control_position': control_pos,
            'tvd': tvd,
            'trojans_inserted': len(trojan_info),
            'depth_change': trojan_circuit.depth() - test_circuit.depth()
        })
        
        print(f"   Control Qubit {control_pos}: TVD={tvd:.3f}, Trojans={len(trojan_info)}, Depth+{trojan_circuit.depth() - test_circuit.depth()}")
    
    # Scenario 2: Gate limit impact
    print("\n🎛️ Scenario 2: Impact of Gate Limit")
    print("-" * 35)
    
    gate_limit_results = []
    for gate_limit in [1, 3, 5, 7, 10]:
        trojan_circuit, trojan_info = trojan_inserter.insert_controlled_trojan(
            test_circuit, gate_limit=gate_limit, activation_prob=0.5
        )
        
        original_counts = trojan_inserter.simulate_circuit(test_circuit, shots=1000)
        trojan_counts = trojan_inserter.simulate_circuit(trojan_circuit, shots=1000)
        tvd = trojan_inserter.calculate_tvd(original_counts, trojan_counts, 1000)
        
        gate_limit_results.append({
            'gate_limit': gate_limit,
            'actual_gates': len(trojan_info),
            'tvd': tvd,
            'depth_change': trojan_circuit.depth() - test_circuit.depth()
        })
        
        print(f"   Limit={gate_limit}: Inserted={len(trojan_info)}, TVD={tvd:.3f}")
    
    # Scenario 3: Activation probability sweep
    print("\n🎲 Scenario 3: Activation Probability Analysis")
    print("-" * 40)
    
    activation_results = []
    for prob in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        total_tvd = 0
        runs = 5  # Multiple runs for statistical significance
        
        for _ in range(runs):
            trojan_circuit, trojan_info = trojan_inserter.insert_controlled_trojan(
                test_circuit, gate_limit=5, activation_prob=prob
            )
            
            original_counts = trojan_inserter.simulate_circuit(test_circuit, shots=500)
            trojan_counts = trojan_inserter.simulate_circuit(trojan_circuit, shots=500)
            tvd = trojan_inserter.calculate_tvd(original_counts, trojan_counts, 500)
            total_tvd += tvd
        
        avg_tvd = total_tvd / runs
        activation_results.append({
            'activation_prob': prob,
            'avg_tvd': avg_tvd
        })
        
        print(f"   Prob={prob:.1f}: Average TVD={avg_tvd:.3f}")
    
    # Summary of advanced analysis
    print(f"\n🔍 Advanced Analysis Summary:")
    print("=" * 35)
    
    best_control = max(control_results, key=lambda x: x['tvd'])
    print(f"   🎯 Most Effective Control Position: Qubit {best_control['control_position']} (TVD={best_control['tvd']:.3f})")
    
    optimal_gates = max(gate_limit_results, key=lambda x: x['tvd'])
    print(f"   ⚡ Optimal Gate Limit: {optimal_gates['gate_limit']} gates (TVD={optimal_gates['tvd']:.3f})")
    
    best_activation = max(activation_results, key=lambda x: x['avg_tvd'])
    print(f"   🎲 Most Impactful Activation Prob: {best_activation['activation_prob']:.1f} (TVD={best_activation['avg_tvd']:.3f})")

def security_recommendations():
    """
    Provide security recommendations based on the analysis.
    """
    print("\n" + "=" * 60)
    print("🛡️ SECURITY RECOMMENDATIONS")
    print("=" * 60)
    
    recommendations = [
        {
            "category": "Detection",
            "priority": "Critical",
            "items": [
                "Implement statistical output monitoring for TVD anomalies",
                "Deploy multi-compiler verification systems",
                "Use machine learning models trained on Trojan patterns",
                "Establish baseline circuit behavior profiles"
            ]
        },
        {
            "category": "Prevention",
            "priority": "High", 
            "items": [
                "Use only trusted, verified quantum compilers",
                "Implement compiler source code auditing",
                "Deploy circuit obfuscation techniques",
                "Establish secure compiler supply chains"
            ]
        },
        {
            "category": "Mitigation",
            "priority": "Medium",
            "items": [
                "Implement runtime circuit monitoring",
                "Use redundant compilation and verification",
                "Deploy circuit integrity checking",
                "Establish incident response procedures"
            ]
        },
        {
            "category": "Research",
            "priority": "Long-term",
            "items": [
                "Develop advanced Trojan detection algorithms",
                "Research quantum circuit forensics",
                "Create hardware-level verification methods",
                "Study quantum compiler security frameworks"
            ]
        }
    ]
    
    for rec in recommendations:
        print(f"\n🔹 {rec['category']} ({rec['priority']} Priority)")
        print("-" * (len(rec['category']) + len(rec['priority']) + 15))
        for i, item in enumerate(rec['items'], 1):
            print(f"   {i}. {item}")

if __name__ == "__main__":
    """
    Main execution flow for comprehensive quantum Trojan analysis.
    """
    print("🚀 Starting Comprehensive Quantum Trojan Analysis")
    print("=" * 60)
    
    try:
        # Run comprehensive analysis across multiple circuits
        results_df = comprehensive_trojan_analysis()
        
        # Detailed single circuit analysis
        single_analysis = demonstrate_single_circuit_analysis()
        
        # Advanced scenarios
        advanced_analysis_scenarios()
        
        # Security recommendations
        security_recommendations()
        
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 60)
        print("📊 Results have been analyzed and security implications assessed.")
        print("🔬 For interactive visualization, run: streamlit run streamlit_app.py")
        print("📚 Refer to the README.md for detailed usage instructions.")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        print("🔧 Please check your Qiskit installation and try again.")
        raise

    # Optional: Save results to file
    try:
        results_df.to_csv('trojan_analysis_results.csv', index=False)
        print(f"💾 Results saved to: trojan_analysis_results.csv")
    except:
        print("⚠️ Could not save results to file.")
    
    print(f"\n🔒 Remember: This analysis is for educational and research purposes only.")
    print(f"   Use responsibly and in accordance with quantum security research ethics.")