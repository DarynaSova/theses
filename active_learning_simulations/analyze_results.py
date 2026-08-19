import json
import pandas as pd

# Load all three result files
baseline = json.load(open('simulation_v1_results/compressed_dashboard_data_baseline_no_disorder.json'))
mean_disorder = json.load(open('simulation_v1_results/compressed_dashboard_data_with_disorder_mean_weight_1.0.json'))
frac_above_5 = json.load(open('simulation_v1_results/compressed_dashboard_data_with_disorder_frac_above_5_weight_1.0.json'))

# Build comparison data
data = []
baseline_map = {exp['name']: exp for exp in baseline['experiments']}
mean_map = {exp['name']: exp for exp in mean_disorder['experiments']}
frac_map = {exp['name']: exp for exp in frac_above_5['experiments']}

dataset_names = {
    1: "MELTOME_MAX",
    2: "MELTOME_MIN",
    3: "SCL",
    4: "AMYLASE",
    5: "PHOT",
    6: "EXOTOX"
}

for exp_name in sorted(baseline_map.keys()):
    b_exp = baseline_map.get(exp_name)
    m_exp = mean_map.get(exp_name)
    f_exp = frac_map.get(exp_name)
    
    if not b_exp:
        continue
    
    dataset_name = dataset_names.get(b_exp['dataset_id'], str(b_exp['dataset_id']))
    model = b_exp['model']
    
    # Extract metrics
    b_hits = b_exp['aggregated_hits']
    b_success_pct = (sum(b_exp['per_sim_is_success']) * 100 // len(b_exp['per_sim_is_success'])) if b_exp['per_sim_is_success'] else 0
    
    m_hits = m_exp['aggregated_hits'] if m_exp else None
    m_success_pct = (sum(m_exp['per_sim_is_success']) * 100 // len(m_exp['per_sim_is_success'])) if m_exp and m_exp['per_sim_is_success'] else 0
    
    f_hits = f_exp['aggregated_hits'] if f_exp else None
    f_success_pct = (sum(f_exp['per_sim_is_success']) * 100 // len(f_exp['per_sim_is_success'])) if f_exp and f_exp['per_sim_is_success'] else 0
    
    # Calculate deltas
    m_hits_delta = ((m_hits - b_hits) / b_hits * 100) if (m_hits and b_hits) else None
    f_hits_delta = ((f_hits - b_hits) / b_hits * 100) if (f_hits and b_hits) else None
    
    data.append({
        'Dataset': dataset_name,
        'Model': model,
        'Baseline Hits': b_hits,
        'Baseline Success %': b_success_pct,
        'Mean Disorder Hits': m_hits,
        'Mean Success %': m_success_pct,
        'Mean Δ Hits %': f"{m_hits_delta:.1f}%" if m_hits_delta is not None else "N/A",
        'Frac>5 Hits': f_hits,
        'Frac>5 Success %': f_success_pct,
        'Frac>5 Δ Hits %': f"{f_hits_delta:.1f}%" if f_hits_delta is not None else "N/A",
    })

df = pd.DataFrame(data)
print("\n" + "="*180)
print("ACTIVE LEARNING SIMULATION RESULTS COMPARISON")
print("="*180)
print(df.to_string(index=False))
print("="*180)

print("\nSUMMARY STATISTICS")
print("-"*180)
print(f"Total experiments: {len(df)}")
print(f"\nBaseline Total Hits: {df['Baseline Hits'].sum()}")
print(f"Mean Disorder Total Hits: {df['Mean Disorder Hits'].sum()}")
print(f"Frac>5 Total Hits: {df['Frac>5 Hits'].sum()}")
print(f"\nAvg Baseline Success Rate: {df['Baseline Success %'].mean():.1f}%")
print(f"Avg Mean Disorder Success Rate: {df['Mean Success %'].mean():.1f}%")
print(f"Avg Frac>5 Success Rate: {df['Frac>5 Success %'].mean():.1f}%")
