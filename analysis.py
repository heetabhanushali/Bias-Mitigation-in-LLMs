import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

def create_results_folder():
    """Create results folder if it doesn't exist"""
    if not os.path.exists('results'):
        os.makedirs('results')
    return 'results'

def load_and_prepare_data(baseline_path, mitigation_path):
    """Load and prepare baseline and mitigation data for analysis"""
    # Check if files are in results folder
    results_folder = create_results_folder()
    
    # Try to load from results folder first, then from current directory
    for path in [baseline_path, mitigation_path]:
        if not os.path.exists(path) and os.path.exists(os.path.join(results_folder, path)):
            if path == baseline_path:
                baseline_path = os.path.join(results_folder, path)
            else:
                mitigation_path = os.path.join(results_folder, path)
    
    baseline_df = pd.read_csv(baseline_path)
    mitigation_df = pd.read_csv(mitigation_path)
    
    # Add baseline indicator
    baseline_df['Mitigation'] = 'baseline'
    
    # Combine datasets
    combined_df = pd.concat([baseline_df, mitigation_df], ignore_index=True)
    
    return baseline_df, mitigation_df, combined_df

def calculate_improvement_metrics(baseline_df, mitigation_df):
    """Calculate improvement metrics for each mitigation strategy"""
    metrics = ['WEAT', 'Toxicity', 'CAT', 'Custom IBS']
    results = []
    
    # Get unique combinations for comparison
    combinations = baseline_df[['Dataset', 'Model']].drop_duplicates()
    
    for _, combo in combinations.iterrows():
        dataset = combo['Dataset']
        model = combo['Model']
        
        # Get baseline values
        baseline_row = baseline_df[
            (baseline_df['Dataset'] == dataset) & 
            (baseline_df['Model'] == model)
        ]
        
        if baseline_row.empty:
            continue
            
        baseline_values = baseline_row.iloc[0]
        
        # Get mitigation results for this combination
        mitigation_rows = mitigation_df[
            (mitigation_df['Dataset'] == dataset) & 
            (mitigation_df['Model'] == model)
        ]
        
        for _, mitigation_row in mitigation_rows.iterrows():
            strategy = mitigation_row['Mitigation']
            
            result = {
                'Dataset': dataset,
                'Model': model,
                'Strategy': strategy,
                'Model_Type': mitigation_row['Model_Type']
            }
            
            # Calculate improvements for each metric
            for metric in metrics:
                baseline_val = baseline_values[metric]
                mitigation_val = mitigation_row[metric]
                
                # Handle NaN values
                if pd.isna(baseline_val) or pd.isna(mitigation_val):
                    result[f'{metric}_Baseline'] = baseline_val
                    result[f'{metric}_Mitigation'] = mitigation_val
                    result[f'{metric}_Absolute_Change'] = np.nan
                    result[f'{metric}_Percent_Change'] = np.nan
                    result[f'{metric}_Improvement'] = np.nan
                else:
                    absolute_change = mitigation_val - baseline_val
                    percent_change = (absolute_change / abs(baseline_val)) * 100 if baseline_val != 0 else np.nan
                    
                    # For bias metrics, lower is better (negative change is improvement)
                    # For toxicity, lower is better (negative change is improvement)
                    improvement = -absolute_change if metric in ['WEAT', 'Toxicity', 'CAT', 'Custom IBS'] else absolute_change
                    
                    result[f'{metric}_Baseline'] = baseline_val
                    result[f'{metric}_Mitigation'] = mitigation_val
                    result[f'{metric}_Absolute_Change'] = absolute_change
                    result[f'{metric}_Percent_Change'] = percent_change
                    result[f'{metric}_Improvement'] = improvement
            
            results.append(result)
    
    return pd.DataFrame(results)

def create_improvement_summary(improvement_df):
    """Create summary statistics for improvements"""
    metrics = ['WEAT', 'Toxicity', 'CAT', 'Custom IBS']
    strategies = improvement_df['Strategy'].unique()
    
    summary_data = []
    
    for strategy in strategies:
        strategy_data = improvement_df[improvement_df['Strategy'] == strategy]
        
        for metric in metrics:
            improvement_col = f'{metric}_Improvement'
            if improvement_col in strategy_data.columns:
                improvements = strategy_data[improvement_col].dropna()
                
                if len(improvements) > 0:
                    summary_data.append({
                        'Strategy': strategy,
                        'Metric': metric,
                        'Mean_Improvement': improvements.mean(),
                        'Median_Improvement': improvements.median(),
                        'Std_Improvement': improvements.std(),
                        'Min_Improvement': improvements.min(),
                        'Max_Improvement': improvements.max(),
                        'Count': len(improvements),
                        'Positive_Improvements': (improvements > 0).sum(),
                        'Negative_Improvements': (improvements < 0).sum(),
                        'Success_Rate': (improvements > 0).mean() * 100
                    })
    
    return pd.DataFrame(summary_data)

def plot_comparative_analysis(combined_df, improvement_df, results_folder, save_plots=True):
    """Create comprehensive plots for comparative analysis"""
    metrics = ['WEAT', 'Toxicity', 'CAT', 'Custom IBS']
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Heatmap of improvements by strategy and metric
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        # Create pivot table for heatmap
        improvement_col = f'{metric}_Improvement'
        if improvement_col in improvement_df.columns:
            pivot_data = improvement_df.pivot_table(
                values=improvement_col, 
                index='Strategy', 
                columns='Dataset', 
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_data, annot=True, fmt='.6f', cmap='RdYlGn', 
                       center=0, ax=axes[i], cbar_kws={'label': 'Improvement'})
            axes[i].set_title(f'{metric} Improvement by Strategy and Dataset')
            axes[i].set_xlabel('Dataset')
            axes[i].set_ylabel('Strategy')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(results_folder, 'improvement_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Box plots comparing baseline vs mitigation for each metric
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        # Prepare data for box plot
        plot_data = []
        labels = []
        
        # Add baseline data
        baseline_vals = combined_df[combined_df['Mitigation'] == 'baseline'][metric].dropna()
        plot_data.append(baseline_vals)
        labels.append('Baseline')
        
        # Add mitigation data for each strategy
        strategies = combined_df[combined_df['Mitigation'] != 'baseline']['Mitigation'].unique()
        for strategy in strategies:
            strategy_vals = combined_df[combined_df['Mitigation'] == strategy][metric].dropna()
            if len(strategy_vals) > 0:
                plot_data.append(strategy_vals)
                labels.append(strategy)
        
        # Create box plot
        axes[i].boxplot(plot_data, labels=labels)
        axes[i].set_title(f'{metric} Distribution: Baseline vs Mitigation Strategies')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(results_folder, 'baseline_vs_mitigation_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Strategy effectiveness comparison
    if not improvement_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            improvement_col = f'{metric}_Improvement'
            if improvement_col in improvement_df.columns:
                # Group by strategy and calculate mean improvement
                strategy_means = improvement_df.groupby('Strategy')[improvement_col].agg(['mean', 'std', 'count']).reset_index()
                strategy_means = strategy_means.dropna()
                
                if not strategy_means.empty:
                    bars = axes[i].bar(strategy_means['Strategy'], strategy_means['mean'], 
                                     yerr=strategy_means['std'], capsize=5, alpha=0.7)
                    axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.5)
                    axes[i].set_title(f'Average {metric} Improvement by Strategy')
                    axes[i].set_ylabel(f'{metric} Improvement')
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for bar, val in zip(bars, strategy_means['mean']):
                        height = bar.get_height()
                        axes[i].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{val:.6f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(results_folder, 'strategy_effectiveness.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 4. Violin plots for distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        # Create violin plot
        sns.violinplot(data=combined_df, x='Mitigation', y=metric, ax=axes[i])
        axes[i].set_title(f'{metric} Distribution by Mitigation Strategy (Violin Plot)')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(results_folder, 'violin_plots.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Scatter plots showing correlations between metrics
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()
    
    metric_pairs = [
        ('WEAT', 'Toxicity'),
        ('WEAT', 'CAT'),
        ('WEAT', 'Custom IBS'),
        ('Toxicity', 'CAT'),
        ('Toxicity', 'Custom IBS'),
        ('CAT', 'Custom IBS')
    ]
    
    for i, (metric1, metric2) in enumerate(metric_pairs):
        # Create scatter plot
        for mitigation in combined_df['Mitigation'].unique():
            subset = combined_df[combined_df['Mitigation'] == mitigation]
            axes[i].scatter(subset[metric1], subset[metric2], label=mitigation, alpha=0.7)
        
        axes[i].set_xlabel(metric1)
        axes[i].set_ylabel(metric2)
        axes[i].set_title(f'{metric1} vs {metric2} Correlation')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(results_folder, 'correlation_scatterplots.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Ridge plots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        # Create ridge-like plot using multiple histograms
        strategies = combined_df['Mitigation'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
        
        for j, strategy in enumerate(strategies):
            data = combined_df[combined_df['Mitigation'] == strategy][metric].dropna()
            if len(data) > 0:
                axes[i].hist(data, alpha=0.6, label=strategy, color=colors[j], bins=15)
        
        axes[i].set_xlabel(metric)
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{metric} Distribution by Strategy (Ridge Plot)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(results_folder, 'ridge_plots.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Radar chart for strategy comparison
    if not improvement_df.empty:
        strategies = improvement_df['Strategy'].unique()
        
        # Calculate mean improvements for each strategy
        strategy_performance = {}
        for strategy in strategies:
            strategy_data = improvement_df[improvement_df['Strategy'] == strategy]
            performance = []
            for metric in metrics:
                improvement_col = f'{metric}_Improvement'
                if improvement_col in strategy_data.columns:
                    mean_imp = strategy_data[improvement_col].mean()
                    performance.append(mean_imp if not pd.isna(mean_imp) else 0)
                else:
                    performance.append(0)
            strategy_performance[strategy] = performance
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))
        
        for i, (strategy, performance) in enumerate(strategy_performance.items()):
            performance += performance[:1]  # Complete the circle
            ax.plot(angles, performance, 'o-', linewidth=2, label=strategy, color=colors[i])
            ax.fill(angles, performance, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title('Strategy Performance Radar Chart', size=16, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(results_folder, 'radar_chart.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 8. Improvement magnitude comparison
    if not improvement_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            improvement_col = f'{metric}_Improvement'
            if improvement_col in improvement_df.columns:
                # Create strip plot
                sns.stripplot(data=improvement_df, x='Strategy', y=improvement_col, 
                             size=8, alpha=0.7, ax=axes[i])
                axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.5)
                axes[i].set_title(f'{metric} Improvement Distribution (Strip Plot)')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(results_folder, 'improvement_stripplots.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 9. Dataset-specific performance
    datasets = combined_df['Dataset'].unique()
    if len(datasets) > 1:
        fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(20, 5*len(datasets)))
        if len(datasets) == 1:
            axes = axes.reshape(1, -1)
        
        for i, dataset in enumerate(datasets):
            for j, metric in enumerate(metrics):
                dataset_data = combined_df[combined_df['Dataset'] == dataset]
                sns.boxplot(data=dataset_data, x='Mitigation', y=metric, ax=axes[i, j])
                axes[i, j].set_title(f'{metric} - {dataset}')
                axes[i, j].tick_params(axis='x', rotation=45)
                axes[i, j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(results_folder, 'dataset_specific_performance.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 10. Model-specific performance
    models = combined_df['Model'].unique()
    if len(models) > 1:
        fig, axes = plt.subplots(len(models), len(metrics), figsize=(20, 5*len(models)))
        if len(models) == 1:
            axes = axes.reshape(1, -1)
        
        for i, model in enumerate(models):
            for j, metric in enumerate(metrics):
                model_data = combined_df[combined_df['Model'] == model]
                sns.boxplot(data=model_data, x='Mitigation', y=metric, ax=axes[i, j])
                axes[i, j].set_title(f'{metric} - {model}')
                axes[i, j].tick_params(axis='x', rotation=45)
                axes[i, j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(results_folder, 'model_specific_performance.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 11. Success rate visualization
    if not improvement_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            improvement_col = f'{metric}_Improvement'
            if improvement_col in improvement_df.columns:
                # Calculate success rates
                success_rates = []
                strategies = improvement_df['Strategy'].unique()
                
                for strategy in strategies:
                    strategy_data = improvement_df[improvement_df['Strategy'] == strategy]
                    improvements = strategy_data[improvement_col].dropna()
                    if len(improvements) > 0:
                        success_rate = (improvements > 0).mean() * 100
                        success_rates.append(success_rate)
                    else:
                        success_rates.append(0)
                
                bars = axes[i].bar(strategies, success_rates, alpha=0.7)
                axes[i].set_title(f'{metric} Success Rate by Strategy')
                axes[i].set_ylabel('Success Rate (%)')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
                
                # Add value labels
                for bar, rate in zip(bars, success_rates):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                               f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(results_folder, 'success_rates.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 12. Pairwise comparison matrix
    if not improvement_df.empty:
        strategies = improvement_df['Strategy'].unique()
        if len(strategies) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            axes = axes.flatten()
            
            for i, metric in enumerate(metrics):
                improvement_col = f'{metric}_Improvement'
                if improvement_col in improvement_df.columns:
                    # Create pairwise comparison matrix
                    comparison_matrix = np.zeros((len(strategies), len(strategies)))
                    
                    for j, strategy1 in enumerate(strategies):
                        for k, strategy2 in enumerate(strategies):
                            if j != k:
                                data1 = improvement_df[improvement_df['Strategy'] == strategy1][improvement_col].dropna()
                                data2 = improvement_df[improvement_df['Strategy'] == strategy2][improvement_col].dropna()
                                
                                if len(data1) > 0 and len(data2) > 0:
                                    # Calculate how often strategy1 outperforms strategy2
                                    comparison_matrix[j, k] = (data1.mean() > data2.mean()) * 100
                    
                    sns.heatmap(comparison_matrix, annot=True, fmt='.0f', 
                               xticklabels=strategies, yticklabels=strategies,
                               cmap='RdYlBu', ax=axes[i], cbar_kws={'label': '% Better'})
                    axes[i].set_title(f'{metric} Pairwise Strategy Comparison')
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(os.path.join(results_folder, 'pairwise_comparison.png'), dpi=300, bbox_inches='tight')
            plt.show()
    
    # 13. Time series style plot (if applicable)
    if not improvement_df.empty:
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Create a pseudo time series by strategy order
        strategy_order = list(improvement_df['Strategy'].unique())
        
        for metric in metrics:
            improvement_col = f'{metric}_Improvement'
            if improvement_col in improvement_df.columns:
                metric_means = []
                for strategy in strategy_order:
                    strategy_data = improvement_df[improvement_df['Strategy'] == strategy]
                    mean_imp = strategy_data[improvement_col].mean()
                    metric_means.append(mean_imp if not pd.isna(mean_imp) else 0)
                
                ax.plot(range(len(strategy_order)), metric_means, marker='o', 
                       linewidth=2, markersize=8, label=metric)
        
        ax.set_xticks(range(len(strategy_order)))
        ax.set_xticklabels(strategy_order, rotation=45)
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Mean Improvement')
        ax.set_title('Strategy Performance Trend')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(results_folder, 'strategy_trend.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 14. Statistical significance testing visualization
    if not improvement_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            improvement_col = f'{metric}_Improvement'
            if improvement_col in improvement_df.columns:
                strategies = improvement_df['Strategy'].unique()
                p_values = []
                strategy_pairs = []
                
                # Perform pairwise t-tests
                from itertools import combinations
                for strategy1, strategy2 in combinations(strategies, 2):
                    data1 = improvement_df[improvement_df['Strategy'] == strategy1][improvement_col].dropna()
                    data2 = improvement_df[improvement_df['Strategy'] == strategy2][improvement_col].dropna()
                    
                    if len(data1) > 1 and len(data2) > 1:
                        try:
                            t_stat, p_val = stats.ttest_ind(data1, data2)
                            p_values.append(p_val)
                            strategy_pairs.append(f"{strategy1} vs {strategy2}")
                        except:
                            p_values.append(1.0)
                            strategy_pairs.append(f"{strategy1} vs {strategy2}")
                
                if p_values:
                    colors = ['red' if p < 0.05 else 'blue' for p in p_values]
                    bars = axes[i].bar(range(len(p_values)), p_values, color=colors, alpha=0.7)
                    axes[i].axhline(y=0.05, color='red', linestyle='--', alpha=0.8, label='p=0.05')
                    axes[i].set_title(f'{metric} Statistical Significance (t-tests)')
                    axes[i].set_ylabel('p-value')
                    axes[i].set_xticks(range(len(strategy_pairs)))
                    axes[i].set_xticklabels(strategy_pairs, rotation=45, ha='right')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(results_folder, 'statistical_significance.png'), dpi=300, bbox_inches='tight')
        plt.show()

def save_detailed_report(baseline_df, mitigation_df, improvement_df, summary_df, results_folder):
    """Generate and save a detailed text report of the analysis"""
    
    report_content = []
    report_content.append("="*80)
    report_content.append("COMPREHENSIVE BIAS MITIGATION ANALYSIS REPORT")
    report_content.append("="*80)
    
    report_content.append(f"\nDATA OVERVIEW:")
    report_content.append(f"- Baseline results: {len(baseline_df)} entries")
    report_content.append(f"- Mitigation results: {len(mitigation_df)} entries")
    report_content.append(f"- Datasets evaluated: {', '.join(baseline_df['Dataset'].unique())}")
    report_content.append(f"- Models evaluated: {', '.join(baseline_df['Model'].unique())}")
    report_content.append(f"- Mitigation strategies: {', '.join(mitigation_df['Mitigation'].unique())}")
    
    report_content.append(f"\nBASELINE PERFORMANCE SUMMARY:")
    report_content.append("-"*50)
    metrics = ['WEAT', 'Toxicity', 'CAT', 'Custom IBS']
    for metric in metrics:
        values = baseline_df[metric].dropna()
        if len(values) > 0:
            report_content.append(f"{metric:15} | Mean: {values.mean():.6f} | Std: {values.std():.6f} | Range: [{values.min():.6f}, {values.max():.6f}]")
    
    if not improvement_df.empty:
        report_content.append(f"\nMITIGATION EFFECTIVENESS SUMMARY:")
        report_content.append("-"*50)
        
        for strategy in improvement_df['Strategy'].unique():
            report_content.append(f"\n{strategy.upper()} STRATEGY:")
            strategy_data = improvement_df[improvement_df['Strategy'] == strategy]
            
            for metric in metrics:
                improvement_col = f'{metric}_Improvement'
                if improvement_col in strategy_data.columns:
                    improvements = strategy_data[improvement_col].dropna()
                    if len(improvements) > 0:
                        positive_count = (improvements > 0).sum()
                        total_count = len(improvements)
                        success_rate = (positive_count / total_count) * 100
                        report_content.append(f"  {metric:15} | Avg Improvement: {improvements.mean():.6f} | Success Rate: {success_rate:.1f}% ({positive_count}/{total_count})")
    
    report_content.append(f"\nSTRATEGY RANKING BY OVERALL EFFECTIVENESS:")
    report_content.append("-"*50)
    
    if not summary_df.empty:
        # Calculate overall effectiveness score for each strategy
        strategy_scores = summary_df.groupby('Strategy').agg({
            'Mean_Improvement': 'mean',
            'Success_Rate': 'mean'
        }).round(6)
        strategy_scores['Overall_Score'] = strategy_scores['Mean_Improvement'] * (strategy_scores['Success_Rate'] / 100)
        strategy_scores = strategy_scores.sort_values('Overall_Score', ascending=False)
        
        for i, (strategy, scores) in enumerate(strategy_scores.iterrows(), 1):
            report_content.append(f"{i}. {strategy:20} | Overall Score: {scores['Overall_Score']:.6f} | Avg Improvement: {scores['Mean_Improvement']:.6f} | Success Rate: {scores['Success_Rate']:.1f}%")
    
    report_content.append(f"\nKEY INSIGHTS:")
    report_content.append("-"*50)
    
    # Generate insights based on the data
    if not improvement_df.empty:
        # Best performing strategy overall
        best_strategy = summary_df.groupby('Strategy')['Mean_Improvement'].mean().idxmax()
        report_content.append(f"• Best overall strategy: {best_strategy}")
        
        # Most consistent strategy
        consistency_scores = summary_df.groupby('Strategy')['Success_Rate'].mean()
        most_consistent = consistency_scores.idxmax()
        report_content.append(f"• Most consistent strategy: {most_consistent} ({consistency_scores[most_consistent]:.1f}% success rate)")
        
        # Metric-specific insights
        for metric in metrics:
            metric_data = summary_df[summary_df['Metric'] == metric]
            if not metric_data.empty:
                best_for_metric = metric_data.loc[metric_data['Mean_Improvement'].idxmax(), 'Strategy']
                report_content.append(f"• Best strategy for {metric}: {best_for_metric}")
    
    report_content.append("\n" + "="*80)
    
    # Save the report to file
    report_text = "\n".join(report_content)
    with open(os.path.join(results_folder, 'analysis_report.txt'), 'w') as f:
        f.write(report_text)
    
    # Also print to console
    print(report_text)

def main_analysis(baseline_path, mitigation_path, save_results=True):
    """Main function to run the complete comparative analysis"""
    
    # Create results folder
    results_folder = create_results_folder()
    
    # Load and prepare data
    print("Loading data...")
    baseline_df, mitigation_df, combined_df = load_and_prepare_data(baseline_path, mitigation_path)
    
    # Calculate improvements
    print("Calculating improvement metrics...")
    improvement_df = calculate_improvement_metrics(baseline_df, mitigation_df)
    
    # Create summary statistics
    print("Generating summary statistics...")
    summary_df = create_improvement_summary(improvement_df)
    
    # Generate plots
    print("Creating visualizations...")
    plot_comparative_analysis(combined_df, improvement_df, results_folder, save_plots=save_results)
    
    # Generate and save detailed report
    print("Generating detailed report...")
    save_detailed_report(baseline_df, mitigation_df, improvement_df, summary_df, results_folder)
    
    # Save results if requested
    if save_results:
        print("Saving results...")
        improvement_df.to_csv(os.path.join(results_folder, 'detailed_improvement_analysis.csv'), index=False)
        summary_df.to_csv(os.path.join(results_folder, 'mitigation_summary_statistics.csv'), index=False)
        combined_df.to_csv(os.path.join(results_folder, 'combined_dataset.csv'), index=False)
        print(f"Results saved to '{results_folder}' folder.")
    
    return {
        'baseline_df': baseline_df,
        'mitigation_df': mitigation_df,
        'combined_df': combined_df,
        'improvement_df': improvement_df,
        'summary_df': summary_df
    }

# Example usage
if __name__ == "__main__":
    # Update these paths to match your file locations
    # The script will automatically check both current directory and results folder
    baseline_path = "report/bias_evaluation_results_20250710_193808.csv"
    mitigation_path = "report/mitigation_results_20250710_192842.csv"
    
    # Run the complete analysis
    results = main_analysis(baseline_path, mitigation_path, save_results=True)
    
    # Access individual components if needed
    # baseline_df = results['baseline_df']
    # mitigation_df = results['mitigation_df']
    # improvement_df = results['improvement_df']
    # summary_df = results['summary_df']
    
    print("\nAll analysis complete! Check the 'results' folder for:")
    print("CSV Files:")
    print("- detailed_improvement_analysis.csv")
    print("- mitigation_summary_statistics.csv")
    print("- combined_dataset.csv")
    print("- analysis_report.txt")
    print("\nVisualization Files:")
    print("- improvement_heatmaps.png")
    print("- baseline_vs_mitigation_boxplots.png")
    print("- strategy_effectiveness.png")
    print("- violin_plots.png")
    print("- correlation_scatterplots.png")
    print("- ridge_plots.png")
    print("- radar_chart.png")
    print("- improvement_stripplots.png")
    print("- dataset_specific_performance.png")
    print("- model_specific_performance.png")
    print("- success_rates.png")
    print("- pairwise_comparison.png")
    print("- strategy_trend.png")
    print("- statistical_significance.png")