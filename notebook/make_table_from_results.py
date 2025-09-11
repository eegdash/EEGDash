import json
import numpy as np
import pandas as pd
from pathlib import Path
import os

# Define tasks and factors from tutorial_pfactor_classification.py
tasks = [
    'DespicableMe',
    'DiaryOfAWimpyKid', 
    'FunwithFractals',
    'RestingState',
    'ThePresent',
    'contrastChangeDetection',
    'seqLearning6target',
    'seqLearning8target',
    'surroundSupp',
    'symbolSearch'
]

factors = ["sex", "p_factor", "attention", "internalizing", "externalizing"]

# Results directory
results_dir = Path("results")

def load_and_analyze_results():
    """Load JSON files and calculate statistics for each task-factor combination."""
    
    # Initialize results matrix
    results_matrix = {}
    ci_matrix = {}
    significant_matrix = {}
    
    for task in tasks:
        results_matrix[task] = {}
        ci_matrix[task] = {}
        significant_matrix[task] = {}
        
        for factor in factors:
            json_file = results_dir / f"{task}_{factor}.json"
            
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract test performance (10 runs)
                test_scores = data['test']
                
                # Calculate statistics
                mean_score = np.mean(test_scores)
                std_error = np.std(test_scores, ddof=1) / np.sqrt(len(test_scores))
                ci_lower = mean_score - 1.96 * std_error
                ci_upper = mean_score + 1.96 * std_error
                
                # Check if 95% CI excludes 50% (0.5)
                significant = not (ci_lower <= 0.5 <= ci_upper)
                
                results_matrix[task][factor] = mean_score
                ci_matrix[task][factor] = (ci_lower, ci_upper)
                significant_matrix[task][factor] = significant
                
                print(f"{task}_{factor}: Mean={mean_score:.3f}, CI=[{ci_lower:.3f}, {ci_upper:.3f}], Significant={significant}")
            else:
                # Handle missing files
                results_matrix[task][factor] = None
                ci_matrix[task][factor] = None
                significant_matrix[task][factor] = False
                print(f"Missing file: {json_file}")
    
    return results_matrix, ci_matrix, significant_matrix

def create_latex_table(results_matrix, ci_matrix, significant_matrix):
    """Create a LaTeX table with colored cells for significant results."""
    
    latex_content = r"""
\documentclass{article}
\usepackage[table]{xcolor}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{graphicx}
\geometry{a4paper, margin=1cm}

\begin{document}

\begin{table}[h]
\centering
\caption{Classification Performance Across Tasks and Psychological Factors}
\label{tab:results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l|ccccc}
\toprule
\textbf{Task} & \textbf{Sex} & \textbf{P-Factor} & \textbf{Attention} & \textbf{Internalizing} & \textbf{Externalizing} \\
\midrule
"""
    
    for task in tasks:
        task_name = task.replace('_', '\\_')
        latex_content += task_name
        
        for factor in factors:
            if results_matrix[task][factor] is not None:
                mean_score = results_matrix[task][factor]
                ci_lower, ci_upper = ci_matrix[task][factor]
                significant = significant_matrix[task][factor]
                
                # Format the cell content
                cell_content = f"{mean_score:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"
                
                # Add color if significant
                if significant:
                    latex_content += f" & \\cellcolor{{yellow!50}} {cell_content}"
                else:
                    latex_content += f" & {cell_content}"
            else:
                latex_content += " & N/A"
        
        latex_content += " \\\\\n"
    
    latex_content += r"""
\bottomrule
\end{tabular}%
}
\end{table}

\vspace{1cm}
\textbf{Note:} Cells highlighted in yellow indicate 95\% confidence intervals that do not include 50\% performance (chance level). Values shown as mean [lower CI, upper CI].

\end{document}
"""
    
    return latex_content

def main():
    """Main function to process results and generate LaTeX table."""
    
    print("Loading and analyzing results...")
    results_matrix, ci_matrix, significant_matrix = load_and_analyze_results()
    
    # Create and save pandas dataframe as PNG
    df = pd.DataFrame(results_matrix).T
    df.to_csv("results_matrix.csv")
    print("Results matrix saved to results_matrix.csv")
    
    print("\nGenerating LaTeX table...")
    latex_content = create_latex_table(results_matrix, ci_matrix, significant_matrix)
    
    # Write LaTeX file
    with open("results_table.tex", "w") as f:
        f.write(latex_content)
    
    print("LaTeX file written to results_table.tex")
    
    # Compile to PDF
    print("Compiling LaTeX to PDF...")
    os.system("pdflatex results_table.tex")
    
    # Convert PDF to PNG using Ghostscript (more reliable than ImageMagick)
    print("Converting PDF to PNG using Ghostscript...")
    gs_cmd = "gs -dNOPAUSE -dBATCH -sDEVICE=png16m -r300 -sOutputFile=results_table.png results_table.pdf"
    gs_result = os.system(gs_cmd)
    
    if gs_result == 0:
        print("PDF converted to PNG successfully: results_table.png")
    else:
        print("Error converting PDF to PNG. Check if Ghostscript is properly installed.")
    
    print("Done! Check results_table.pdf for the output.")

if __name__ == "__main__":
    main()