import json
from re import T
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys

plot_what = "tasks_factors"
# plot_what = "releases_r12"
# plot_what = "releases_releases"

# Results directory
if plot_what == "tasks_factors":
    results_dir = Path("results_R5")
    results_dir = Path("results_tasks_factor_allRs")
    results_dir = Path("results")
    title = "Classification Performance Across Tasks and Psychological Factors"
    ignore_diagonal = False
elif plot_what == "releases_r12":
    results_dir = Path("results_R12")
    title = "Classification Performance Across Releases and Test Sets"
    ignore_diagonal = False
elif plot_what == "releases_releases":
    factor_name = "p_factor"
    # factor_name = "sex"
    results_dir = Path(f"results_{factor_name}_contrast")
    title = f"{factor_name} classification Performance (Contrast Change Detection) Across Releases"
    ignore_diagonal = True
    
# check if folder exists
if not os.path.exists(results_dir):
    print(f"Results directory {results_dir} does not exist")
    sys.exit()

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
tasks = [
    'movies',
    'restingstate',
    'contrastChangeDetection',
    'seqLearning',
    'surroundSupp',
    'symbolSearch',
    'all_tasks'
]
factors = ["sex", "age", "p_factor", "attention", "internalizing", "externalizing"]

releases = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"]
releases_train = ["R1train", "R2train", "R3train", "R4train", "R5train", "R6train", "R7train", "R8train", "R9train", "R10train", "R11train", "R12train"]
releases_test  = ["R1test", "R2test", "R3test", "R4test", "R5test", "R6test", "R7test", "R8test", "R9test", "R10test", "R11test", "R12test"]
testset = ["Internal_test_set", "R12_test_set"]
model_name = 'EEGNeX'
# model_name = 'TSception'
# model_name = 'EEGConform'

def load_and_analyze_results_tasks_factors():
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
            json_file = results_dir / f"{task}_{factor}_{model_name}.json"
            
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                
                if 1:
                # Calculate statistics
                    results_matrix[task][factor] = data['test'][0]
                    ci_matrix[task][factor] = data['test_ci'] 
                    significant_matrix[task][factor] = not (data['test_ci'][0] <= 0.5 <= data['test_ci'][1])
                    print(f"{task}_{factor}: Mean={data['test'][0]:.3f}, CI=[{data['test_ci'][0]:.3f}, {data['test_ci'][1]:.3f}], Significant={significant_matrix[task][factor]}")
                else:
                    # Extract test performance (10 runs)
                    test_scores = data['test']
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

def load_and_analyze_results_releases_releases():
    """Load JSON files and calculate statistics for each task-factor combination."""
    
    # Initialize results matrix
    results_matrix = {}
    ci_matrix = {}
    significant_matrix = {}
    
    for release1 in releases_train:
        results_matrix[release1] = {}
        ci_matrix[release1] = {}
        significant_matrix[release1] = {}
        
        for release2 in releases_test:
            json_file = results_dir / f"{release1[:-5]}train_{release2[:-4]}test_contrastChangeDetection_{factor_name}.json"
            
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
                
                results_matrix[release1][release2] = mean_score
                ci_matrix[release1][release2] = (ci_lower, ci_upper)
                significant_matrix[release1][release2] = significant
                
                print(f"{release1}train_{release2}test_contrastChangeDetection_p_factor: Mean={mean_score:.3f}, CI=[{ci_lower:.3f}, {ci_upper:.3f}], Significant={significant}")
            else:
                # Handle missing files
                results_matrix[release1][release2] = None
                ci_matrix[release1][release2] = None
                significant_matrix[release1][release2] = False
                print(f"Missing file: {json_file}")
    
    return results_matrix, ci_matrix, significant_matrix

def load_and_analyze_results_releases_r12():
    """Load JSON files and calculate statistics for each release combination."""
    
    # Initialize results matrix
    results_matrix = {}
    ci_matrix = {}
    significant_matrix = {}
    task = "contrastChangeDetection"
    factor = "p_factor"
    
    for release in releases:
        results_matrix[release] = {}
        ci_matrix[release] = {}
        significant_matrix[release] = {}
        
        json_file = results_dir / f"{release}_{task}_{factor}.json"
        
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract test performance (10 runs)
            test_scores = data['test']
            test_scores_r12 = data['test_r12']
            
            # Calculate statistics
            mean_score = np.mean(test_scores)
            std_error = np.std(test_scores, ddof=1) / np.sqrt(len(test_scores))
            ci_lower = mean_score - 1.96 * std_error
            ci_upper = mean_score + 1.96 * std_error
            significant = not (ci_lower <= 0.5 <= ci_upper)
            
            mean_score_r12 = np.mean(test_scores_r12)
            std_error_r12 = np.std(test_scores_r12, ddof=1) / np.sqrt(len(test_scores_r12))
            ci_lower_r12 = mean_score_r12 - 1.96 * std_error_r12
            ci_upper_r12 = mean_score_r12 + 1.96 * std_error_r12
            significant_r12 = not (ci_lower_r12 <= 0.5 <= ci_upper_r12)
                        
            results_matrix[release][testset[0]] = mean_score
            ci_matrix[release][testset[0]] = (ci_lower, ci_upper)
            significant_matrix[release][testset[0]] = significant
            results_matrix[release][testset[1]] = mean_score_r12
            ci_matrix[release][testset[1]] = (ci_lower_r12, ci_upper_r12)
            significant_matrix[release][testset[1]] = significant_r12
            
            print(f"{release}_{task}_{factor}: Mean={mean_score:.3f}, CI=[{ci_lower:.3f}, {ci_upper:.3f}], Significant={significant}")
        else:
            # Handle missing files
            results_matrix[release][testset[0]] = None
            ci_matrix[release][testset[0]] = None
            significant_matrix[release][testset[0]] = False
            results_matrix[release][testset[1]] = None
            ci_matrix[release][testset[1]] = None
            significant_matrix[release][testset[1]] = False
            print(f"Missing file: {json_file}")
    
    return results_matrix, ci_matrix, significant_matrix

def create_latex_table(results_matrix, ci_matrix, significant_matrix, vars1, vars2):
    """Create a LaTeX table with colored cells for significant results."""
    
    # Determine table format based on plot_what
    if plot_what == "tasks_factors":
        # 1 row label + 6 factors = 7 columns
        table_format = "l|cccccc"
        header = r"""
        \textbf{{Task}} & \textbf{{Sex}} & \textbf{{Age}} & \textbf{{P-Factor}} & \textbf{{Attention}} & \textbf{{Internalizing}} & \textbf{{Externalizing}} \\
        \midrule
        """
    elif plot_what == "releases_r12":
        # 1 row label + 2 test sets = 3 columns
        table_format = "l|cc"
        header = r"""
        \textbf{{Release}} & \textbf{{Internal test set}} & \textbf{{R12 test set}} \\
        \midrule
        """
    elif plot_what == "releases_releases":
        # 1 row label + 11 releases = 12 columns
        table_format = "l|" + "c" * len(vars2)
        header = r"""
        \textbf{{Train Release}} & """ + " & ".join([f"\\textbf{{{var}}}" for var in vars2]) + r""" \\
        \midrule
        """
    
    latex_content = f"""
\\documentclass{{article}}
\\usepackage[table]{{xcolor}}
\\usepackage{{booktabs}}
\\usepackage{{geometry}}
\\usepackage{{graphicx}}
\\geometry{{a4paper, margin=1cm}}

\\begin{{document}}

\\begin{{table}}[h]
\\centering
\\caption{{{title}}}
\\label{{tab:results}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{{table_format}}}
\\toprule
{header}"""
    
    for i1, var1 in enumerate(vars1):
        task_name = var1.replace('_', '\\_')
        latex_content += task_name
        
        for i2, var2 in enumerate(vars2):
            if ignore_diagonal and i1 == i2:
                latex_content += " & N/A"
                continue
            if results_matrix[var1][var2] is not None:
                mean_score = results_matrix[var1][var2]
                ci_lower, ci_upper = ci_matrix[var1][var2]
                significant = significant_matrix[var1][var2]
                
                # Format the cell content (confidence interval only)
                cell_content = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
                
                # Add color if significant
                if significant:
                    if mean_score > 0.5:
                        latex_content += f" & \\cellcolor{{green!50}} {cell_content}"
                    else:
                        latex_content += f" & \\cellcolor{{red!50}} {cell_content}"
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
\textbf{Note:} Cells highlighted in green/red indicate 95\% confidence intervals that do not include 50\% performance (chance level). Values shown as [lower CI, upper CI].

\end{document}
"""
    
    return latex_content

def main():
    """Main function to process results and generate LaTeX table."""
    
    print("Loading and analyzing results...")
    if plot_what == "tasks_factors":
        results_matrix, ci_matrix, significant_matrix = load_and_analyze_results_tasks_factors()
    elif plot_what == "releases_r12":
        results_matrix, ci_matrix, significant_matrix = load_and_analyze_results_releases_r12()
    elif plot_what == "releases_releases":
        results_matrix, ci_matrix, significant_matrix = load_and_analyze_results_releases_releases()
    
    # Create and save pandas dataframe as PNG
    df = pd.DataFrame(results_matrix).T
    df.to_csv("results_matrix.csv")
    print("Results matrix saved to results_matrix.csv")
    
    print("\nGenerating LaTeX table...")
    if plot_what == "tasks_factors":
        latex_content = create_latex_table(results_matrix, ci_matrix, significant_matrix, tasks, factors)
    elif plot_what == "releases_r12":
        latex_content = create_latex_table(results_matrix, ci_matrix, significant_matrix, releases, testset)
    elif plot_what == "releases_releases":
        latex_content = create_latex_table(results_matrix, ci_matrix, significant_matrix, releases_train, releases_test)
    
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