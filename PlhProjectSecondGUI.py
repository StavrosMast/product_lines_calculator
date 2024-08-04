import pandas as pd
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter as tk
from CTkMessagebox import CTkMessagebox
import customtkinter as ctk
from tkinter import filedialog, IntVar, StringVar
from itertools import combinations
import threading

def open_factory(file_path):
    try:
        with open(file_path, 'r') as file:
            data = file.read()
            content = data.split('\n')
            n = int(content[0].split('=')[1])
            m = int(content[1].split('=')[1])
            c = (content[2].split('=')[1])
            c = c.strip().lstrip('[').rstrip(']').split(',')
            c = list(map(float, c))
            a = (content[3].split('=')[1])
            a = a.strip('\'')
            a_list = a.strip(' ').split('[')
            b_list = []
            for x in a_list:
                if len(x) > 0:
                    b_list.append(x.strip('], '))

            a = []
            for b in b_list:
                tmp = []
                for k in b.split(','):
                    tmp.append(int(k))
                a.append(tmp)

    except Exception as e:
        print(f'Error: {e} while loading file {file_path}')
        return None

    return n, m, c, a

def open_prob(file_path):
    try:
        with open(file_path, 'r') as file:
            data = file.read()
            content = data.split('\n')
            m = int(content[0].split('=')[1])
            p = content[1].split('=')[1]
            p = p.strip().strip('[]').split(',')
            p = list(map(int, p))
    except Exception as e:
        print(f'Error {e} while loading file {file_path}')
        return None
    return p

def maximum_rate(a):
    arr = np.array(a)
    max_rate = np.argmax(arr)
    return max_rate

def minimum_cost(c):
    arr = np.array(c)
    min_cost = np.argmin(arr)
    return min_cost

def min_cost_prod_list(c, m):
    min_list = []
    c_copy = c.copy()
    for _ in range(m):
        i = minimum_cost(c_copy)
        min_list.append(i)
        c_copy[i] = float('inf')
    return min_list

def all_combinations(M, N):
    total_combinations = list(combinations(range(1, N + 1), M))
    print("All possible combinations are {} ".format(len(total_combinations)))
    return total_combinations

def optimum_combination(n, m, c, a, p):
    c_array = np.array(c)
    p_array = np.array(p).reshape(-1, 1)
    a_array = np.array(a)

    best_indices = min_cost_prod_list(c, m)
    sub_a = a_array[:, best_indices]

    if sub_a.shape[0] != p_array.shape[0]:
        print(f"Shape mismatch: sub_a has {sub_a.shape[0]} rows, but p_array has {p_array.shape[0]} elements")
        return None, None

    if np.linalg.det(sub_a) != 0:
        sub_a_inv = np.linalg.inv(sub_a)
        try:
            T = np.linalg.solve(sub_a, p_array)
            total_cost = np.dot(c_array[best_indices], T).sum()
            best_combination = best_indices
            min_total_cost = total_cost
        except np.linalg.LinAlgError as e:
            print(f'Linear algebra error for combination {best_indices}: {e}')
            best_combination = None
            min_total_cost = None
    else:
        print('Submatrix a is not invertible.')
        best_combination = None
        min_total_cost = None

    return best_indices, min_total_cost

def sensitivity_analysis(c, best_combination, a, p, selected_line, distribution, num_simulations):
    costs = np.array(c)
    p_array = np.array(p).reshape(-1, 1)
    sub_A = np.array(a)[:, best_combination]
    sub_A_inv = np.linalg.inv(sub_A)
    original_costs = costs[list(best_combination)]
    original_cost = original_costs[selected_line]
    sensitivity_results = []

    for i in range(num_simulations):
        modified_costs = original_costs.copy()

        if distribution == 'Uniform':
            modified_costs = np.random.uniform(0.8 * original_costs, 1.2 * original_costs)
        elif distribution == 'Normal':
            modified_costs = np.random.normal(original_costs, 0.1 * original_costs)
        else:
            raise ValueError("Invalid distribution type")

        try:
            T = np.linalg.solve(sub_A_inv, p_array)
            total_cost = np.dot(modified_costs, T).sum()
            sensitivity_results.append(total_cost)
        except np.linalg.LinAlgError as e:
            print(f'Linear algebra error during sensitivity analysis: {e}')
            continue

    plt.figure(figsize=(10, 6))
    plt.plot(range(num_simulations), sensitivity_results, alpha=0.75, label='Total Production Cost')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Deviation from Initial Production Cost')
    plt.title('Sensitivity Analysis - Line Plot')
    plt.legend()
    plt.show()

class ProductionLineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Production Lines")
        self.factory_file = None
        self.prob_file = None
        self.n = None
        self.m = None
        self.c = None
        self.a = None
        self.p = None
        self.best_combination = None

        self.set_theme()
        self.create_widgets()

    def set_theme(self):
        self.root.configure(bg="#0D1117")
        self.button_bg = "#21262D"
        self.button_hover = "#30363D"
        self.button_text = "#58A6FF"
        self.primary_text = "#C9D1D9"
        self.frame_bg = "#0D1117"

    def create_widgets(self):
        center_frame = ctk.CTkFrame(self.root, fg_color=self.frame_bg)
        center_frame.pack(expand=True, fill="both")
        center_frame.grid_columnconfigure(0, weight=1)
        center_frame.grid_columnconfigure(1, weight=1)
        center_frame.grid_rowconfigure(0, weight=1)
        center_frame.grid_rowconfigure(1, weight=1)
        center_frame.grid_rowconfigure(2, weight=1)
        center_frame.grid_rowconfigure(3, weight=1)
        center_frame.grid_rowconfigure(4, weight=1)
        center_frame.grid_rowconfigure(5, weight=1)

        success_label = ctk.CTkLabel(center_frame, text="", font=('Arial', 12), fg_color=self.primary_text)
        success_label.grid(row=6, column=0, padx=10, pady=10, columnspan=2)

        self.factory_file_map = self.get_factory_file_map()
        self.selected_factory = tk.StringVar(self.root)
        self.selected_factory.set("Factory 1")
        self.factory_files = list(self.factory_file_map.keys())

        factory_label = ctk.CTkLabel(center_frame, text="Select a factory:", fg_color=self.frame_bg,
                                     text_color=self.primary_text)
        factory_label.grid(row=0, column=0, padx=5, pady=5)
        factory_dropdown = ctk.CTkOptionMenu(center_frame, variable=self.selected_factory, values=self.factory_files,
                                             command=lambda value: self.load_factory(success_label), fg_color=self.button_bg,
                                             button_hover_color=self.button_hover, text_color=self.button_text)
        factory_dropdown.grid(row=1, column=0, padx=5, pady=5)

        self.selected_factory.trace('w', lambda *args: self.update_problem_dropdown())

        self.problem_file_map = {}
        self.selected_problem = tk.StringVar(self.root)
        self.selected_problem.set("Problem 1")

        problem_label = ctk.CTkLabel(center_frame, text="Select a Problem:", fg_color=self.frame_bg,
                                     text_color=self.primary_text)
        problem_label.grid(row=0, column=1, padx=5, pady=5)
        self.problem_dropdown = ctk.CTkOptionMenu(center_frame, variable=self.selected_problem, values=[],
                                                  state="disabled", command=lambda value: self.load_problem(success_label),
                                                  fg_color=self.button_bg, button_hover_color=self.button_hover,
                                                  text_color=self.button_text)
        self.problem_dropdown.grid(row=1, column=1, padx=5, pady=5)

        optimize_button = ctk.CTkButton(center_frame, text="Run Optimization", command=self.run_optimization,
                                        state=ctk.DISABLED, fg_color=self.button_bg, hover_color=self.button_hover,
                                        text_color=self.button_text)
        optimize_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        self.optimize_button = optimize_button

        sensitivity_button = ctk.CTkButton(center_frame, text="Run Sensitivity Analysis", command=self.run_sensitivity_analysis,
                                           state=ctk.DISABLED, fg_color=self.button_bg, hover_color=self.button_hover,
                                           text_color=self.button_text)
        self.sensitivity_button = sensitivity_button
        sensitivity_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

        close_button = ctk.CTkButton(center_frame, text="Close", command=self.root.quit, fg_color=self.button_bg,
                                     hover_color=self.button_hover, text_color=self.button_text)
        close_button.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

    def update_problem_dropdown(self):
        factory_name = self.selected_factory.get()
        factory_file_path = self.factory_file_map[factory_name]
        self.problem_file_map = self.get_problem_file_map(factory_file_path)
        self.problem_dropdown.configure(values=list(self.problem_file_map.keys()))
        self.problem_dropdown.set("Problem 1")

    def get_factory_file_map(self):
        current_directory = Path.cwd()
        production_lines_path = current_directory / 'Production_lines'
        factory_files = [file for file in production_lines_path.iterdir() if file.is_file() and file.suffix == '.factory']
        factory_files.sort()  # Ensure the files are sorted to maintain correct order
        factory_file_map = {f"Factory {i + 1}": str(file) for i, file in enumerate(factory_files)}
        return factory_file_map

    def get_problem_file_map(self, factory_file_path):
        production_lines_path = Path(factory_file_path).parent
        factory_name = Path(factory_file_path).stem
        problem_files = [file for file in production_lines_path.iterdir() if file.is_file() and file.suffix == '.prob' and file.stem.startswith(factory_name)]
        problem_files.sort()  # Ensure the files are sorted to maintain correct order
        problem_file_map = {f"Problem {i + 1}": str(file) for i, file in enumerate(problem_files)}
        return problem_file_map

    def load_factory(self, label):
        factory_name = self.selected_factory.get()
        file_path = self.factory_file_map[factory_name]
        factory_data = open_factory(file_path)
        if factory_data:
            self.n, self.m, self.c, self.a = factory_data
            label.configure(text=f"{factory_name} successfully loaded",fg_color="green")
            self.problem_dropdown.configure(state=ctk.NORMAL)
            # self.optimize_button.configure(state=ctk.NORMAL)
        else:
            label.configure(text=f"Error loading {factory_name}",fg_color="red")

    def load_problem(self, label):
        problem_name = self.selected_problem.get()
        file_path = self.problem_file_map[problem_name]
        problem_data = open_prob(file_path)
        if problem_data:
            self.p = problem_data
            label.configure(text=f"{problem_name} successfully loaded",fg_color="green")
            self.optimize_button.configure(state=ctk.NORMAL)
        else:
            label.configure(text=f"Error loading {problem_name}",fg_color="red")

    def run_optimization(self):
        if self.n and self.m and self.c and self.a and self.p:
            self.best_combination, min_total_cost = optimum_combination(self.n, self.m, self.c, self.a, self.p)
            max_rate = maximum_rate(self.a)
            min_cost = minimum_cost(self.c)
            result_message = f"Best Combination: {self.best_combination}\n" \
                             f"Max Production Rate Line: {max_rate + 1}\n" \
                             f"Min Cost Line: {min_cost + 1}\n" \
                             f"Minimum Total Cost: {min_total_cost}"
            CTkMessagebox(title="Optimization Result", message=result_message)
            self.sensitivity_button.configure(state=ctk.NORMAL)
        else:
            CTkMessagebox(title="Error", message="Please load all files first.")

    def run_sensitivity_analysis(self):
        if self.best_combination is not None:
            sensitivity_window = ctk.CTkToplevel(self.root)
            sensitivity_window.title("Sensitivity Analysis")
            sensitivity_window.geometry('600x400+50+50')

            ctk.CTkLabel(sensitivity_window, text="Select Production Line:").pack(pady=5)
            selected_line = IntVar()  # Use standard tkinter IntVar
            ctk.CTkEntry(sensitivity_window, textvariable=selected_line).pack(pady=5)

            ctk.CTkLabel(sensitivity_window, text="Select Distribution:").pack(pady=5)
            distribution = StringVar()  # Use standard tkinter StringVar
            ctk.CTkOptionMenu(sensitivity_window, variable=distribution, values=["Uniform", "Normal"]).pack(pady=5)

            def perform_analysis():
                line = selected_line.get() - 1  # Adjust for zero-based indexing
                dist = distribution.get()

                # # Mapping Greek terms to the correct distribution types
                # if dist == "Uniform":
                #     dist = "Uniform"
                # elif dist == "Normal":
                #     dist = "Normal"
                # else:
                #     CTkMessagebox(title="Error", message="Invalid distribution type.").show()
                #     return

                if line >= 0 and line < len(self.best_combination):
                    sensitivity_analysis(self.c, self.best_combination, self.a, self.p, line, dist, 1000)
                    sensitivity_window.destroy()
                else:
                    CTkMessagebox(title="Error", message="Invalid input for line.").show()

            ctk.CTkButton(sensitivity_window, text="Run Analysis", command=perform_analysis).pack(pady=5)

        else:
            CTkMessagebox(title="Error", message="Please run optimization first.").show()

if __name__ == "__main__":
    root = ctk.CTk()
    app = ProductionLineApp(root)
    root.geometry("400x400")
    root.mainloop()
