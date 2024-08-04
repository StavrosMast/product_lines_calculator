import pandas as pd
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from pathlib import Path # Δυναμική ανάκτηση αρχείων
from CTkMessagebox import CTkMessagebox
import customtkinter as ctk
from tkinter import filedialog,IntVar, StringVar


def open_factory(file_path):
    try:
        with open(file_path,'r') as file:
            data=file.read()
            content=data.split('\n')  #Χωρίζω το αρχείο σε τμήματα
            n=int(content[0].split('=')[1]) #Αποθηκεύω τις τιμές σε μεταβλητές
            m=int(content[1].split('=')[1])
            c=(content[2].split('=')[1])
            c=c.strip().lstrip('[').rstrip(']').split(',')
            c=list(map(float,c))
            a=(content[3].split('=')[1])
            a=a.strip('\'')
            a_list = a.strip(' ').split('[')
            b_list=[]
            for x in a_list:
                if len(x)>0:
                    b_list.append(x.strip('], '))

            a=[]
            for b in b_list:
                tmp=[]
                for k in b.split(','):
                    tmp.append(int(k))
                a.append(tmp)

    except Exception as e:
        print(f'Σφάλμα: {e} κατα την φόρτωση του αρχείου {file_path}')
        return None

    return n,m,c,a


def open_prob(file_path):
    try:
        with open(file_path,'r') as file:
            data=file.read()
            content=data.split('\n') #χωρίζω το περιεχόμενο
            m=int(content[0].split('=')[1]) #αποθηκεύω το περιεχόμενο σε μεταβλητές
            p=content[1].split('=')[1]
            p=p.strip().strip('[]').split(',')
            p=list(map(int,p))
    except Exception as e:
        print(f'Σφάλμα {e} κατά την φόρτωση του αρχείου {file_path}')
        return None
    return  p



def maximum_rate(a): #Συνάρτηση με την οποία βρίσκουμε μέγιστο ρυθμό παραγωγής για κάθε προϊόν.Σαν ορισμα δέχεται τη λίστα με το ρυθμό παραγωγής κάθε προϊόντος
    arr=np.array(a)
    max_rate=np.argmax(arr)
    #print("Η γραμμή παραγωγής με το μέγιστο ρυθμό είναι: ",max_rate+1)
    return max_rate


def minimum_cost(c): #Συνάρτηση με την οποία βρίσκω το ελάχιστο κόστος για κάθε προϊόν.Σαν όρισμα δέχεται τη λίστα τα κόστη κάθε γραμμης παραγωγής
    arr=np.array(c)
    min_cost=np.argmin(arr)
    #print("Η γραμμή παραγωγής με το χαμηλότερο κόστος είναι :",min_cost+1)
    return min_cost
def min_cost_prod_list(c,m):
    min_list=[]
    c_copy = c.copy()
    for _ in range(m):
        i=minimum_cost(c_copy)
        min_list.append(i)
        c_copy[i]=float('inf')
    return min_list

def all_combinations(M,N): #Συνάρτηση η οποία επιστρέφει όλους τους δυνατούς συνδυασμούς και τον αριθμό τους.Παίρνει σαν ορίσματα τις επιλεγμένες γραμμές παραγωγής και τις συνολικές γραμμές παραγωγής
    total_combinations=list(combinations(range(1,N+1), M))
    print("Όλοι οι δυνατοί συνδυασμοί είναι {} ".format(len(total_combinations)))
    return total_combinations


def optimum_combination(n, m, c, a, p):
    c_array = np.array(c)
    p_array = np.array(p).reshape(-1, 1)  # Μετατρέπουμε το p σε κάθετο διάνυσμα.
    a_array = np.array(a)

    best_indices = min_cost_prod_list(c, m)

    sub_a = a_array[:, best_indices]  # Δημιουργία υποπίνακα με τις επιλεγμένες γραμμές παραγωγής



    if np.linalg.det(sub_a)!=0:
        sub_a_inv = np.linalg.inv(sub_a)
         #print(f"Dimensions of p_array: {p_array.shape}")
        try:
            T = np.linalg.solve(sub_a_inv, p_array)
            total_cost = np.dot(c_array[best_indices], T).sum()  # Υπολογισμός συνολικού κόστους
            best_combination=best_indices
            min_total_cost = total_cost



        except np.linalg.LinAlgError as e:
            print(f'Σφάλμα γραμμικής άλγεβρας για συνδυασμό {best_indices}: {e}')
            best_combination=None
            min_total_cost=None
    else:
        print('Ο υποπίνακας a δεν είναι αντιστρέψιμος.')
        best_combination = None
        min_total_cost = None


    return best_indices, min_total_cost

def sensitivity_analysis(c, best_combination, a, p,selected_line, distribution, num_simulations):
    costs = np.array(c)
    p_array = np.array(p).reshape(-1, 1)  # Μετατρέπουμε το p σε κάθετο διάνυσμα
    sub_A = np.array(a)[:, best_combination]

    sub_A_inv = np.linalg.inv(sub_A)
    original_costs = costs[list(best_combination)]
    original_cost = original_costs[selected_line]
    sensitivity_results = []

    for i in range(num_simulations):
        modified_costs = original_costs.copy()

        if distribution == 'Ομοιόμορφη':
            modified_costs = np.random.uniform(0.8 * original_costs, 1.2 * original_costs)
        elif distribution == 'Κανονική':
            modified_costs = np.random.normal(original_costs, 0.1 * original_costs)
        else:
            raise ValueError("Λάθος τύπος κατανομής")

        try:
            T = np.linalg.solve(sub_A_inv, p_array)

            total_cost = np.dot(modified_costs, T).sum()

            sensitivity_results.append(total_cost)
        except np.linalg.LinAlgError as e:
            print(f'Σφάλμα γραμμικής άλγεβρας κατά την ανάλυση ευαισθησίας: {e}')
            continue

    plt.figure(figsize=(10, 6))
    plt.plot(range(num_simulations), sensitivity_results, alpha=0.75, label='Συνολικό Κόστος Παραγωγής')
    plt.xlabel('Αριθμός Προσομοιώσεων')
    plt.ylabel('Απόκλιση από το Αρχικό Κόστος Παραγωγής')
    plt.title('Ανάλυση Ευαισθησίας - Line Plot')
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

        self.create_widgets()

    def create_widgets(self):
        # Center Frame
        center_frame = ctk.CTkFrame(self.root)
        center_frame.pack(expand=True)

        button_width = 260
        button_height = 55

        # Factory File Button
        factory_button = ctk.CTkButton(center_frame, text="Select Factory File", command=self.load_factory_file,width=button_width, height=button_height)
        factory_button.pack(pady=15)

        # Problem File Button
        prob_button = ctk.CTkButton(center_frame, text="Select Problem File", command=self.load_prob_file, state=ctk.DISABLED,width=button_width, height=button_height)
        self.prob_button = prob_button
        prob_button.pack(pady=15)

        # Optimization Button
        optimize_button = ctk.CTkButton(center_frame, text="Run Optimization", command=self.run_optimization, state=ctk.DISABLED,width=button_width, height=button_height)
        self.optimize_button = optimize_button
        optimize_button.pack(pady=15)

        # Sensitivity Analysis Button
        sensitivity_button = ctk.CTkButton(center_frame, text="Run Sensitivity Analysis", command=self.run_sensitivity_analysis, state=ctk.DISABLED,width=button_width, height=button_height)
        self.sensitivity_button = sensitivity_button
        sensitivity_button.pack(pady=15)

        # Close Button
        close_button = ctk.CTkButton(center_frame, text="Close", command=self.root.quit,width=button_width, height=button_height)
        close_button.pack(pady=15)

    def load_factory_file(self):
        self.factory_file = filedialog.askopenfilename()
        if self.factory_file:
            if not self.factory_file.endswith('.factory'):
                CTkMessagebox(title="Error", message="Please select a valid factory file (.factory).")
                return
            self.n, self.m, self.c, self.a = open_factory(self.factory_file)
            if self.n is not None and self.m is not None and self.c is not None and self.a is not None:
                CTkMessagebox(title="Success", message="Factory file loaded successfully!")
                self.prob_button.configure(state=ctk.NORMAL)
            else:
                CTkMessagebox(title="Error", message="Failed to load factory file.")

    def load_prob_file(self):
        self.prob_file = filedialog.askopenfilename()
        if self.prob_file:
            if not self.prob_file.endswith('.prob'):
                CTkMessagebox(title="Error", message="Please select a valid problem file (.prob).")
                return
            self.p = open_prob(self.prob_file)
            if self.p is not None:
                CTkMessagebox(title="Success", message="Problem file loaded successfully!")
                self.optimize_button.configure(state=ctk.NORMAL)
            else:
                CTkMessagebox(title="Error", message="Failed to load problem file.")

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
            distribution = StringVar()
            ctk.CTkOptionMenu(sensitivity_window, variable=distribution, values=["Ομοιόμορφη", "Κανονική"]).pack(pady=5)

            def perform_analysis():
                try:
                    line = selected_line.get() - 1  # Adjust for zero-based indexing
                    dist = distribution.get()
                    if line >= 0 and line < len(self.best_combination) and dist in ["Ομοιόμορφη", "Κανονική"]:
                        sensitivity_analysis(self.c, self.best_combination, self.a, self.p, line, dist, 1000)
                        sensitivity_window.destroy()
                    else:
                        CTkMessagebox(title="Error", message="Invalid input for line or distribution type.")
                except Exception as e:
                    CTkMessagebox(title="Error", message=f"Invalid input: {e}")

            ctk.CTkButton(sensitivity_window, text="Run Analysis", command=perform_analysis).pack(pady=5)

        else:
            CTkMessagebox(title="Error", message="Please run optimization first.")

if __name__ == "__main__":
    root = ctk.CTk()
    app = ProductionLineApp(root)
    root.geometry("400x600")
    root.mainloop()


#