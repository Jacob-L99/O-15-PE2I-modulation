import tkinter as tk
from tkinter import filedialog, messagebox

# Global variables to track selected directory and choice
selected_directory = None
choice_var = None  # Initialize the choice_var variable

def parkinson_function(initial_window, initial, on_proceed):
    global selected_directory, choice_var  # Ensure choice_var is global

    # Clear the current window to update it for the Parkinson screen
    for widget in initial_window.winfo_children():
        widget.destroy()

    # Variable to store the choice between F-18 and C-11
    choice_var = tk.StringVar(value="")

    # Function to handle selection of F-18 or C-11
    def select_option(option):
        choice_var.set(option)
        # Update button appearances
        if option == "F-18":
            F18_btn.config(bg='grey', fg='white')  # Selected button
            c11_btn.config(bg='light grey', fg='dark grey')   # Deactivated appearance
        elif option == "C-11":
            F18_btn.config(bg='light grey', fg='dark grey')  # Deactivated appearance
            c11_btn.config(bg='grey', fg='white')   # Selected button

    # Function to reset button appearances (initial state)
    def reset_buttons():
        F18_btn.config(bg='light grey', fg='white')
        c11_btn.config(bg='light grey', fg='white')

    # Continue button at the top
    continue_btn = tk.Button(initial_window, text="Fortsätt", command=lambda: proceed(on_proceed), fg='white', bg='gray')
    continue_btn.pack(pady=20)

    # Frame to hold the Buttons under "Fortsätt"
    choice_frame = tk.Frame(initial_window, bg='black')
    choice_frame.pack(pady=10)

    # Buttons for F-18 and C-11
    F18_btn = tk.Button(choice_frame, text="F-18", command=lambda: select_option("F-18"), fg='white', bg='grey')
    F18_btn.pack(side=tk.LEFT, padx=10)

    c11_btn = tk.Button(choice_frame, text="C-11", command=lambda: select_option("C-11"), fg='white', bg='grey')
    c11_btn.pack(side=tk.LEFT, padx=10)

    # Reset buttons to initial appearance
    reset_buttons()

    # "Välj mapp" button
    choose_dir_btn = tk.Button(initial_window, text="Välj mapp", command=lambda: choose_directory(choose_dir_btn), fg='white', bg='grey')
    choose_dir_btn.pack(pady=20)

    # "15-O" button to return to the initial screen
    back_btn = tk.Button(initial_window, text="15-O", command=initial, fg='white', bg='grey')
    back_btn.pack(pady=20)

# Function to handle selecting directories and update button text
def choose_directory(button):
    global selected_directory
    directory = filedialog.askdirectory(title="Välj mapp")
    if directory:
        selected_directory = directory
        # Update button text to show the selected folder name
        button.config(text=f"Mapp: {directory.split('/')[-1]}")
        print(f"Selected directory: {directory}")

# Function to handle the "Fortsätt" button press and return values
def proceed(on_proceed):
    global selected_directory, choice_var

    # Check if at least one directory is selected
    if selected_directory:
        if choice_var and choice_var.get():
            print(f"Proceeding with directory: {selected_directory} and choice: {choice_var.get()}")
            on_proceed(selected_directory, choice_var.get())  # Call the callback function with the selected paths and choice
        else:
            messagebox.showinfo("Info", "Välj en av alternativen 'F-18' eller 'C-11' för att fortsätta.")
    else:
        messagebox.showinfo("Info", "Välj åtminstone en mapp för att fortsätta.")
