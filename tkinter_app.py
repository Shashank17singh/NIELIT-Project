"""
Tkinter desktop client for the House Price Predictor.
Shares ML logic and SQLite persistence with the main Streamlit application.
"""

import tkinter as tk
from tkinter import messagebox, ttk

import pandas as pd

# Import our load_and_train from main to keep logic DRY
from main import format_inr, load_and_train, save_prediction

print("Loading model and dataset... Please wait.")
model_rf, model_lr, df, metrics = load_and_train()
print("Model loaded successfully.")


class HousePricePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("House Price Predictor (Tkinter)")
        self.root.geometry("600x450")
        self.root.configure(bg="#F0FDFA")

        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TLabel", background="#F0FDFA", foreground="#134E4A", font=("Helvetica", 11)
        )
        style.configure("TButton", font=("Helvetica", 11, "bold"))

        title_label = tk.Label(
            root,
            text="House Price Predictor",
            font=("Helvetica", 16, "bold"),
            bg="#F0FDFA",
            fg="#0F766E",
        )
        title_label.pack(pady=15)

        frame = tk.Frame(root, bg="#F0FDFA")
        frame.pack(pady=10)

        # Area
        ttk.Label(frame, text="Area (in sqft):").grid(
            row=0, column=0, padx=10, pady=10, sticky="w"
        )
        self.area_var = tk.StringVar(value="1000")
        ttk.Entry(frame, textvariable=self.area_var).grid(
            row=0, column=1, padx=10, pady=10
        )

        # BHK
        ttk.Label(frame, text="Number of BHK:").grid(
            row=1, column=0, padx=10, pady=10, sticky="w"
        )
        self.bhk_var = tk.StringVar(value="2")
        ttk.Entry(frame, textvariable=self.bhk_var).grid(
            row=1, column=1, padx=10, pady=10
        )

        # Region
        ttk.Label(frame, text="Region:").grid(
            row=2, column=0, padx=10, pady=10, sticky="w"
        )
        self.region_var = tk.StringVar()
        regions = sorted(df["region_clean"].unique())
        self.region_cb = ttk.Combobox(
            frame, textvariable=self.region_var, values=regions, state="readonly"
        )
        self.region_cb.grid(row=2, column=1, padx=10, pady=10)
        if regions:
            self.region_cb.current(0)

        # Type
        ttk.Label(frame, text="Property Type:").grid(
            row=3, column=0, padx=10, pady=10, sticky="w"
        )
        self.type_var = tk.StringVar()
        types = list(df["type"].unique())
        self.type_cb = ttk.Combobox(
            frame, textvariable=self.type_var, values=types, state="readonly"
        )
        self.type_cb.grid(row=3, column=1, padx=10, pady=10)
        if types:
            self.type_cb.current(0)

        # Status
        ttk.Label(frame, text="Status:").grid(
            row=4, column=0, padx=10, pady=10, sticky="w"
        )
        self.status_var = tk.StringVar()
        statuses = list(df["status"].unique())
        self.status_cb = ttk.Combobox(
            frame, textvariable=self.status_var, values=statuses, state="readonly"
        )
        self.status_cb.grid(row=4, column=1, padx=10, pady=10)
        if statuses:
            self.status_cb.current(0)

        # Age
        ttk.Label(frame, text="Age of Property:").grid(
            row=5, column=0, padx=10, pady=10, sticky="w"
        )
        self.age_var = tk.StringVar()
        ages = list(df["age"].unique())
        self.age_cb = ttk.Combobox(
            frame, textvariable=self.age_var, values=ages, state="readonly"
        )
        self.age_cb.grid(row=5, column=1, padx=10, pady=10)
        if ages:
            self.age_cb.current(0)

        # Predict Button
        self.predict_btn = tk.Button(
            root,
            text="Predict Price",
            font=("Helvetica", 12, "bold"),
            bg="#0ea5e9",
            fg="white",
            command=self.predict,
        )
        self.predict_btn.pack(pady=20)

    def predict(self):
        try:
            area = float(self.area_var.get())
            bhk = int(self.bhk_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid Input", "Please enter valid numeric values for Area and BHK."
            )
            return

        region = self.region_var.get()
        prop_type = self.type_var.get()
        status = self.status_var.get()
        age = self.age_var.get()

        input_data = pd.DataFrame(
            [
                {
                    "bhk": bhk,
                    "area": area,
                    "region_clean": region,
                    "type": prop_type,
                    "status": status,
                    "age": age,
                }
            ]
        )

        pred = model_rf.predict(input_data)[0]

        # Save to SQLite
        try:
            save_prediction(area, bhk, prop_type, region, status, age, pred)
        except Exception as e:  # noqa: BLE001
            print("Failed to save to SQLite:", e)

        messagebox.showinfo(
            "Prediction Result", f"Estimated Price:\n{format_inr(pred)}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = HousePricePredictorApp(root)
    root.mainloop()
