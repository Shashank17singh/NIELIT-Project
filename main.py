import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.linear_model import LinearRegression


def train_model():
    """Train and return a Linear Regression model with sample dataset."""
    data = {
        "Area": [1200, 800, 1600, 1000, 1400, 1800, 1100, 950, 1300],
        "Bedrooms": [3, 2, 4, 2, 3, 4, 2, 1, 3],
        "Bathrooms": [2, 1, 3, 2, 2, 3, 1, 1, 2],
        "Location": ["Standard", "Budget", "Premium", "Standard", "Premium", "Premium", "Budget", "Budget", "Standard"],
        "Parking": ["Yes", "No", "Yes", "No", "Yes", "Yes", "No", "No", "Yes"],
        "Age": [5, 10, 2, 8, 3, 1, 7, 9, 4],
        "Price": [6800000, 3500000, 9500000, 5000000, 8200000, 10500000, 4200000, 3100000, 7200000]
    }

    df = pd.DataFrame(data)

    location_map = {"Budget": 0, "Standard": 1, "Premium": 2}
    parking_map = {"No": 0, "Yes": 1}

    df["Location"] = df["Location"].map(location_map)
    df["Parking"] = df["Parking"].map(parking_map)

    X = df.drop("Price", axis=1)
    y = df["Price"]

    model = LinearRegression()
    model.fit(X, y)

    return model, location_map, parking_map


class HousePricePredictorApp:
    def __init__(self, root, model, location_map, parking_map):
        self.root = root
        self.model = model
        self.location_map = location_map
        self.parking_map = parking_map

        self.root.title("🏡 House Price Predictor")
        self.root.geometry("400x570")
        self.root.configure(bg="#f4f4f4")
        self.root.resizable(False, False)

        # Fonts
        self.font_label = ("Calibri", 11)
        self.font_heading = ("Calibri", 18, "bold")
        self.font_button = ("Calibri", 13, "bold")
        self.font_result = ("Calibri", 14, "bold")
        self.font_dropdown = ("Calibri", 11)

        self.build_ui()

    def build_ui(self):
        tk.Label(
            self.root,
            text="🏠 House Price Predictor",
            font=self.font_heading,
            bg="#f4f4f4"
        ).pack(pady=(10, 15))

        # Inputs
        self.area_entry = self.labeled_entry("Area (in sqft):")
        self.bedrooms_entry = self.labeled_entry("Number of Bedrooms:")
        self.bathrooms_entry = self.labeled_entry("Number of Bathrooms:")

        # Location Dropdown
        self.location_var = tk.StringVar()
        self.create_dropdown(
            label="Location:",
            var=self.location_var,
            values=["Budget", "Standard", "Premium"]
        )

        # Parking Dropdown
        self.parking_var = tk.StringVar()
        self.create_dropdown(
            label="Parking:",
            var=self.parking_var,
            values=["No", "Yes"]
        )

        self.age_entry = self.labeled_entry("Age of Property (Years):")

        # Result
        self.result_var = tk.StringVar()
        self.result_frame = tk.Frame(self.root, bg="#e0f8e0", bd=2, relief="ridge", padx=10, pady=12)
        self.result_label = tk.Label(
            self.result_frame,
            textvariable=self.result_var,
            font=self.font_result,
            fg="#064420",
            bg="#e0f8e0",
            justify="center",
            wraplength=320
        )
        self.result_label.pack()
        self.result_frame.pack_forget()

        # Predict Button
        tk.Button(
            self.root,
            text="Predict Price",
            command=self.predict_price,
            bg="#007F5F",
            fg="white",
            font=self.font_button,
            relief="flat",
            padx=10,
            pady=5
        ).pack(pady=(15, 0))

    def labeled_entry(self, label_text):
        frame = tk.Frame(self.root, bg="#f4f4f4")
        tk.Label(frame, text=label_text, font=self.font_label, bg="#f4f4f4").pack(anchor="w", pady=(5, 0))
        entry = tk.Entry(frame, font=self.font_label)
        entry.pack(fill="x", pady=(0, 5))
        frame.pack(fill="x", padx=10)
        return entry

    def create_dropdown(self, label, var, values):
        frame = tk.Frame(self.root, bg="#f4f4f4")

        tk.Label(frame, text=label, font=self.font_label, bg="#f4f4f4").pack(anchor="w", pady=(5, 0))

        dropdown = ttk.Combobox(
            frame,
            textvariable=var,
            values=values,
            state="readonly",
            font=self.font_dropdown
        )
        dropdown.pack(fill="x", pady=(0, 5))
        frame.pack(fill="x", padx=10)

    def predict_price(self):
        try:
            area = float(self.area_entry.get())
            bedrooms = int(self.bedrooms_entry.get())
            bathrooms = int(self.bathrooms_entry.get())
            age = int(self.age_entry.get())

            loc_text = self.location_var.get()
            park_text = self.parking_var.get()

            if not loc_text or not park_text:
                raise ValueError("Dropdown not selected")

            loc_encoded = self.location_map[loc_text]
            park_encoded = self.parking_map[park_text]

            input_df = pd.DataFrame([{
                "Area": area,
                "Bedrooms": bedrooms,
                "Bathrooms": bathrooms,
                "Location": loc_encoded,
                "Parking": park_encoded,
                "Age": age
            }])

            predicted_price = self.model.predict(input_df)[0]

            self.result_var.set(f"Estimated House Price:\n₹{int(predicted_price):,}")
            self.result_frame.pack(pady=15, padx=10, fill="x")

        except Exception:
            messagebox.showerror("Input Error", "Please fill all fields correctly (numbers + dropdowns).")


def main():
    model, location_map, parking_map = train_model()

    root = tk.Tk()
    HousePricePredictorApp(root, model, location_map, parking_map)
    root.mainloop()


if __name__ == "__main__":
    main()
