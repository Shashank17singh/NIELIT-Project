<div align="center">

# 🏠 House Price Predictor

**A desktop GUI app that predicts house prices using Linear Regression, built with Tkinter and scikit-learn**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Tkinter](https://img.shields.io/badge/Tkinter-GUI-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://docs.python.org/3/library/tkinter.html)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Linear%20Regression-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

</div>

---

## 📖 Overview

A simple House Price Prediction app built using a **Python Tkinter GUI** and **Linear Regression** (scikit-learn). Users pick a location and parking option from dropdowns, enter house details, and get an instant price estimate — no browser or server required.

---



### 🏗️ Application Flow

`mermaid
graph TD
    subgraph "Tkinter Frontend"
    A[User Selects Location & Parking]
    B[User Enters Square Footage etc.]
    A --> C(Click 'Predict Price')
    B --> C
    end
    
    subgraph "Backend Logic"
    C --> D{Input Validation}
    D -->|Valid| E(Data Encoding)
    D -->|Invalid| F[Show Error Popup]
    end
    
    subgraph "Machine Learning"
    E --> G{Scikit-Learn Model}
    G -->|Linear Regression| H[Price Estimate]
    H --> I[Update UI Label]
    end
    
    classDef io fill:#f9f0ff,stroke:#8a2be2,stroke-width:2px,color:#000;
    classDef core fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000;
    classDef logic fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000;
    
    class A,B,C,F,I io;
    class D,E core;
    class G,H logic;
`

## ✨ Features

| | |
|---|---|
| 🖥️ **User-Friendly GUI** | Built entirely with Python's standard-library Tkinter |
| 🧠 **Linear Regression Model** | Trained on a sample housing dataset with pandas + scikit-learn |
| 📍 **Dropdown Inputs** | Location and parking selection via dropdown menus |
| 🛡️ **Error Handling** | Validates inputs and handles invalid entries gracefully |

---

## 🛠️ Tech Stack

**Language** — Python 3.x
**GUI Framework** — Tkinter (Python standard library)
**Machine Learning** — pandas · scikit-learn (Linear Regression)

---

## ⚙️ Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/Shashank17singh/NIELIT-Project.git
cd NIELIT-Project
```

### 2. Install dependencies

```bash
pip install pandas scikit-learn
```

### 3. Run the app

```bash
python main.py
```

Fill in the house details, select a location and parking option, and click predict to see the estimated price.

---
