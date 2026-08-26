<div align="center">

# House Price Predictor

**A responsive web application that predicts house prices using Linear Regression, built with Streamlit and scikit-learn**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Linear%20Regression-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

</div>

---

## Overview

A simple House Price Prediction web application built using **Streamlit** and **Linear Regression** (scikit-learn). Users pick a location and parking option from dropdowns, enter house details (area, bedrooms, bathrooms, and age), and get an instant price estimate powered by localized INR pricing algorithms and age-depreciation logic.

---

### Application Flow

```mermaid
graph TD
    subgraph "Streamlit Frontend"
    A[User Selects Location & Parking]
    B[User Enters Square Footage, Age, etc.]
    A --> C(Click 'Predict Price')
    B --> C
    end
    
    subgraph "Backend Logic"
    C --> D{Input Validation}
    D -->|Valid| E(Data Encoding)
    D -->|Invalid| F[Show Error Message]
    end
    
    subgraph "Machine Learning"
    E --> G{Scikit-Learn Model}
    G -->|Linear Regression| H[Price Estimate]
    H --> I[Update UI with formatted INR]
    end
    
    classDef io fill:#f9f0ff,stroke:#8a2be2,stroke-width:2px,color:#000;
    classDef core fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000;
    classDef logic fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000;
    
    class A,B,C,F,I io;
    class D,E core;
    class G,H logic;
```

## Features

| | |
|---|---|
| **Responsive Web App** | Built entirely with Streamlit for a fast, interactive experience |
| **Linear Regression Model** | Trained on a sample housing dataset with pandas + scikit-learn |
| **Realistic INR Pricing** | Predicts prices mapped to the Indian number system formatting |
| **Age Depreciation** | Automatically accounts for property age in pricing logic |

---

## Tech Stack

**Language** - Python 3.x
**Web Framework** - Streamlit
**Machine Learning** - pandas · scikit-learn (Linear Regression)

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/Shashank17singh/NIELIT-Project.git
cd NIELIT-Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run main.py
```

Fill in the house details, select a location and parking option, and click predict to see the estimated price.

---
