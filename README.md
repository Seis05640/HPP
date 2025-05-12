
# 🏠 House Price Prediction Web App

This project is a complete machine learning pipeline and web application to predict housing prices based on various structural and locational features using a stacked regression model. It includes data preprocessing, feature engineering, model training, Flask-based deployment, and a hosted live app via Render.

---

## 🚀 Features

- 🧼 Data cleaning and preprocessing
- 🧠 Feature engineering (e.g., amenities, social infrastructure)
- 📊 Stacked regression model (XGBoost + Random Forest + Linear Regression)
- 🌐 Web interface built with Flask
- ☁️ Deployed on [Render](https://render.com)

---

## 🗂️ Folder Structure

```
house_price_app/
├── app.py                  # Flask application
├── model/
│   ├── model.pkl           # Trained stacking model
│   ├── scaler.pkl          # Scaler for numerical features
│   └── features.pkl        # List of selected features
├── static/
│   └── css/style.css       # Custom styles
├── templates/
│   ├── index.html          # Input form
│   └── result.html         # Result output
├── requirements.txt        # Python dependencies
├── Procfile                # For Heroku/Render deployment
└── README.md               # This file
```

---

## 🧪 Tech Stack

- **Python 3.9+**
- **Flask**
- **Pandas, NumPy**
- **scikit-learn, XGBoost**
- **HTML/CSS (Flask templates)**
- **Render for deployment**

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/house-price-predictor.git
cd house_price_app
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app locally
```bash
python app.py
```

Then open your browser at `http://127.0.0.1:5000`

---

## ☁️ Deployment (Render)

- Create a new Web Service on [Render](https://render.com)
- Connect your GitHub repository
- Set the **start command**:
  ```bash
  python app.py
  ```
- Add a `PORT` environment variable if needed
- Done ✅

---

## 🧠 Machine Learning Overview

- Preprocessing: Handling missing values, encoding categoricals
- Feature Selection: `SelectKBest` with `f_regression`
- Model: Stacking Regressor with:
  - `XGBRegressor`
  - `RandomForestRegressor`
  - `LinearRegression`

Performance Metrics:
- MAE
- RMSE
- R² Score

---

## 👥 Team Contributions

| Member     | Contributions |
|------------|----------------|
| Sellaiya   | Data cleaning, initial analysis |
| Anbumani   | Feature engineering and selection |
| Sriram     | Model development and tuning |
| Gokul      | Frontend (Flask) development |
| Poovarasan | Deployment, documentation, GitHub management |

---

## 📌 License

MIT License – feel free to use, modify, and share with attribution.

---

## 🙌 Acknowledgments

- Dataset sourced from Kaggle (Ames Housing)
- Inspired by real-world housing price models
