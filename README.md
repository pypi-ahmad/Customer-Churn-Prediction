# Customer Churn Prediction Dashboard 🚀

An interactive Streamlit dashboard for predicting customer churn with multi-model comparison, advanced EDA, and built-in evaluation.

---

## ✨ New Features

- **Multi-Model Factory**: Trains Random Forest, XGBoost, SVM, and Decision Trees simultaneously.
- **Advanced EDA**: Interactive correlation heatmaps, distribution plots, and data previews.
- **Excel Support**: Native handling of .xlsx and .csv files (up to 1GB).
- **Model Comparison**: Side-by-side performance metrics and confusion matrices.

---

## 🧰 Tech Stack

- **Streamlit**
- **Plotly**
- **Pandas**
- **OpenPyXL**
- **Joblib**
- **Docker**

---

## 🚀 Quick Start

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Run with Docker

```bash
docker build -t churn-dashboard .
docker run -p 8501:8501 churn-dashboard
```

---

## 🖼️ Screenshots

### Dashboard View
![Dashboard View](docs/images/dashboard-view.png)

### Model Evaluation
![Model Evaluation](docs/images/model-evaluation.png)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

- **Dataset:** Bank customer churn data (BankChurners.csv)
- **Libraries:** Scikit-Learn, Pandas, Seaborn, Imbalanced-Learn
- **Tools:** Python 3.13, Jupyter, dtale

---

<div align="center">

**Made with ❤️ using Python 3.13+ and modern ML practices**

**⭐ Star this repository if you found it helpful!**

[Back to Top](#-customer-churn-prediction)

</div>