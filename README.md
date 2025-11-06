🧴 Skin Disorder Classification App  
An interactive **Machine Learning Web App** built with **Streamlit**, that predicts the type of **skin disorder** based on dermatological parameters.  
This project showcases an end-to-end Data Science workflow — from data preprocessing and model training to live deployment with a CSV-first interface.

---

🌐 **Live Demo**  
👉 [Skin Disorder Classification – Streamlit App](https://skin-disorder-app-z8vnf3qzghtub72mbyfg5s.streamlit.app/)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://skin-disorder-app-z8vnf3qzghtub72mbyfg5s.streamlit.app/)


---

🧠 **Project Overview**  
This application allows users to:  
- **Upload a CSV** for instant multi-row predictions (CSV-first UI)  
- Automatically handle headers, scaling, and numeric corrections  
- Use a pre-trained ML model (Random Forest + Logistic Regression baseline)  
- Optionally test a single manual input row in an expander section  

---

🧩 **Features**  
✅ CSV upload with preview  
✅ Auto header correction & feature alignment  
✅ NaN-safe median imputation  
✅ Scaler + model auto-loading from artifacts  
✅ ~98.61 % test accuracy (reproduced)  
✅ Downloadable predictions CSV  
✅ Optional single-row manual test input  

---

🧮 **Input Parameters**  
34 numeric dermatological features such as:  
`erythema`, `scaling`, `definite_borders`, `itching`, `koebner_phenomenon`, `follicular_papules`, `family_history`, etc.  

---

📊 **Dataset Summary**  
- **Rows:** 358 (after cleaning missing values)  
- **Features:** 34  
- **Target:** `class` (6 skin disorder categories)  
- **Source:** Dermatology Dataset (UCI Machine Learning Repository)

---

🧰 **Tech Stack**  
- Python 3.11 +  
- Streamlit (Web UI)  
- scikit-learn (ML Pipeline)  
- pandas · numpy (Data Processing)  
- joblib (Model Serialization)  
- FastAPI (optional API)  
- GitHub + Streamlit Cloud (Deployment)

---

🚀 **How to Run Locally**
```bash
# Clone this repository
git clone <your-repo-url>.git
cd <your-repo-folder>

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app_streamlit.py

✨ Author

👩🏻‍💻 Navjot Kaur
🎓 MSc (IT) | Certified Data Scientist | Streamlit Developer
📍 Jalandhar, Punjab, India

🌐 Connect with me:

💼 GitHub(https://github.com/Navjotkaur-22)

🔗 LinkedIn(https://www.linkedin.com/in/navjot-kaur-b61aab299/)

💬 [Upwork — Navjot Kaur](https://www.upwork.com/freelancers/~01b30aa09d478b524c)
