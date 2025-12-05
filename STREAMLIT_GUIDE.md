# 🚀 Streamlit Dashboard - Quick Start Guide

## Installation & Setup

### 1. Install Streamlit Dependencies
```bash
pip install -r requirements_streamlit.txt
```

Or install individually:
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost shap scipy pillow
```

### 2. Run the Dashboard
```bash
cd /Users/mounusha/Downloads/CEE501_final_project
streamlit run app.py
```

The dashboard will automatically open in your browser at: `http://localhost:8501`

---

## 📊 Dashboard Features

### **5 Interactive Pages:**

#### 1. **📊 Overview & Models**
- Model performance comparison (R², RMSE, MAE)
- Interactive bar charts and metrics
- Best model highlights (Stacking Ensemble)
- Performance summary tables

#### 2. **🔍 Feature Importance**
- Top 10 most important features
- SHAP value interpretations
- Feature categories (Energy, Building, Performance)
- Engineering impact analysis

#### 3. **📈 BEPS Compliance**
- 2030 & 2040 compliance forecasts
- Economic impact analysis (\$75M-\$450M investment)
- Environmental impact (GHG emissions)
- Interactive compliance visualization

#### 4. **🏢 Risk Assessment**
- **Interactive Risk Calculator** (try different building scenarios!)
- Top 20 high-risk buildings
- Risk scoring system
- Retrofit cost estimates
- Actionable recommendations

#### 5. **📉 Diagnostics**
- Residual analysis plots
- Q-Q plot for normality
- Bootstrap confidence intervals
- Cross-validation results
- Model validation summary

---

## 🎨 Dashboard Highlights

### **Interactive Features:**
- ✅ **Sliders & Inputs**: Adjust building parameters to calculate risk
- ✅ **Real-time Calculations**: See risk scores and projections instantly
- ✅ **Data Tables**: Sortable, filterable performance metrics
- ✅ **Professional Styling**: Custom CSS with color-coded metrics
- ✅ **Responsive Design**: Works on desktop and tablet

### **Visual Elements:**
- 📊 15+ Interactive Charts
- 🎨 Color-coded Risk Levels (🔴 High, 🟡 Medium, 🟢 Low)
- 📈 Trend Visualizations
- 🎯 Metric Cards with Highlights
- 📋 Styled DataFrames with Gradients

---

## 💡 How to Use

### **For Presentations:**
1. Run `streamlit run app.py`
2. Navigate through 5 tabs using the top menu
3. Use the **Risk Assessment** tab for live demos
4. Show interactive calculations to audience

### **For Analysis:**
1. Explore **Feature Importance** to understand model drivers
2. Review **BEPS Compliance** for policy insights
3. Use **Risk Assessment** to evaluate specific buildings
4. Check **Diagnostics** for model validation

### **For Stakeholders:**
1. Start with **Overview** for big picture
2. Jump to **BEPS Compliance** for economic impacts
3. Use **Risk Assessment** calculator to estimate their building's risk
4. Show **confidence intervals** for prediction reliability

---

## 🎯 Key Interactions to Demo

### **Risk Calculator (Tab 4):**
Try these scenarios:

**Scenario 1: High-Risk Building**
- Current EUI: 150 kBtu/sf
- Building Type: NonResidential
- Energy Star: 30
- YoY Change: +5%
- **Result**: Risk Score 9/10 🔴

**Scenario 2: Compliant Building**
- Current EUI: 50 kBtu/sf
- Building Type: Multifamily
- Energy Star: 80
- YoY Change: -3%
- **Result**: Risk Score 2/10 🟢

**Scenario 3: Borderline Building**
- Current EUI: 70 kBtu/sf
- Building Type: NonResidential
- Energy Star: 55
- YoY Change: +1%
- **Result**: Risk Score 5/10 🟡

---

## 🚀 Advanced Features

### **Customization Options:**

1. **Update Data**: Replace mock data with real model outputs
2. **Add Pages**: Create additional tabs in the main file
3. **Custom Styling**: Modify CSS in the `st.markdown()` section
4. **Export Features**: Add download buttons for reports

### **Deployment Options:**

**Option 1: Streamlit Cloud (Free)**
```bash
# Push to GitHub, then deploy on streamlit.io
git add .
git commit -m "Add Streamlit dashboard"
git push
# Go to share.streamlit.io and deploy
```

**Option 2: Local Network**
```bash
streamlit run app.py --server.address 0.0.0.0
# Access from other devices on network
```

**Option 3: Docker Container**
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements_streamlit.txt
CMD ["streamlit", "run", "app.py"]
```

---

## 📝 Tips for Best Experience

1. **Use Full Screen**: Click the hamburger menu → Settings → Wide mode
2. **High-DPI Display**: Charts render crisp on Retina/4K screens
3. **Dark Mode**: Streamlit supports dark theme (Settings → Theme)
4. **Mobile**: Dashboard is responsive but best on desktop/tablet

---

## 🐛 Troubleshooting

### **Dashboard won't start?**
```bash
# Check Streamlit installation
streamlit --version

# Reinstall if needed
pip install --upgrade streamlit
```

### **Import errors?**
```bash
# Install all dependencies
pip install -r requirements_streamlit.txt --upgrade
```

### **Port already in use?**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

### **Charts not showing?**
```bash
# Clear Streamlit cache
streamlit cache clear
```

---

## 📊 Adding Real Model Data

To connect with your actual trained models:

1. **Save Models** (in notebook):
```python
import pickle
with open('models/stacking_model.pkl', 'wb') as f:
    pickle.dump(stacking_model, f)
```

2. **Load in Dashboard** (in app.py):
```python
@st.cache_resource
def load_models():
    with open('models/stacking_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model
```

3. **Make Predictions**:
```python
model = load_models()
prediction = model.predict(input_features)
```

---

## 🎓 Educational Use

Perfect for:
- ✅ **Class Presentations**: Interactive demos engage audience
- ✅ **Office Hours**: Show professors live calculations
- ✅ **Portfolio**: Deploy to cloud for resume/interviews
- ✅ **Peer Review**: Share local link with classmates
- ✅ **Stakeholder Demos**: Non-technical audience friendly

---

## 🔗 Useful Links

- **Streamlit Docs**: https://docs.streamlit.io
- **Component Gallery**: https://streamlit.io/gallery
- **Deployment Guide**: https://docs.streamlit.io/streamlit-community-cloud
- **Custom Components**: https://streamlit.io/components



**Last Updated**: December 4, 2024  
**Status**: ✅ Ready to deploy  
**Runtime**: < 5 seconds startup  
**Browser**: Chrome, Firefox, Safari, Edge
