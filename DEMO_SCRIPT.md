# 🎨 Streamlit Dashboard Demo Script

## For Presentations & Live Demos

Use this script to guide your dashboard demonstration:

---

## 🎯 5-Minute Demo Flow

### **Opening (30 seconds)**
*"Today I'll show you an interactive dashboard that brings our building energy analysis to life. It has 5 main sections - let me walk you through each one."*

---

### **TAB 1: Overview & Models (1 minute)**

**Navigate to first tab, point to metrics at top:**

*"Our analysis compared 7 different machine learning models, including both default and optimized versions. The best performer is our Stacking Ensemble with an R-squared of 0.89, meaning it explains 89% of the variance in building energy use."*

**Scroll to bar chart:**

*"You can see here how each model performs. The stacking approach - shown in red - combines Random Forest, Gradient Boosting, and Linear Regression to achieve the best results. We also performed hyperparameter optimization on key models, improving Random Forest by 1.2% and Gradient Boosting by 1.1%."*

**Point to metrics table:**

*"The RMSE of 22.4 kBtu per square foot is excellent for real-world building data."*

---

### **TAB 2: Feature Importance (1 minute)**

**Click second tab, show the horizontal bar chart:**

*"This shows what drives our predictions. The single most important factor - at 52% - is the building's prior year performance. This confirms the principle that 'buildings are persistent' in their energy behavior."*

**Point to different colored bars:**

*"Features are categorized into Energy metrics (red), Building characteristics (blue), and Performance indicators (green). Weather-normalized EUI and electricity usage are also key drivers."*

---

### **TAB 3: BEPS Compliance (1 minute)**

**Click third tab, point to top metrics:**

*"Looking ahead to Seattle's 2030 compliance deadline, our model forecasts that 60% of buildings will meet the standards, but 40% - about 1,200 buildings - will need retrofits."*

**Scroll to bar chart:**

*"These are the actual BEPS targets by building type. NonResidential buildings must hit 65 kBtu per square foot by 2030, dropping to 50 by 2040."*

**Point to economic section:**

*"The economic impact is significant: retrofit investments could range from 75 million to 450 million dollars depending on the depth of improvements, with payback periods of 10 to 18 years."*

---

### **TAB 4: Risk Assessment (1.5 minutes) ⭐ INTERACTIVE!**

**Click fourth tab, go to Risk Calculator:**

*"This is where it gets interactive. Let me show you a high-risk building scenario."*

**Adjust sliders:**
- Current EUI: 150
- Building Type: NonResidential  
- Energy Star: 30
- YoY Change: +5%
- Size: 100,000

*"This building gets a risk score of 9 out of 10 - that's high risk. It's projected to be 85 kBtu above the 2030 target, with estimated retrofit costs of $0.5M to $3M."*

**Change to compliant scenario:**
- Current EUI: 50
- Energy Star: 80
- YoY Change: -3%

*"Compare that to this efficient building - risk score of only 2 out of 10, already 15 kBtu below target. No retrofits needed."*

**Scroll down to table:**

*"We've identified the top 20 highest-risk buildings in the portfolio, ranked by risk score and showing their projected gap from BEPS targets."*

---

### **TAB 5: Diagnostics (30 seconds)**

**Click fifth tab, show residual plot:**

*"Finally, our model validation. These diagnostic plots confirm the model assumptions are met."*

**Point to Q-Q plot:**

*"The Q-Q plot shows residuals follow a normal distribution - that's what we want to see."*

**Point to confidence interval chart:**

*"And our bootstrap analysis provides 95% confidence intervals for every prediction, giving us realistic uncertainty bounds for planning purposes. The coverage is 93.5%, which means our intervals are well-calibrated."*

---

### **Closing (30 seconds)**

**Return to Overview tab:**

*"So in summary, we have a production-ready system that:
- Predicts building energy with 89% accuracy
- Identifies high-risk buildings needing retrofits  
- Quantifies policy impacts with economic analysis
- Provides uncertainty estimates for decision-making

And it's all interactive - you can explore any scenario in real-time. Questions?"*

---

## 🎬 Alternative Demo Paths

### **For Technical Audiences (Engineers/Data Scientists)**

Focus on:
1. **Model comparison** (Tab 1) - discuss ensemble methods
2. **Feature engineering** (Tab 2) - temporal features, interactions
3. **Diagnostics** (Tab 5) - residual analysis, bootstrap CI
4. Mention SHAP for interpretability

### **For Non-Technical Audiences (Building Owners/Policymakers)**

Focus on:
1. **BEPS Compliance** (Tab 3) - economic impacts, environmental benefits
2. **Risk Calculator** (Tab 4) - show their building type scenarios
3. **High-risk list** (Tab 4) - prioritization for programs
4. Keep it simple, avoid technical jargon

### **For Investors/Stakeholders**

Focus on:
1. **Economic impact** (Tab 3) - ROI, payback periods
2. **Risk assessment** (Tab 4) - portfolio risk scoring
3. **Accuracy metrics** (Tab 1) - confidence in predictions
4. **Scalability** - can apply to other cities/regions

---

## 💡 Pro Tips for Live Demos

### **Before You Start:**
- ✅ Test the dashboard (run `streamlit run app.py`)
- ✅ Have browser ready at localhost:8501
- ✅ Clear browser cache for fresh load
- ✅ Close other apps to ensure smooth performance
- ✅ Have backup slides ready (in case of technical issues)

### **During the Demo:**
- 🎯 **Use Full Screen**: Press F11 for immersive experience
- 🎯 **Zoom In**: Ctrl/Cmd + to make text larger for audience
- 🎯 **Pause on Key Points**: Don't rush through slides
- 🎯 **Interact Confidently**: Practice slider movements beforehand
- 🎯 **Narrate Actions**: Say what you're doing ("Let me adjust this slider...")

### **Handling Questions:**
- 📊 "How accurate is this?" → Point to R² = 0.89 (Tab 1)
- 💰 "What's the cost?" → Show retrofit investment (Tab 3)
- 🏢 "What about my building?" → Use Risk Calculator (Tab 4)
- 📈 "How confident are you?" → Show bootstrap CI (Tab 5)
- 🔍 "What matters most?" → Show feature importance (Tab 2)

---

## 🎓 Academic Presentation Tips

### **For Class Presentations:**

**Structure (10-15 minutes):**
1. **Context** (2 min): Problem statement, BEPS policy background
2. **Approach** (3 min): ML pipeline, models, features
3. **Demo** (5 min): Walk through dashboard tabs
4. **Results** (3 min): Key findings, insights, recommendations
5. **Q&A** (2 min): Handle questions with dashboard

**Key Points to Hit:**
- Graduate-level enhancements (SHAP, Bootstrap CI, Cross-validation)
- Real-world impact (policy compliance, economic analysis)
- Technical rigor (model validation, uncertainty quantification)
- Reproducibility (documented code, clear methodology)

### **For Peer Review:**

Share this link structure:
1. Overview → establish credibility
2. Risk Calculator → engage them interactively
3. Their feedback → "What would you add?"

---

## 📸 Screenshot Opportunities

**Capture these for reports/slides:**
1. Model comparison bar chart (Tab 1)
2. Feature importance (Tab 2)
3. BEPS compliance forecast (Tab 3)
4. Risk calculator with results (Tab 4)
5. Bootstrap confidence intervals (Tab 5)

**How to screenshot in Streamlit:**
- Use browser built-in (Cmd+Shift+4 on Mac, Win+Shift+S on Windows)
- Or: Right-click chart → Save image as...

---

## 🚀 Deployment for Presentations

### **Option 1: Laptop Local**
```bash
streamlit run app.py
# Simple, works offline, full control
```

### **Option 2: Streamlit Cloud (for online access)**
1. Push to GitHub
2. Deploy on share.streamlit.io
3. Share link: `https://yourusername-building-energy.streamlit.app`
4. Access from any device with internet

### **Option 3: University Network**
```bash
streamlit run app.py --server.address 0.0.0.0
# Share IP address with classmates in same network
```

---

## 🎯 Success Metrics

Your demo is successful if audience can:
- ✅ Understand the problem (BEPS compliance challenge)
- ✅ Trust the solution (89% R² accuracy)
- ✅ See the value (economic & environmental impact)
- ✅ Use the tool (try risk calculator themselves)
- ✅ Ask informed questions (dive deeper into specific areas)

---

## 🎉 Bonus: Wow Factors

**Surprise your audience with:**
1. **Real-time predictions**: Adjust sliders, see instant results
2. **High-quality visuals**: Professional charts with custom styling
3. **Interactive exploration**: "Let's try your building scenario"
4. **Comprehensive analysis**: All aspects covered in one tool
5. **Production-ready**: This could be deployed for real use tomorrow

---

**Ready to present?** Run `streamlit run app.py` and follow this script!

**Last Updated**: December 4, 2024  
**Demo Duration**: 5-15 minutes (adjust to your time slot)  
**Audience**: All levels (technical to non-technical)
