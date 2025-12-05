# 🎤 5-Minute Presentation Speaker Notes
## Building Energy Performance Dashboard

**By: Mounusha Ram Metti & Kashish Patel**  
**Course: CEE 501 - AI For Civil Engineers**

---

## 🎯 **SLIDE 1: OVERVIEW (30 seconds)**

*[Open dashboard - Overview tab visible]*

**SAY:**

"Good morning everyone. Today I'm presenting our Building Energy Performance prediction system for Seattle's commercial buildings. This is a comprehensive machine learning pipeline we built to help building owners and city planners prepare for Seattle's Building Energy Performance Standards - or BEPS - which require buildings to meet strict energy efficiency targets by 2030 and 2040."

*[Point to the 4 metric cards at top]*

"Our analysis covers approximately 3,000 buildings tracked from 2015 to 2016. We developed a 26-phase pipeline testing 7 different machine learning models. The best performer achieved an R-squared of 0.944 - meaning it explains 94.4% of the variance in building energy use - with a root mean squared error of just 13.3 kBtu per square foot."

---

## 📊 **SLIDE 2: MODEL COMPARISON (45 seconds)**

*[Scroll down to show the bar chart]*

**SAY:**

"Here you can see our comprehensive model comparison. We tested everything from simple Linear Regression to advanced ensemble methods including Random Forest, Gradient Boosting, XGBoost, and even a Stacking Ensemble that combines multiple models."

*[Point to the bars on the chart]*

"What's interesting here is that Linear Regression - shown in red - actually outperformed all the complex models. This isn't a failure of methodology - it's actually a validation of proper feature engineering, which I'll explain at the end."

*[Point to the table on the right]*

"This table shows all performance metrics. Notice all models achieve R-squared above 0.90, which indicates excellent predictive performance across the board. But Linear Regression leads with 0.944."

---

## 🔍 **SLIDE 3: FEATURE IMPORTANCE (45 seconds)**

*[Click on "Feature Importance" tab]*

**SAY:**

"Moving to feature importance - this shows what drives our predictions. The single most important factor, at 52%, is the building's prior year performance. This makes intuitive sense from a civil engineering perspective - buildings are persistent in their energy behavior."

*[Point to the colored bars]*

"We categorized features into three groups: Energy metrics in red, Building characteristics in blue, and Performance indicators in green. You can see that weather-normalized EUI and electricity usage are also key drivers."

*[Scroll down slightly to show feature engineering section]*

"The bottom section explains our feature engineering approach. We created over 60 engineered features from just 20 raw variables - including temporal changes, physics-based ratios like energy per square foot, and interaction terms that capture how building type and size work together to affect energy use."

---

## 📈 **SLIDE 4: BEPS COMPLIANCE FORECASTING (1 minute)**

*[Click on "BEPS Compliance" tab]*

**SAY:**

"Now for the policy application. BEPS - Building Energy Performance Standards - requires Seattle buildings to meet specific energy targets. This section forecasts compliance."

*[Point to the 4 metric cards]*

"Our model predicts that by 2030, 60% of buildings - about 1,800 - will meet the standards naturally. However, 40% - roughly 1,200 buildings - will need retrofits. The investment required ranges from 75 million to 450 million dollars depending on retrofit depth, with payback periods of 10 to 18 years."

*[Point to the bar chart on left]*

"This chart shows the actual BEPS targets. For example, non-residential buildings must hit 65 kBtu per square foot by 2030, dropping to 50 by 2040. These targets become progressively more stringent."

*[Scroll down to economic impact section]*

"The economic analysis breaks down three retrofit scenarios - low cost, medium cost, and high cost - showing both the upfront investment and the annual energy savings. This helps building owners and policymakers make data-driven decisions about compliance strategies."

---

## 🏢 **SLIDE 5: BUILDING RISK ASSESSMENT (45 seconds)**

*[Click on "Risk Assessment" tab]*

**SAY:**

"This is the interactive risk calculator. Let me demonstrate with a high-risk building scenario."

*[Adjust sliders while talking]*

"If we have a non-residential building with current EUI of 150, low Energy Star score of 30, and energy use increasing by 5% year-over-year - *[finish adjusting]* - the model assigns a risk score of 9 out of 10."

*[Point to the projection]*

"It projects this building will be 85 kBtu above the 2030 target, requiring estimated retrofits of half a million to 3 million dollars."

*[Quickly change sliders to compliant scenario]*

"Compare that to an efficient building with EUI of 50, Energy Star of 80, and improving performance - risk score of only 2 out of 10, already 15 kBtu below target."

*[Scroll to table at bottom]*

"This table lists the top 20 highest-risk buildings in our dataset, giving city planners a prioritized action list."

---

## 📉 **SLIDE 6: MODEL DIAGNOSTICS (30 seconds)**

*[Click on "Diagnostics" tab]*

**SAY:**

"Finally, the diagnostics tab validates our model assumptions. The residual plot shows prediction errors are randomly distributed with no systematic patterns - exactly what we want. The Q-Q plot confirms our residuals follow a normal distribution, and the bootstrap confidence intervals demonstrate our predictions are reliable with 93.5% coverage."

*[Scroll to show bootstrap plot if time permits]*

"These diagnostics prove our model is statistically sound and ready for real-world deployment."

---

## 🔬 **SLIDE 7: WHY LINEAR REGRESSION WON (1 minute)**

*[Go back to Overview tab and scroll down to "Key Academic Finding" section]*

**SAY:**

"Now, let me address the elephant in the room - why did simple Linear Regression beat all the complex ensemble methods?"

*[Point to the blue info box on left]*

"There are four key reasons. First, building energy use has extremely strong temporal persistence - a building's 2016 performance is highly predictable from its 2015 performance with a near-perfect linear correlation."

"Second, our extensive feature engineering pre-computed all the non-linear relationships. We created ratios, normalized metrics, and interaction terms - so Linear Regression could fit these linearized relationships directly."

"Third, we have high-quality clean data with minimal noise and strong signals."

"Fourth - and this is the academic lesson - Occam's Razor applies. When relationships are fundamentally linear, simpler models generalize better."

*[Point to the yellow warning box on right]*

"The evidence is in our results. Notice that Random Forest actually got WORSE when we tuned it - dropping from 0.910 to 0.900. This is textbook overfitting. The complex models added unnecessary complexity to what is fundamentally a linear problem."

*[Face the audience]*

"This demonstrates something important: domain knowledge plus proper feature engineering can be more valuable than algorithm complexity. Our result validates that we understand when to use which model - and that's a key skill for AI engineers."

---

## 🎯 **CLOSING (15 seconds)**

**SAY:**

"In conclusion, we've built a complete end-to-end system that not only predicts building energy performance with 94% accuracy but also provides actionable insights for Seattle's BEPS compliance. Our analysis will help the city prioritize interventions and building owners plan retrofits efficiently."

*[Pause]*

"I'm happy to take any questions. Thank you."

---

## ⏱️ **TIMING BREAKDOWN:**
- Slide 1 (Overview): 30s
- Slide 2 (Models): 45s  
- Slide 3 (Features): 45s
- Slide 4 (BEPS): 60s
- Slide 5 (Risk): 45s
- Slide 6 (Diagnostics): 30s
- Slide 7 (Why Linear): 60s
- Closing: 15s

**TOTAL: 5 minutes 30 seconds** (with buffer for navigation)

---

## 💡 **TIPS:**

1. **Practice with a timer** - rehearse until you hit 5:00-5:30 consistently
2. **Have dashboard open before you start** - don't waste time loading
3. **Know your tab order** - smooth transitions matter
4. **Memorize the key numbers**: 0.944 R², 13.3 RMSE, 60% compliant, $75M-$450M
5. **The Linear Regression section is your strength** - it shows analytical maturity
6. **Smile and make eye contact** - confidence matters as much as content

---

## 🚨 **IF YOU RUN OVER TIME:**

**Cut these sections first:**
1. Skip the interactive risk calculator demo (just mention it exists)
2. Skip the diagnostics tab entirely (just say "diagnostics confirm validity")
3. Shorten the economic impact details in BEPS section

**Never cut:**
- Overview metrics
- Model comparison chart
- Why Linear Regression won (this is your strongest point)

---

## 🎯 **ANTICIPATED QUESTIONS & ANSWERS:**

**Q: "Why only 2 years of data?"**  
A: "This is the publicly available Seattle benchmarking data. However, the temporal modeling approach we developed can scale to more years as data becomes available."

**Q: "How did you validate the model?"**  
A: "We used 5-fold cross-validation, bootstrap confidence intervals with 100 iterations, and comprehensive residual diagnostics. All validation metrics confirm the model is robust."

**Q: "Can this be applied to other cities?"**  
A: "Yes, the pipeline is generalizable. We'd need to adjust the BEPS targets and retrain on local data, but the methodology transfers directly."

**Q: "What about new buildings not in the training data?"**  
A: "The model uses building characteristics like type, size, and age, so it can generalize to new buildings with similar profiles. However, extreme outliers would need careful evaluation."

**Q: "Is the code available?"**  
A: "Yes, everything is on GitHub at Mounusha25/CEE501_Final_project - including the complete 26-phase notebook, dashboard code, and all documentation."

---

**Good luck with your presentation! 🎓🚀**
