# Energy Consumption Forecasting App

![App Screenshot](screenshot.png) 
![image](https://github.com/user-attachments/assets/189a4427-9746-45c4-8f37-223f60146106)
![image](https://github.com/user-attachments/assets/f4b2beba-cfd9-41a5-9d47-4055c758213f)
![image](https://github.com/user-attachments/assets/76045890-efd5-4033-8d08-1a8d2648bff5)
![image](https://github.com/user-attachments/assets/6bcf9461-cc82-443d-911f-ec497dc90d23)

A Streamlit-based web application that predicts future energy consumption using LSTM neural networks, with features designed specifically for Bengali-speaking users.
## **Problem Statement**

In Bangladesh, both residential and commercial users face significant challenges in managing electricity consumption due to:

1. **Unpredictable Usage Patterns**: Most consumers lack visibility into their future energy needs, leading to:
   - Unexpected high bills
   - Poor load management
   - Inefficient energy budgeting

2. **Limited Bengali-Language Tools**: Existing energy monitoring solutions:
   - Are primarily in English
   - Lack localized explanations
   - Don't provide actionable insights for Bangladeshi users

3. **Cost Management Difficulties**: Consumers struggle to:
   - Estimate electricity costs in local currency (BDT)
   - Identify peak/off-peak usage periods
   - Detect abnormal consumption patterns

4. **Energy Inefficiency**: Without proper forecasting:
   - 20-30% of energy is wasted unnecessarily (source: SREDA)
   - Users miss opportunities for cost savings

---

## **Objectives**

| # | Primary Objective | Technical Approach | User Benefit |
|---|-------------------|--------------------|--------------|
| 1 | Accurate Consumption Forecasting | Implement LSTM neural network for time-series prediction | Predict energy needs 24 hours in advance with 85%+ accuracy |
| 2 | Bengali-Localized Insights | Translate metrics and provide contextual explanations | Make energy data accessible to non-English speakers |
| 3 | Cost Conversion Feature | Integrate BDT/kWh rate input | Show estimated bills in local currency |
| 4 | Anomaly Detection System | Implement threshold-based alerts | Identify faulty appliances/unusual usage |
| 5 | Time-Based Analysis | Classify usage into peak/off-peak hours | Help users shift loads to cheaper periods |
| 6 | Actionable Recommendations | Context-aware energy saving tips in Bengali | Reduce consumption by 10-15% through behavioral changes |

**Key Innovation**: Combines technical forecasting with localized, practical guidance specifically designed for Bangladeshi users' needs.

---

### **Expected Outcomes**
1. 80% reduction in bill surprises through accurate forecasting
2. 30% improvement in user understanding through Bengali explanations
3. 15-20% potential cost savings through time-shifting recommendations

This format clearly shows:
- The real-world problems being addressed
- The technical solutions proposed
- The direct benefits to end-users
- Measurable success criteria

## Features

- **Bengali Language Support**: Fully localized interface with Bengali explanations
- **Accurate Forecasting**: Uses LSTM neural networks for time-series prediction
- **Practical Insights**:
  - Electricity cost conversion (kWh to Taka)
  - Anomaly detection for unusual consumption patterns
  - Time-based usage analysis (peak/off-peak hours)
- **Energy Saving Tips**: Context-aware recommendations in Bengali
- **Interactive Visualization**: Clear comparison of actual vs predicted consumption

## Key Metrics Explained

| Metric | English | Bengali Explanation |
|--------|---------|---------------------|
| MAE | Mean Absolute Error | গড় পরম ত্রুটি - Average absolute difference between predicted and actual values |
| RMSE | Root Mean Square Error | রুট মিন স্কোয়ার ত্রুটি - Emphasizes larger errors |
| MAPE | Mean Absolute Percentage Error | গড় শতকরা পরম ত্রুটি - Percentage difference from actual values |

## Installation
1. pychrarm 
2. Clone the repository:
```bash
git clone https://github.com/yourusername/energy-forecasting-app.git
cd energy-forecasting-app
3. pip install -r requirements.txt
4. streamlit run app.py
## Dataset
https://drive.google.com/file/d/12F0_QLu5saM8HpLlENi7Wtn0bZRXgnJp/view?usp=drive_link

## Requirements
Python 3.10
Streamlit
TensorFlow/Keras
scikit-learn
pandas
numpy
matplotlib








