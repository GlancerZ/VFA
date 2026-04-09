# Personalized VFA Loss Prediction and Dietary Recommendation Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This tool provides personalized dietary recommendations by analyzing baseline characteristics and early response data, predicting optimal strategies for visceral fat reduction. Based on a seven-arm randomized clinical trial with a within-subject control design, it offers evidence-based guidance to enhance metabolic health in young adults with normal weight through targeted interventions.

**Live Demo**: [http://huilab-vfa.com/](http://huilab-vfa.com/)

## Features

- 🔬 **RCT-Based**: Built on evidence from a rigorous randomized clinical trial, ensuring validated metabolic outcomes.
- 🧠 **LLM-Augmented**: Leveraging advanced Large Language Model technology to deliver tailored, data-driven recommendations.
- 🎯 **Personalized**: Providing individualized predictions based on unique participant profiles for optimized metabolic health outcomes.

## Research Foundation

This tool is built on a seven-arm randomized clinical trial that evaluated the effects of various dietary regimens, including calorie restriction and intermittent fasting, on visceral fat reduction in young adults with normal weight. The trial utilized a within-subject control design and gathered comprehensive data, including:

- Baseline participant characteristics
- Continuous monitoring data
- Dietary adherence
- Recovery-period health outcomes

Leveraging this data, the tool applies advanced predictive models to provide personalized, evidence-based dietary recommendations and predictions for optimizing metabolic health.

## Study Design

The study employed a within-subject control design, where participants were assigned to one of seven dietary groups. The interventions were implemented over a four-week period, with regular monitoring of multi-dimensional health outcomes.

### Participant Characteristics

The trial enrolled 84 healthy young adults, aged 18-35 years, including both male and female participants. Comprehensive baseline measurements were taken, including:

- Demographic factors
- Body composition
- Biochemical and metabolic factors

### Measurement Methods

- **VFA Measurement**: Using the InBody body composition analyzer
- **Metabolic Factors**: Evaluated with the VMAX metabolic cart
- **Biochemical Indicators**: Assessed through complete blood count and blood chemistry analysis
- **Follow-up Assessments**: Conducted at 1, 2, 3, 4, 5, 8, and 32 weeks

## Dietary Regimens

The tool evaluates seven different dietary approaches:

1. **Balanced Diet (100% Energy)**: Balanced Diet with 100% guideline intake
2. **TRF 16:8 (100% Energy)**: Time-Restricted Eating with 16:8 schedule and 100% caloric intake
3. **ADF (75% Energy)**: Alternate day fasting with 125% caloric intake on non-fasting days and 25% caloric intake on fasting days
4. **TRF 16:8 (75% Energy)**: Time-Restricted Eating with 16:8 schedule and 75% caloric intake
5. **IF 5:2 (75% Energy)**: Intermittent Fasting 5:2 with 95% caloric intake on non-fasting days and 25% caloric intake on two continuous fasting days
6. **CCR (75% Energy)**: Continuous Caloric Restriction with 75% caloric intake
7. **VLCD (45% Energy)**: Very-Low-Calorie Diet with 45% caloric intake

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/eqwwqe167/huilab-vfa.git
   cd huilab-vfa
   ```

2. Install the required dependencies:

   **Using pip:**
   ```bash
   pip install Flask==2.3.3 scikit-learn==1.6.1 pandas==2.2.3 numpy==1.26.4 joblib==1.3.2 torch==2.8.0
   ```

   **Using conda:**
   ```bash
   conda install flask scikit-learn pandas numpy joblib
   conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia  # For CUDA support
   # Or for CPU-only:
   # conda install pytorch cpuonly -c pytorch
   ```
3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to `http://localhost:5000`

### Model Artifacts

The app supports two modes:

- **Model mode**: place `task1/best_model.pkl` and `task2/best_model_task2.pkl`
  in the repository to run real predictions.
- **Demo mode**: if those files are missing, the app uses a deterministic fallback
  so the Code Ocean Capsule can still run end-to-end for environment validation.

The current published model bundles require `torch` and `scikit-learn 1.6.1`.

### Code Ocean Quick Start

1. Import the repository into a new Capsule.
2. Install the packages from `requirements.txt`.
3. Keep the included `run` file as the Reproducible Run entrypoint.
4. Launch a Reproducible Run and confirm outputs are written to `/results`.
5. Replace the demo fallback with the trained model files before final release.

## Usage

### Task 1: Optimal Dietary Regimen Prediction

1. Navigate to the "Prediction" page
2. Select "Task 1: Predict optimal dietary regimen"
3. Fill in the required personal information:
   - Demographic factors (Age, Sex, Birth Weight)
   - Body composition measurements
   - Metabolic indicators
   - Lifestyle factors
4. Submit the form to receive personalized dietary recommendations

### Task 2: Current Diet Continuation Prediction

1. Navigate to the "Prediction" page
2. Select "Task 2: Predict VFA change with current diet"
3. Fill in the same set of personal information
4. Submit the form to see predicted VFA changes if continuing current dietary habits

## Model Architecture

The tool uses machine learning models trained on comprehensive clinical trial data. The models incorporate multiple feature categories:

- **Demographic & Birth Factors**: Sex, Age, Birth Weight
- **Lifestyle Factors**: Nap Duration
- **Body Composition Factors**: TBW/FFM, Trunk FFM %, Leg BFM
- **Immune Cell Factors**: Lymphocyte %, Monocyte Count
- **Metabolic Factors**: Respiratory Quotient
- **Liver & Kidney Function Factors**: ALP, Urea

## Energy Calculation

### BMR Calculation (Mifflin-St Jeor Equation)

- **For Men**: BMR = 10 × weight(kg) + 6.25 × height(cm) - 5 × age(years) + 5
- **For Women**: BMR = 10 × weight(kg) + 6.25 × height(cm) - 5 × age(years) - 161

### Activity Factors for Total Daily Energy Expenditure (TDEE)

- **Sedentary** (little or no exercise): BMR × 1.2
- **Lightly Active** (light exercise): BMR × 1.35
- **Moderately Active** (moderate exercise): BMR × 1.5
- **Very Active** (hard exercise): BMR × 1.75
- **Super Active** (very hard exercise): BMR × 1.9

### Chinese Dietary Guidelines 2022

- **Adult Men**: ~2250 kcal daily
- **Adult Women**: ~1800 kcal daily
- **Macronutrient Distribution**:
  - **Carbohydrates**: 50-65% of total energy
  - **Proteins**: 10-15% of total energy
  - **Fats**: 20-30% of total energy

## Sample Meal Plans

### 1800 kcal Daily Meal Plan

- **Breakfast**: Boiled egg (50g), milk (150ml), pork bun (flour 50g, pork 30g)
- **Lunch**: Rice (100g), diced duck with mixed vegetables (duck breast 100g, cucumber 50g, sweet pepper 50g, soybean oil 5g), shredded pork in thick gravy (pork 20g, carrot 70g, soybean oil 5g), cabbage with glass noodles (cabbage 100g, bok choy 120g, glass noodles 30g, soybean oil 5g)
- **Dinner**: Rice (100g), blanched river prawns (120g), stir-fried pork slices with lettuce (pork 100g, lettuce 150g, soybean oil 10g), stir-fried greens with shiitake mushrooms (bok choy 100g, shiitake mushrooms 20g, soybean oil 5g)

### 450 kcal Modified Fasting Day Meal Plan

- Boiled purple sweet potato (150g), boiled egg (50g), ready-to-eat chicken breast (150g), stir-fried greens (bok choy 120g)

## File Structure

```
huilab-vfa-main/
├── app.py
├── predictor.py                # Shared prediction and fallback logic
├── requirements.txt
├── run                         # Code Ocean reproducible run entrypoint
├── CODE_OCEAN.md               # Capsule setup checklist
├── sample_inputs/
│   ├── task1_form.json
│   └── task2_form.json
├── scripts/
│   └── generate_codeocean_results.py
├── templates/
│   ├── index.html
│   ├── about.html
│   ├── predict.html
│   ├── result_T1.html
│   └── result_T2.html
├── static/
│   ├── css/
│   └── js/
├── task1/
│   ├── README.md
│   ├── best_model.pkl          # Add before final release
│   └── infer_from_pkl.py
└── task2/
    ├── README.md
    ├── best_model_task2.pkl    # Add before final release
    └── infer_from_pkl.py
```

## Dependencies

The following Python packages are required to run this application:
- Flask==2.3.3
- scikit-learn==1.6.1
- pandas==2.2.3
- numpy==1.26.4
- joblib==1.3.2
- torch==2.8.0

If you replace the current model bundles with new artifacts, adjust these package
versions as needed to match the training environment.

You can install these packages using either pip or conda as shown in the Installation section.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{vfa_prediction_tool,
  title={Personalized VFA Loss Prediction and Dietary Recommendation Tool},
  author={[Author Names]},
  year={2025},
  url={http://huilab-vfa.com/}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Research participants who contributed to the clinical trial data
- Research team members who conducted the study
- Healthcare professionals who provided medical supervision
- Technical team who developed the prediction models

## Contact

For questions, suggestions, or collaborations, please contact [184514@shsmu.edu.cn].

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
