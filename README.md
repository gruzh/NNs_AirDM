# Using neural networks as an alternative to air dispersion modeling in environmental impact assessment

## Authors
[Mateo Concha](https://www.researchgate.net/profile/Mateo-Concha)<sup>1</sup>, &nbsp; 
[Gonzalo A. Ruz](https://scholar.google.cl/citations?user=jkovdhYAAAAJ&hl=en)<sup>1,2,3,4</sup>, &nbsp;

<sup>1</sup> Facultad de Ingeniería y Ciencias, Universidad Adolfo Ibáñez, Santiago, Chile. <br>
<sup>2</sup> Millennium Nucleus for Social Data Science (SODAS), Santiago, Chile. <br>
<sup>3</sup> Center of Applied Ecology and Sustainability (CAPES), Santiago, Chile. <br>
<sup>4</sup> Data Observatory Foundation, Santiago, Chile. <br>

## Description
This project explores the use of neural networks (NNs) as an alternative to traditional air dispersion models for environmental impact assessments. We compare the CALPUFF dispersion model against neural network-based predictions of SO₂ concentrations in the Industrial Bay of Mejillones, Chile.

## Dataset Information
- **Location**: `Dataset/` folder
- **Content**: Emission data, meteorological data, and stack parameters from official, private, and public sources related to the Mejillones industrial area.
- **Format**: CSV files

## Code Information
- **Location**: `Script/` folder
- **Main script**:
  - `main.py`: Data preprocessing, neural network model building and training, and model evaluation on FFEE and CV datasets.

## Usage Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/gruzh/NNs_AirDM.git
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
   (Alternatively, install manually: `pandas`, `matplotlib`, `scikit-learn`, `tensorflow`)

3. Run the main script:
   ```bash
   python Script/main.py
   ```

## Requirements
* Python >= 3.10.11
* Pandas >= 1.5.3
* Matplotlib >= 3.7.1
* Scikit-learn >= 1.2.2
* Tensorflow >= 2.13.1

## Methodology
<!-- - **Data Collection**: Emission, meteorological, and operational data were collected from various sources. -->
- **Data Preprocessing**: Normalization, and preparation for modeling.
- **Model Training**: Neural networks were trained on historical data to predict SO₂ concentrations.
- **Comparison**: Model outputs were compared against the SO₂ from the FFEE and CV datasets using performance metrics.

## Citations
If you use this code or data, please cite:
> Mateo Concha, Gonzalo A. Ruz. *Using neural networks as an alternative to air dispersion modeling in environmental impact assessment*. (Submitted to PeerJ Computer Science, 2025)

## License
This project is licensed under the [MIT License](LICENSE).
 
## Contribution Guidelines
Contributions are welcome! Feel free to submit pull requests or suggest improvements via issues.
