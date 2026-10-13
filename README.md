# PneumoInsight

Explainable pneumonia detection from chest X-rays. A clinician uploads an X-ray and the app returns a diagnosis, a confidence tier and a Grad-CAM heatmap of the regions that drove the decision. Every result can be exported as a PDF report.

BSc Honours capstone, University of Zimbabwe, Department of Analytics and Informatics, 2026. This is an academic research prototype and not a medical device. It has not been clinically validated.

Portfolio: https://blessingchikumbirike.github.io/

![Result view with diagnosis, confidence tier and Grad-CAM overlay](docs/images/analysis.webp)

## Results

Three models were trained under the same conditions on the public Kermany paediatric chest X-ray dataset and compared on its held-out test set of 624 images (390 pneumonia and 234 normal). DenseNet121 scored highest on accuracy, recall, F1 and AUC, so it powers the application. It caught 367 of the 390 pneumonia cases in the test set.

| Model | Accuracy | Precision | Recall | F1 | AUC |
| --- | --- | --- | --- | --- | --- |
| Custom CNN | 89.10% | 0.9149 | 0.9103 | 0.9126 | 0.9399 |
| ResNet-50 | 78.04% | 0.7949 | 0.8744 | 0.8327 | 0.8523 |
| **DenseNet121** | **89.74%** | 0.8995 | **0.9410** | **0.9198** | **0.9589** |

![ROC curves for the three models](docs/images/roc.webp)

## Features

- Pneumonia or normal prediction with a confidence score
- Confidence tiers: High (90% and above), Moderate (70% to 89%) and Low (below 70%)
- Grad-CAM heatmap over the original X-ray, plus a plain-language explanation that names the lung regions with the strongest activation
- Role-based access for admins, doctors, radiologists and nurses. Admins manage users and nurses can view records but not run analyses
- Patient history with automatically generated patient IDs that records who ran each assessment and when
- PDF reports with ReportLab and Excel export of results

## How it works

Every upload is sanitized, resized to 224 by 224 and rescaled. DenseNet121 outputs a sigmoid score and anything above 0.5 is labelled pneumonia. The app then calculates the confidence, assigns a patient ID, generates the Grad-CAM heatmap and saves the record.

![Prediction flow](docs/images/flow.webp)

## Training

- **Data:** the 5,216 training images are split 80/20 into 4,173 for training and 1,043 for validation. The 624-image test set is never used for training or tuning.
- **Augmentation:** rotation up to 20 degrees, width and height shifts up to 20%, shear, zoom and horizontal flips.
- **Models:** DenseNet121 and ResNet-50 use frozen ImageNet weights with a new classification head. The custom CNN stacks four convolution blocks of 32, 64, 128 and 256 filters with batch normalization, max pooling and rising dropout, and uses class weights to offset the imbalance.
- **Optimization:** binary cross-entropy with Adam, up to 50 epochs, early stopping on validation loss and learning rate reduction on plateau.

![Custom CNN architecture](docs/images/cnn.webp)

The notebooks are in [`notebooks/`](notebooks): `CustomCNN.ipynb`, `ResNet50.ipynb` and `DenseNet121.ipynb`.

## Run locally

1. Download the dataset ([Kaggle copy of the Kermany data](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)) into a `chest_xray/` folder if you want to rerun the notebooks.
2. Install and start the app:

```powershell
python -m venv venv
venv\Scripts\activate            # macOS or Linux: source venv/bin/activate
pip install -r requirements.txt
$env:ADMIN_PASSWORD = "choose-a-strong-password"   # macOS or Linux: export ADMIN_PASSWORD=...
python app.py
```

3. Open http://127.0.0.1:5000 and sign in as `admin`. If you skip `ADMIN_PASSWORD`, a random password is printed in the console on first run. Create the other users from the admin panel.

Optional environment variables: `SECRET_KEY` keeps sessions valid across restarts and `FLASK_DEBUG=1` turns on Flask debug mode for local development only.

**Model weights.** The trained model (`best_densenet121_pneumonia.keras`) is not in this repository because of its size. Place it in the project root next to `app.py`. Without it the app starts but cannot classify. Contact me and I will share it, or rebuild it by running `notebooks/DenseNet121.ipynb`.

## Known limitations

- Trained only on paediatric radiographs from a single institution, so performance on adults or other scanners cannot be assumed. Local validation would be needed before any clinical use.
- The sigmoid probabilities are not calibrated, so confidence tiers are a guide and not true probabilities. Temperature scaling is future work.
- SQLite does not handle many simultaneous users. An institutional deployment would move to PostgreSQL.
- Uploads are not checked for file type and forms have no CSRF protection yet, so do not expose this app to the public internet as it stands.

## Screenshots

![Patient history with names blurred](docs/images/history.webp)

![PDF report](docs/images/report.webp)

![Login screen](docs/images/login.webp)

## Author

Blessing Chikumbirike, Harare, Zimbabwe.
[LinkedIn](https://www.linkedin.com/in/blessingchikumbirike/) | [Email](mailto:chikumbirikeblessing@gmail.com)
