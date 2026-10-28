# Dataset: PaySim (Synthetic Financial Datasets For Fraud Detection)

## Verified from public descriptions (web search, 28 Oct 2026)
- Synthetic mobile-money transactions from the PaySim simulator, built on aggregated data from
  one month of real logs from a mobile-money service in an African country.
- 744 time steps, 1 step = 1 hour (30 days).
- Types: CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER.
- Columns described: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest,
  oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud.
- Fraud behaviour in the simulation: take over accounts, transfer funds out, then cash out.
- isFlaggedFraud flags a single transfer above 200,000.
- Balance info is not available for merchant (M-prefixed) destinations.
- **Leakage warning from the dataset description:** fraudulent transactions are cancelled, so
  `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` must not be used for fraud
  detection. Our EDA tests this with a leakage check; the model decision is made after seeing real output.
- Paper: Lopez-Rojas, Elmir, Axelsson, "PaySim: A financial mobile money simulator for fraud
  detection", EMSS 2016.
- One third-party mirror lists 6,362,620 rows and 11 columns. **Not yet confirmed on the real file.**

## NOT yet verified (fill in after download)
- [ ] Licence (read it on the Kaggle page and copy it here)
- [ ] Actual row count, columns, fraud rate (from `docs/eda/dataset_inspection.txt`)
- [ ] File SHA-256

## Limitations for this project
- No device, card, or location fields exist. Do not invent them. Graph work is account-to-account only.
- Synthetic data: results show behaviour on simulated fraud, not real-world performance.

## How to obtain
1. Free Kaggle account, open the PaySim dataset page (search "Synthetic Financial Datasets For Fraud Detection").
2. Download, unzip, place the CSV in `ml/data/raw/`. It stays out of Git.
