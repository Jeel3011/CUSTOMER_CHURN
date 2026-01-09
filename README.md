# Customer Churn Prediction System

A machine learning system that predicts which customers are likely to leave. Uses XGBoost for predictions and provides both a REST API and a web dashboard.

Status: LIVE - Deployed on AWS EC2

- Dashboard: http://13.51.45.223:8501
- API: http://13.51.45.223:8000
- Health: http://13.51.45.223:8000/health

## What It Does

This system takes customer data and predicts the probability they will churn. It categorizes customers into risk levels:
- Low risk (0-40%)
- Medium risk (40-70%)
- High risk (70-100%)

Use it to identify customers before they leave and prioritize retention efforts.

## Quick Start

Clone and run with Docker:

```bash
git clone https://github.com/Jeel3011/customer_churn.git
cd customer_churn
docker-compose up -d
```

Access:
- Dashboard: http://localhost:8501
- API: http://localhost:8000

## API Examples

Check health:
```bash
curl http://localhost:8000/health
```

Single customer prediction:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "C001",
    "tenure_months": 24,
    "monthly_usage": 85.5,
    "subscription_plan": "premium",
    "monthly_revenue": 99.99,
    "support_tickets": 3,
    "last_login_days": 2,
    "payment_delay": 0
  }'
```

Batch prediction (upload CSV):
```bash
curl -X POST http://localhost:8000/predict_csv \
  -F "file=@customers.csv"
```

Your CSV needs these columns:
```
customer_id,tenure_months,monthly_usage,subscription_plan,monthly_revenue,support_tickets,last_login_days,payment_delay
```

## Dashboard

Access at http://localhost:8501

Upload CSV files to get predictions for multiple customers. View KPIs and download results.

## Deployment on AWS EC2

```bash
ssh -i your_key.pem ec2-user@your_instance_ip
sudo yum install docker -y
sudo systemctl start docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone and run
git clone https://github.com/Jeel3011/customer_churn.git
cd customer_churn
docker-compose up -d

curl http://localhost:8000/health
```

## Troubleshooting

Containers not running:
```bash
docker-compose ps
docker-compose restart
```

Check logs:
```bash
docker logs customer_churn_api
```

Port 8000 already in use:
```bash
lsof -i :8000
kill -9 <PID>
```

Model file missing:
```bash
ls -la artifacts/model.pkl
```

## Project Structure

```
customer_churn/
├── api/              FastAPI endpoints
├── ui/               Streamlit dashboard
├── src/              ML pipeline
├── artifacts/        Model files
├── logs/             Application logs
└── docker-compose.yml
```

## Model Details

- Type: XGBoost classifier
- Accuracy: 85%
- Input features: 8 customer attributes
- Output: Churn probability (0-1)

Performance:
- Single prediction: 50ms
- 100 customers: 200ms
- 1000 customers: 1.5s

## Input Fields

- customer_id: Unique customer ID
- tenure_months: How long they have been customer (0-72 months)
- monthly_usage: Service usage percentage (0-100%)
- subscription_plan: Type of plan (basic/standard/premium)
- monthly_revenue: Revenue from customer per month in USD
- support_tickets: Number of support tickets opened
- last_login_days: Days since they last logged in
- payment_delay: Days they delayed payment

## Python Example

```python
import requests

API_URL = "http://localhost:8000"

customer = {
    "customer_id": "C001",
    "tenure_months": 24,
    "monthly_usage": 85.5,
    "subscription_plan": "premium",
    "monthly_revenue": 99.99,
    "support_tickets": 3,
    "last_login_days": 2,
    "payment_delay": 0
}

response = requests.post(f"{API_URL}/predict", json=customer)
result = response.json()
print(f"Risk: {result['risk_level']}")
print(f"Probability: {result['churn_probability']:.0%}")
```

## Requirements

- Docker 20.10+
- Docker Compose 1.29+
- Git
- Python 3.10+ (for local development)

## License

Private project

---

Made by Jeel Thummar | GitHub: github.com/Jeel3011
