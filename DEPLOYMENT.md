# Azure App Service Deployment Guide

## Prerequisites
- Azure account ([sign up for free](https://azure.microsoft.com/free/))
- Azure CLI installed
- Git repository (GitHub recommended)

## Option 1: Deploy via Azure CLI (Quickest)

### 1. Install Azure CLI
```powershell
winget install Microsoft.AzureCLI
```

### 2. Login to Azure
```powershell
az login
```

### 3. Create Resource Group
```powershell
az group create --name pulsar-polarimetry-rg --location eastus
```

### 4. Create App Service Plan
```powershell
az appservice plan create --name pulsar-plan --resource-group pulsar-polarimetry-rg --sku B1 --is-linux
```

### 5. Create Web App
```powershell
az webapp create --resource-group pulsar-polarimetry-rg --plan pulsar-plan --name pulsar-polarimetry-api --runtime "PYTHON:3.11"
```

### 6. Configure Startup Command
```powershell
az webapp config set --resource-group pulsar-polarimetry-rg --name pulsar-polarimetry-api --startup-file "startup.txt"
```

### 7. Deploy from Local Git
```powershell
# Enable local git deployment
az webapp deployment source config-local-git --name pulsar-polarimetry-api --resource-group pulsar-polarimetry-rg

# Get deployment credentials
az webapp deployment list-publishing-credentials --name pulsar-polarimetry-api --resource-group pulsar-polarimetry-rg

# Add Azure as git remote and push
git remote add azure <DEPLOYMENT_URL_FROM_ABOVE>
git push azure main
```

---

## Option 2: Deploy via GitHub Actions (Recommended for CI/CD)

### 1. Create Azure Web App (same as steps 3-6 above)

### 2. Get Publish Profile
```powershell
az webapp deployment list-publishing-profiles --resource-group pulsar-polarimetry-rg --name pulsar-polarimetry-api --xml
```

### 3. Add GitHub Secrets
Go to your GitHub repository:
- Settings → Secrets and variables → Actions
- Add two secrets:
  - `AZURE_WEBAPP_NAME`: `pulsar-polarimetry-api` (or your chosen name)
  - `AZURE_WEBAPP_PUBLISH_PROFILE`: Paste the XML output from step 2

### 4. Push to GitHub
The workflow in `.github/workflows/azure-deploy.yml` will automatically deploy when you push to main branch.

```powershell
git add .
git commit -m "Configure Azure deployment"
git push origin main
```

---

## Option 3: Deploy via VS Code Extension

### 1. Install Azure App Service Extension
- Open VS Code
- Go to Extensions (Ctrl+Shift+X)
- Search "Azure App Service"
- Install the extension

### 2. Sign in to Azure
- Click Azure icon in sidebar
- Click "Sign in to Azure"

### 3. Deploy
- Right-click your folder
- Select "Deploy to Web App"
- Follow the prompts

---

## Post-Deployment Configuration

### Set Environment Variables (if needed)
```powershell
az webapp config appsettings set --resource-group pulsar-polarimetry-rg --name pulsar-polarimetry-api --settings SETTING_NAME=value
```

### Enable HTTPS Only
```powershell
az webapp update --resource-group pulsar-polarimetry-rg --name pulsar-polarimetry-api --https-only true
```

### View Logs
```powershell
az webapp log tail --resource-group pulsar-polarimetry-rg --name pulsar-polarimetry-api
```

---

## Access Your API

Your API will be available at:
- `https://pulsar-polarimetry-api.azurewebsites.net`

Update your frontend's `VITE_API_BASE_URL` to this URL.

---

## Pricing

- **B1 Basic Plan**: ~$13/month
- **F1 Free Plan**: Available but has limitations (60 CPU minutes/day)

To use Free plan:
```powershell
az appservice plan create --name pulsar-plan --resource-group pulsar-polarimetry-rg --sku F1 --is-linux
```

---

## Troubleshooting

### Check deployment logs
```powershell
az webapp log deployment show --resource-group pulsar-polarimetry-rg --name pulsar-polarimetry-api
```

### SSH into container
```powershell
az webapp ssh --resource-group pulsar-polarimetry-rg --name pulsar-polarimetry-api
```

### Restart app
```powershell
az webapp restart --resource-group pulsar-polarimetry-rg --name pulsar-polarimetry-api
```
