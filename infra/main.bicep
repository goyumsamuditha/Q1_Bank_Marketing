// main.bicep
// ----------
// Deploys the Azure resources that sit UNDERNEATH Databricks: the
// storage account (ADLS Gen2), the VNet-injected Databricks workspace
// itself, and a Key Vault for secrets. Databricks then handles compute,
// Delta tables, MLflow, and Model Serving on top of these.
//
// Deploy with:
//   az group create -n rg-bank-mlops -l uksouth
//   az deployment group create -g rg-bank-mlops -f infra/main.bicep -p infra/parameters.json

@description('Environment name - used to suffix every resource name, so dev/staging/prod never collide.')
@allowed(['dev', 'staging', 'prod'])
param environmentName string = 'dev'

@description('Azure region for all resources.')
param location string = resourceGroup().location

var resourceSuffix = '${environmentName}-bankmlops'
var storageAccountName = toLower(replace('st${resourceSuffix}', '-', ''))
var databricksWorkspaceName = 'dbw-${resourceSuffix}'
var keyVaultName = 'kv-${resourceSuffix}'
var vnetName = 'vnet-${resourceSuffix}'

// --- Networking: Databricks needs its own VNet with two dedicated subnets ---
resource vnet 'Microsoft.Network/virtualNetworks@2023-09-01' = {
  name: vnetName
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: ['10.0.0.0/16']
    }
    subnets: [
      {
        name: 'public-subnet'
        properties: {
          addressPrefix: '10.0.1.0/24'
          delegations: [
            {
              name: 'databricks-delegation'
              properties: { serviceName: 'Microsoft.Databricks/workspaces' }
            }
          ]
        }
      }
      {
        name: 'private-subnet'
        properties: {
          addressPrefix: '10.0.2.0/24'
          delegations: [
            {
              name: 'databricks-delegation'
              properties: { serviceName: 'Microsoft.Databricks/workspaces' }
            }
          ]
        }
      }
    ]
  }
}

// --- Storage: ADLS Gen2 landing zone for the raw CSV before it reaches Delta ---
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    isHnsEnabled: true // enables the hierarchical namespace = ADLS Gen2, not plain blob storage
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

resource landingContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: '${storageAccount.name}/default/landing'
  properties: {}
}

// --- Key Vault: backs the Databricks secret scope (connection strings, tokens) ---
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    enableRbacAuthorization: true
    networkAcls: {
      defaultAction: 'Deny'
      bypass: 'AzureServices'
    }
  }
}

// --- Azure Databricks workspace, VNet-injected for private networking ---
resource databricksWorkspace 'Microsoft.Databricks/workspaces@2023-02-01' = {
  name: databricksWorkspaceName
  location: location
  sku: { name: 'premium' } // premium tier required for Unity Catalog + RBAC
  properties: {
    managedResourceGroupId: subscriptionResourceId(
      'Microsoft.Resources/resourceGroups',
      'rg-${resourceSuffix}-managed'
    )
    parameters: {
      customVirtualNetworkId: { value: vnet.id }
      customPublicSubnetName: { value: 'public-subnet' }
      customPrivateSubnetName: { value: 'private-subnet' }
    }
  }
}

output storageAccountName string = storageAccount.name
output databricksWorkspaceUrl string = databricksWorkspace.properties.workspaceUrl
output keyVaultName string = keyVault.name
