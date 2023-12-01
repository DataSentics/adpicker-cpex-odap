# Databricks notebook source
# MAGIC %md ####Check what's connected to this workspace

# COMMAND ----------

display(dbutils.fs.mounts())

# COMMAND ----------

# MAGIC %md Set(up) environment

# COMMAND ----------

dbutils.widgets.dropdown("ENV", "dev", ["dev", "prod"])

# COMMAND ----------

ENV = dbutils.widgets.get("ENV")

# COMMAND ----------

# MAGIC %md #####Check scopes and secrets

# COMMAND ----------

dbutils.secrets.listScopes()

# COMMAND ----------

dbutils.secrets.list(f"adls-gen2-{ENV}")

# COMMAND ----------

# MAGIC %md ###Working with `cpexblobstoragedev` or `cpexblobstorageprod` 

# COMMAND ----------

# MAGIC %md SETUP universal variables

# COMMAND ----------

adlsAccountName = f"cpexstorageblob{ENV}"
scope_name = f"adls-gen2-{ENV}"

# Application (Client) ID
applicationId = dbutils.secrets.get(scope=scope_name,key=f"cpex-adpicker-{ENV}-adls-gen2-clientid") 
# Application (Client) Secret Key
authenticationKey = dbutils.secrets.get(scope=scope_name,key=f"cpex-adpicker-{ENV}-adls-gen2-clientsecret") 
# Directory (Tenant) ID
tenandId = dbutils.secrets.get(scope=scope_name,key=f"cpex-adpicker-{ENV}-adls-gen2-tenantid")
endpoint = "https://login.microsoftonline.com/" + tenandId + "/oauth2/token"

# Connecting using Service Principal secrets and OAuth
configs = {"fs.azure.account.auth.type": "OAuth",
           "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
           "fs.azure.account.oauth2.client.id": applicationId,
           "fs.azure.account.oauth2.client.secret": authenticationKey,
           "fs.azure.account.oauth2.client.endpoint": endpoint}

# COMMAND ----------

# MAGIC %md Mounting function

# COMMAND ----------

def mount_storage(container_name: str, mount_point: str = None):
    if mount_point is None:
        mount_point = f"/mnt/aam-cpex-{ENV}/{container_name}/"

    # path that we are linking
    source = "abfss://" + container_name + "@" + adlsAccountName + ".dfs.core.windows.net/"
    
    # Mount ADLS Storage to DBFS only if the directory is not already mounted
    if not any(mount.mountPoint == mount_point for mount in dbutils.fs.mounts()):
        dbutils.fs.mount(
            source = source,
            mount_point = mount_point,
            extra_configs = configs)
    else:
        print(f"Container {container_name} is already mounted")

# COMMAND ----------

# MAGIC %md ####Raw

# COMMAND ----------

adlsContainerName = "raw"

# COMMAND ----------

# unmount, as we no longer need old link
dbutils.fs.unmount(f"/mnt/aam-cpex-{ENV}/{adlsContainerName}")

# COMMAND ----------

# mounting itself
mount_storage(adlsContainerName)

# COMMAND ----------

# check if everything works just fine
dbutils.fs.ls(f"dbfs:/mnt/aam-cpex-{ENV}/{adlsContainerName}")

# COMMAND ----------

# MAGIC %md ####Bronze

# COMMAND ----------

adlsContainerName = "bronze"

# COMMAND ----------

dbutils.fs.unmount(f"/mnt/aam-cpex-{ENV}/{adlsContainerName}")

# COMMAND ----------

mount_storage(adlsContainerName)

# COMMAND ----------

dbutils.fs.ls(f"dbfs:/mnt/aam-cpex-{ENV}/{adlsContainerName}")

# COMMAND ----------

# MAGIC %md ####Silver

# COMMAND ----------

adlsContainerName = "silver"

# COMMAND ----------

dbutils.fs.unmount(f"/mnt/aam-cpex-{ENV}/{adlsContainerName}")

# COMMAND ----------

mount_storage(adlsContainerName)

# COMMAND ----------

dbutils.fs.ls(f"dbfs:/mnt/aam-cpex-{ENV}/{adlsContainerName}")

# COMMAND ----------

# MAGIC %md ####Gold

# COMMAND ----------

adlsContainerName = "gold"

# COMMAND ----------

dbutils.fs.unmount(f"/mnt/aam-cpex-{ENV}/{adlsContainerName}")

# COMMAND ----------

mount_storage(adlsContainerName)

# COMMAND ----------

dbutils.fs.ls(f"dbfs:/mnt/aam-cpex-{ENV}/{adlsContainerName}")

# COMMAND ----------

# MAGIC %md ####Scm-releases

# COMMAND ----------

adlsContainerName = "scm-releases"

# COMMAND ----------

dbutils.fs.unmount(f"/mnt/aam-cpex-{ENV}/{adlsContainerName}")

# COMMAND ----------

mount_storage(adlsContainerName)

# COMMAND ----------

dbutils.fs.ls(f"dbfs:/mnt/aam-cpex-{ENV}/{adlsContainerName}")

# COMMAND ----------

# MAGIC %md ####Models

# COMMAND ----------

adlsContainerName = "models"

# COMMAND ----------

dbutils.fs.unmount(f"/mnt/aam-cpex-{ENV}/{adlsContainerName}")

# COMMAND ----------

mount_storage(adlsContainerName)

# COMMAND ----------

dbutils.fs.ls(f"dbfs:/mnt/aam-cpex-{ENV}/{adlsContainerName}")

# COMMAND ----------

# MAGIC %md ####Piano

# COMMAND ----------

adlsContainerName = "piano"

# COMMAND ----------

dbutils.fs.unmount(f"/mnt/{adlsContainerName}")

# COMMAND ----------

mount_storage(adlsContainerName, "/mnt/piano")

# COMMAND ----------

dbutils.fs.ls(f"dbfs:/mnt/{adlsContainerName}")
