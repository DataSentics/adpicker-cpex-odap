# Databricks notebook source
# MAGIC %md
# MAGIC # Segment inspection
# MAGIC This notebook serves to inspect a CPEx segment/audience based on interests. It currently works for simple segments based on a single interest and a certainty value.

# COMMAND ----------

# MAGIC %run ../../app/bootstrap

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import logging
from logging import Logger

import matplotlib.pyplot as plt
import pandas as pd
import pyspark.pandas as ps
import pyspark.sql.functions as F
import seaborn as sns
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

import daipe as dp
from adpickercpex.lib.FeatureStoreTimestampGetter import FeatureStoreTimestampGetter
from adpickercpex.lib.display_result import display_result

# COMMAND ----------

a_logger = logging.getLogger("py4j")
a_logger.setLevel(logging.ERROR)

a_logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Init and widgets
# MAGIC Fill the interest name and certainty value into the widgets. For the interest name, fill *only* the name of the interest and *replace* spaces with underscores (e.g. Electric Vehicle -> electric_vehicle).
# MAGIC
# MAGIC Choose a valid timestamp for the *Feature Store* date widget on which a successful run of the DEV pipeline took place.

# COMMAND ----------

user_entity = dp.fs.get_entity()
feature = dp.fs.feature_decorator_factory.create(user_entity)

# COMMAND ----------

@dp.notebook_function()
def create_widgets(widgets: dp.Widgets):

    widgets.add_text(name="interest_name",
                   default_value="university_students",
                   label="Interest name")
    
    widgets.add_text(name="certainty",
                   default_value="0.75",
                   label="Certainty value")
    
    widgets.add_text(name="timestamp",
                   default_value="2023-05-11",
                   label="Feature Store date")
    
    widgets.add_select(name="url",
                   choices=["True", "False"],
                   default_value="True",
                   label="URLs")
    
    widgets.add_select(name="correlation",
                   choices=["True", "False"],
                   default_value="True",
                   label="Correlations")

    widgets.add_select(name="hits",
                   choices=["True", "False"],
                   default_value="True",
                   label="Hits")
    
    widgets.add_select(name="socdemo",
                   choices=["True", "False"],
                   default_value="True",
                   label="Soc-Demo")
    


# COMMAND ----------

print(f"Inserted interest: {dbutils.widgets.get('interest_name')}")
print(f"Inserted certainty: {dbutils.widgets.get('certainty')}")
print(f"Inserted timestamp: {dbutils.widgets.get('timestamp')}")

# COMMAND ----------

@dp.notebook_function(dp.read_delta("%interests.delta_path%"))
def get_interests(df_interests):
    lst = df_interests.select("subinterest_fs").collect()
    return [row.subinterest_fs for row in lst]

# COMMAND ----------

@dp.notebook_function()
def get_interests_in_fs(fs: dp.fs.FeatureStore):
    lst = (
        fs.get_metadata()
        .filter(F.col("category") == "digital_interest")
        .select("feature")
        .collect()
    )
    return [row.feature for row in lst]

# COMMAND ----------

print("Intersets defined in %interests.delta_path% but not available in FS: ")
for interest in (set(get_interests.result) - set(get_interests_in_fs.result)):
    print(f"\t{interest}")

# COMMAND ----------

@dp.transformation(user_entity, get_interests_in_fs, dp.get_widget_value("timestamp"))
def get_fs_all(entity, interests_list, timestamp, fs: FeatureStoreTimestampGetter, logger: Logger,):
    df_fs = fs.get_for_timestamp(
        entity_name=entity.name,
        timestamp=timestamp,
        skip_incomplete_rows=True,
        features=interests_list,
    )
    logger.info(f"Feature store rows: {df_fs.count()}")
    return df_fs

# COMMAND ----------

@dp.transformation(
    user_entity,
    get_interests_in_fs,
    dp.get_widget_value("timestamp"),
    dp.get_widget_value("interest_name"),
    dp.get_widget_value("certainty"), display=False
)
def get_fs_filtered(
    entity,
    interests_list,
    timestamp,
    interest_widget,
    certainty_widget,
    fs: FeatureStoreTimestampGetter,
    logger: Logger,
):
    interest_col = f"ad_interest_affinity_{interest_widget.lower()}"
    certainty = float(certainty_widget)

    df_picked = fs.get_for_timestamp(
        entity_name=entity.name,
        timestamp=timestamp,
        skip_incomplete_rows=True,
        features=interests_list,
    ).filter(F.col(interest_col) >= certainty)

    picked_count = df_picked.count()
    logger.info(f"Feature store rows after filtering: {df_picked.count()}")
    if picked_count < 500:
        raise ValueError(f"Choosen combination of parameters has only {picked_count} rows, which is probably a mistake?")
    return df_picked

# COMMAND ----------

# MAGIC %md
# MAGIC #### Interest means
# MAGIC Calculates:
# MAGIC - Mean affinity of all interests
# MAGIC - Mean affinity of all insterests where affinity of chosen interest is greater than chosen certainty value
# MAGIC - Difference between these two means - this metric favours interests that are popular in general
# MAGIC - **Proportion** (divition) of these two means - this metric favours interests that are not popular in general
# MAGIC
# MAGIC Interests with highest proportion and difference are connected to chosen interest, proportion metric is a bit more informative.

# COMMAND ----------

@dp.transformation(get_fs_all, get_fs_filtered)
@display_result()
def join_datasets(df_all, df_filtered):

    cols_list = df_all.columns
    cols_list.remove('user_id')
    cols_list.remove('timestamp')

    mean_df_all = df_all.select(*[F.mean(c).alias(c) for c in cols_list]).toPandas().T.reset_index()
    mean_df_all.columns = ['interest', 'mean']

    mean_df_filtered = df_filtered.select(*[F.mean(c).alias(c) for c in cols_list]).toPandas().T.reset_index()
    mean_df_filtered.columns = ['interest', 'mean_in_segment']

    df = mean_df_all.merge(mean_df_filtered, how="left", on="interest")
    df["difference_in_means"] = df['mean_in_segment'] - df['mean']
    df["proportion_of_means"] = df['mean_in_segment'] / df['mean']

    df['interest'] = df['interest'].str.replace(r'ad_interest_affinity_', '', regex=False)
    df = spark.createDataFrame(df)
    df = df.select('interest', *[F.round(c, 2).alias(c) for c in ['mean', 'mean_in_segment', "difference_in_means", "proportion_of_means"]])
    return  df

# COMMAND ----------

@dp.notebook_function(join_datasets)
def plot_join_datasets(df):
    means_df = df.toPandas()
    _, faxs = plt.subplots(1, 2, figsize=(16, 8))
    means_df.sort_values(by=['proportion_of_means'], ascending=False, inplace=True)
    sns.barplot(data=means_df.head(30), x="proportion_of_means", y="interest", ax=faxs[0])
    faxs[0].set(xlabel="", ylabel="", title="Proportion in means all vs. segment")

    means_df.sort_values(by=['difference_in_means'], ascending=False, inplace=True)
    sns.barplot(data=means_df.head(30), x="difference_in_means", y="interest", ax=faxs[1])
    faxs[1].set(xlabel="", ylabel="", title="Difference of means all vs. segment")
    plt.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC #### URLs preparation:

# COMMAND ----------

@dp.notebook_function(dp.read_table("silver.income_url_coeffs"))
def get_url_list_cleaned(df):
    domains = df.select("domain").collect()
    return [row.domain.replace(".", "_") for row in domains]

# COMMAND ----------

@dp.transformation(dp.read_table("silver.income_url_coeffs"))
@display_result()
def show_df(df):
    return df.limit(1000)

# COMMAND ----------

@dp.transformation(
    user_entity,
    dp.read_table("silver.income_url_scores"),
    dp.get_widget_value("timestamp"),
    dp.get_widget_value("interest_name"),
    display=False)
def join_fs_with_urls(
    entity,
    df_url,
    timestamp,
    interest_widget,
    fs: FeatureStoreTimestampGetter,
):
    interest_col = f"ad_interest_affinity_{interest_widget.lower()}"

    df_fs = fs.get_for_timestamp(
        entity_name=entity.name,
        timestamp=timestamp,
        features=[interest_col],
        skip_incomplete_rows=True,
    )

    return df_fs.join(df_url, on="user_id").select(
        "user_id", interest_col, "collected_urls"
    )

# COMMAND ----------

@dp.transformation(join_fs_with_urls,
                   get_url_list_cleaned,
                   dp.get_widget_value("interest_name"),
                   display=False)
def get_domain_flags(df_joined, domains_cleaned, interest_widget):
    interest_col = f"ad_interest_affinity_{interest_widget.lower()}"

    return df_joined.select(
        "user_id",
        interest_col,
        *[
            (F.array_contains("collected_urls", domain.replace("_", ".")))
            .cast("int")
            .alias(f"{domain}_flag")
            for domain in domains_cleaned
        ],
    )

# COMMAND ----------

@dp.transformation(
    get_domain_flags,
    get_url_list_cleaned,
    dp.get_widget_value("interest_name"),
    dp.get_widget_value("certainty"), display=False)
def calculate_url_percentages(df_domain_flags, domains_cleaned, interest_widget, certainty_widget):

    interest_col = f"ad_interest_affinity_{interest_widget.lower()}"
    certainty = float(certainty_widget)

    df_ps = df_domain_flags.select(
        *[
            (F.round(100 * F.mean(f"{domain}_flag"), 2)).alias(domain)
            for domain in domains_cleaned
        ]
    )

    df_melted = df_ps.to_pandas_on_spark().melt(
        id_vars=[],
        value_vars=domains_cleaned,
        value_name="Percentage_all",
        var_name="domain",
    )

    sub_df_melted = ps.DataFrame(
        df_domain_flags.filter(F.col(interest_col) >= certainty).select(
            *[
                (100 * F.mean(f"{domain}_flag")).alias(domain)
                for domain in domains_cleaned
            ]
        )
    ).melt(
        id_vars=[],
        value_vars=domains_cleaned,
        value_name="Percentage_subset",
        var_name="domain",
    )
    df_melted = df_melted.merge(sub_df_melted, on="domain", suffixes=("", ""))
    return df_melted.to_spark()

# COMMAND ----------

@dp.transformation(calculate_url_percentages)
def calculate_url_diffs_and_proportion(df):
    res = df.select(
        "domain",
        "Percentage_all",
        F.round(F.col("Percentage_subset"), 2).alias("Percentage_subset"),
        (F.round(F.col("Percentage_subset") - F.col("Percentage_all"), 2)).alias(
            "Difference"
        ),
        (F.round(F.col("Percentage_subset") / F.col("Percentage_all"), 2)).alias(
            "Proportion"
        ),
    )
    return res

# COMMAND ----------

# MAGIC %md
# MAGIC #### URLs:
# MAGIC Compare visit ratio between all users and users in chosen subset (for each "relevant" site URL). 
# MAGIC
# MAGIC Calculates
# MAGIC - percentage of users who visit given domain (together they don't give 100% because users can visit many different domains during one session)
# MAGIC - percentage of users, with chosen interest affinity greater than chosen certainty value, who visit given domain
# MAGIC - difference between these two percentages - favors popular domains
# MAGIC - **proportion** (division) between these two percentage - favors unpopular domains

# COMMAND ----------

@dp.transformation(calculate_url_diffs_and_proportion, display=False)
@display_result(display=dbutils.widgets.get('url') == "True")
def url_diffs_and_proportion_plot(df):
    return df

# COMMAND ----------

@dp.notebook_function(calculate_url_diffs_and_proportion)
def plot_url_diffs(df):
    if dbutils.widgets.get('url') == "False":
        return
    
    means_df = df.toPandas()
    _, faxs = plt.subplots(1, 2, figsize=(16, 8))
    means_df.sort_values(by=['Proportion'], ascending=False, inplace=True)
    sns.barplot(data=means_df.head(30), x="Proportion", y="domain", ax=faxs[0])
    faxs[0].set(xlabel="", ylabel="", title="Proportion in means all vs. segment")

    means_df.sort_values(by=['Difference'], ascending=False, inplace=True)
    sns.barplot(data=means_df.head(30), x="Difference", y="domain", ax=faxs[1])
    faxs[1].set(xlabel="", ylabel="", title="Difference of means all vs. segment")
    plt.tight_layout()

    return 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Correlations
# MAGIC Calculate Pearson correlation matrix (faster) and show correlations between affinities of chosen interest and all other defined interests.
# MAGIC
# MAGIC The graph shows the same correlations as the table.

# COMMAND ----------

@dp.notebook_function(get_fs_all)
def get_corrs(df):
    if dbutils.widgets.get('correlation') == "False":
        return
    interest_cols = [col for col in df.columns if col not in ["user_id", "timestamp"]]
    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=interest_cols, outputCol=vector_col)
    df_vector = assembler.transform(df.select(interest_cols)).select(vector_col)
    matrix = Correlation.corr(df_vector, vector_col)
    res = pd.DataFrame(matrix.collect()[0][0].toArray().tolist(), index=interest_cols, columns=interest_cols)
    return res

# COMMAND ----------

if dbutils.widgets.get('correlation') == "True":
    spark.createDataFrame(pd.DataFrame(get_corrs.result.loc[:, f"ad_interest_affinity_{dbutils.widgets.get('interest_name').lower()}"]).reset_index()).display()

# COMMAND ----------

# DBTITLE 0,The exactly same correlations but in graph:
if dbutils.widgets.get('correlation') == "True":
    plt.figure(figsize=(10,8))
    # pylint: disable=unsubscriptable-object
    # pylint: disable=unsupported-assignment-operation
    interest_corr: pd.DataFrame = pd.DataFrame(get_corrs.result.loc[:, f"ad_interest_affinity_{dbutils.widgets.get('interest_name').lower()}"]).reset_index()
    interest_corr.columns = ["interest", "correlation"]
    interest_corr['interest'] = interest_corr['interest'].str.replace(r'ad_interest_affinity_', '', regex=False)
    interest_corr.sort_values(by=['correlation'], ascending=False, inplace=True)

    sns.barplot(x=interest_corr["correlation"].head(30), y=interest_corr["interest"].head(30))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Most hit tokens 
# MAGIC List the most frequently hit tokens in the chosen interest.

# COMMAND ----------

@dp.notebook_function(
    dp.read_table("silver.sdm_tokenized_domains"),
    get_fs_filtered,
    dp.read_delta("%interests.delta_path%"),
    dp.get_widget_value("interest_name"),
    dp.get_widget_value("hits"),
)
def get_segment_token_hits(df_sdm_tokens, fs_filtered, interests, interest_widget, hits_widget):
    if hits_widget == "False":
        return
    interest_col = f"ad_interest_affinity_{interest_widget.lower()}"
    keywords_list = (
        interests.filter(F.col("subinterest_fs") == interest_col)
        .select("keywords")
        .collect()[0][0])
    
    df_joined = fs_filtered.select("user_id", "timestamp").join(df_sdm_tokens, on="user_id")
    
    return (df_joined.groupby("TOKEN")
        .agg(F.count(F.lit(1)).alias("token_hits"))
        .sort(F.col("token_hits").desc())
        .filter(F.col("TOKEN").isin(keywords_list))).toPandas()

if dbutils.widgets.get('hits') == "True":
    spark.createDataFrame(get_segment_token_hits.result).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Socdemo interactions
# MAGIC Horizontal line in scatterplots is given by linear regression coefficients, no slope (line is horizontal) means that there is no linear relationship between interest affinity and soc-demo characteristic.
# MAGIC
# MAGIC Soc-demo **targets** are used so the plot should be reliable.
# MAGIC
# MAGIC
# MAGIC 1. Scatterplot of chosen interest affinity versus age (we look for a slope)
# MAGIC 2. Scatterplot of chosen interest affinity versus gender (we look for a slope)
# MAGIC 3. Histogram of chosen interest affinity stratified by gender (we look for bars where frequency of genders differ)

# COMMAND ----------

@dp.transformation(get_fs_all, 
                   dp.read_table("silver.sdm_sociodemo_targets"), 
                   dp.get_widget_value("interest_name"), 
                   dp.get_widget_value("socdemo"), 
                   display=False)
def socdemo_interest(fs_all, df_socdemo, interest_widget, socdemo_widget):
    if not socdemo_widget:
        return
    interest_col = f"ad_interest_affinity_{interest_widget.lower()}"
    df_socdemo = df_socdemo.select("AGE", "GENDER", "user_id").filter(F.col("AGE")!="unknown").filter(F.col("GENDER")!="unknown").dropDuplicates()
    res = fs_all.select("user_id", interest_col).join(df_socdemo, on="user_id")
    return res

if dbutils.widgets.get('socdemo') == "True":
    socdemo_interest_df = socdemo_interest.result.toPandas().sample(5000)
    socdemo_interest_df["GENDER"] = pd.to_numeric(socdemo_interest_df["GENDER"])
    socdemo_interest_df["AGE"] = pd.to_numeric(socdemo_interest_df["AGE"])
    fig, axs = plt.subplots(1, 3, figsize=(21, 8))
    sns.regplot(data=socdemo_interest_df, x=f"ad_interest_affinity_{dbutils.widgets.get('interest_name').lower()}", y="AGE", ax=axs[0])
    sns.regplot(data=socdemo_interest_df, x=f"ad_interest_affinity_{dbutils.widgets.get('interest_name').lower()}", y="GENDER", ax=axs[1])
    sns.histplot(data=socdemo_interest_df, x=f"ad_interest_affinity_{dbutils.widgets.get('interest_name').lower()}", hue="GENDER", ax=axs[2])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Income interaction
# MAGIC **NOT reliable** because income estimation is based on about 60 interest affinities (circular definition problem).
# MAGIC 1. Scatterplot of chosen interest affinity versus *low* income (we look for a slope)
# MAGIC 2. Scatterplot of chosen interest affinity versus *high* income (we look for a slope)

# COMMAND ----------

@dp.transformation(get_fs_all,
                   dp.read_table("silver.income_interest_scores"),
                   dp.get_widget_value("interest_name"),
                   dp.get_widget_value("socdemo"))
def income_interest(fs_all, df_income, interest_widget, socdemo_widget):
    if not socdemo_widget:
        return
    interest_col = f"ad_interest_affinity_{interest_widget.lower()}"
    df_income = df_income.select("final_interest_score_low", "final_interest_score_high", "user_id").dropDuplicates()
    res = fs_all.select("user_id", interest_col).join(df_income, on="user_id")
    return res

if dbutils.widgets.get('socdemo') == "True":
    income_interest_df = income_interest.result.toPandas().sample(5000)
    fig, axs = plt.subplots(1, 2, figsize=(16,9))
    sns.regplot(data=income_interest_df, x=f"ad_interest_affinity_{dbutils.widgets.get('interest_name').lower()}", y="final_interest_score_low", ax=axs[0])
    sns.regplot(data=income_interest_df, x=f"ad_interest_affinity_{dbutils.widgets.get('interest_name').lower()}", y="final_interest_score_high", ax=axs[1])
