from pyspark.sql import types as T

# schema - saving loaded raw data to delta
def get_schema_cpex_piano_cleansed():
    schema = T.StructType([
            T.StructField("activeTime", T.IntegerType()),
            T.StructField("adspaces", T.ArrayType(T.StringType())),
            T.StructField("browser", T.StringType()),
            T.StructField("browserLanguage", T.StringType()),
            T.StructField("browserTimezone", T.ShortType()),
            T.StructField("browserVersion", T.StringType()),
            T.StructField("capabilities", T.ArrayType(T.StringType())),
            T.StructField("city", T.StringType()),
            T.StructField("colorDepth", T.ShortType()),
            T.StructField("company", T.StringType()),
            T.StructField("connectionSpeed", T.StringType()),
            T.StructField("connectionType", T.StringType()),
            T.StructField("country", T.StringType()),
            T.StructField("customParameters", T.ArrayType(T.StructType([T.StructField("group", T.StringType(), True), T.StructField("item", T.StringType(), True)]))),
            T.StructField("deviceType", T.StringType()),
            T.StructField("eventId", T.LongType()),
            T.StructField("exitLinkHost", T.StringType()),
            T.StructField("exitLinkQuery", T.StringType()),
            T.StructField("exitLinkUrl", T.StringType()),
            T.StructField("externalUserIds", T.ArrayType(T.StringType())),
            T.StructField("host", T.StringType()),
            T.StructField("intents", T.ArrayType(T.StringType())),
            T.StructField("isoRegion", T.StringType()),
            T.StructField("metrocode", T.StringType()),
            T.StructField("mobileBrand", T.StringType()),
            T.StructField("newUser", T.LongType()),
            T.StructField("os", T.StringType()),
            T.StructField("postalCode", T.StringType()),
            T.StructField("query", T.StringType()),
            T.StructField("referrerHost", T.StringType()),
            T.StructField("referrerHostClass", T.StringType()),
            T.StructField("referrerNewsAggregator", T.StringType()),
            T.StructField("referrerQuery", T.StringType()),
            T.StructField("referrerSearchEngine", T.StringType()),
            T.StructField("referrerSocialNetwork", T.StringType()),
            T.StructField("referrerUrl", T.StringType()),
            T.StructField("region", T.StringType()),
            T.StructField("resolution", T.StringType()),
            T.StructField("retargetingParameters", T.ArrayType(T.StringType())),
            T.StructField("scrollDepth", T.LongType()),
            T.StructField("sessionBounce", T.BooleanType()),
            T.StructField("sessionStart", T.BooleanType()),
            T.StructField("sessionStop", T.BooleanType()),
            T.StructField("site", T.StringType()),
            T.StructField("rp_pageurl", T.StringType()),
            T.StructField("userAgent", T.StringType()),
            T.StructField("userCorrelationId", T.StringType()),
            T.StructField("DEVICE", T.StringType()),
            T.StructField("userParameters", T.ArrayType(T.StringType())),
            T.StructField("SOURCE_FILE", T.StringType()),
            T.StructField("OWNER_NAME", T.StringType()),
            T.StructField("OWNER_ID", T.StringType()),
            T.StructField("FLAG_ADVERTISER", T.StringType()),
            T.StructField("FLAG_PUBLISHER", T.StringType()),
            T.StructField("EVENT_TIME", T.TimestampType()),
            T.StructField("day", T.DateType()),
            T.StructField("hour", T.IntegerType()),
            T.StructField("rp_pagetitle", T.StringType()),
            T.StructField("rp_pagekeywords", T.StringType()),
            T.StructField("rp_pagedescription", T.StringType()),
            T.StructField("rp_c_p_pageclass", T.StringType()),
            T.StructField("rp_c_p_publishedDateTime", T.StringType()),
            T.StructField("AGE", T.StringType()),
            T.StructField("GENDER", T.StringType()),
        ])

       # Define additional information
    info ={
        "primary_key": [ 'EVENT_TIME', 'DEVICE'],
        "partition_by": ['day'], # INSERT PARTITION KEY(s) HERE (OPTIONAL)
        "table_properties": {
            'delta.autoOptimize.optimizeWrite': 'true'
            }
    }
    return schema, info
