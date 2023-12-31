from pyspark.sql import types as T


# schema - saving loaded raw data to delta
def get_schema_cpex_piano_cleansed():
    schema = T.StructType(
        [
            T.StructField("activeTime", T.IntegerType(), True),
            T.StructField("adspaces", T.ArrayType(T.StringType(), True), True),
            T.StructField("browser", T.StringType(), True),
            T.StructField("browserLanguage", T.StringType(), True),
            T.StructField("browserTimezone", T.ShortType(), True),
            T.StructField("browserVersion", T.StringType(), True),
            T.StructField("capabilities", T.ArrayType(T.StringType(), True), True),
            T.StructField("city", T.StringType(), True),
            T.StructField("colorDepth", T.ShortType(), True),
            T.StructField("company", T.StringType(), True),
            T.StructField("connectionSpeed", T.StringType(), True),
            T.StructField("connectionType", T.StringType(), True),
            T.StructField("country", T.StringType(), True),
            T.StructField(
                "customParameters",
                T.ArrayType(
                    T.StructType(
                        [
                            T.StructField("group", T.StringType(), True),
                            T.StructField("item", T.StringType(), True),
                        ]
                    ),
                    True,
                ),
                True,
            ),
            T.StructField("deviceType", T.StringType(), True),
            T.StructField("eventId", T.LongType(), True),
            T.StructField("exitLinkHost", T.StringType(), True),
            T.StructField("exitLinkQuery", T.StringType(), True),
            T.StructField("exitLinkUrl", T.StringType(), True),
            T.StructField(
                "externalUserIds",
                T.ArrayType(
                    T.StructType(
                        [
                            T.StructField("id", T.StringType(), True),
                            T.StructField("type", T.StringType(), True),
                        ]
                    ),
                    False,
                ),
                False,
            ),
            T.StructField("host", T.StringType(), True),
            T.StructField("intents", T.ArrayType(T.StringType(), True), True),
            T.StructField("isoRegion", T.StringType(), True),
            T.StructField("metrocode", T.StringType(), True),
            T.StructField("mobileBrand", T.StringType(), True),
            T.StructField("newUser", T.LongType(), True),
            T.StructField("os", T.StringType(), True),
            T.StructField("postalCode", T.StringType(), True),
            T.StructField("query", T.StringType(), True),
            T.StructField("referrerHost", T.StringType(), True),
            T.StructField("referrerHostClass", T.StringType(), True),
            T.StructField("referrerNewsAggregator", T.StringType(), True),
            T.StructField("referrerQuery", T.StringType(), True),
            T.StructField("referrerSearchEngine", T.StringType(), True),
            T.StructField("referrerSocialNetwork", T.StringType(), True),
            T.StructField("referrerUrl", T.StringType(), True),
            T.StructField("region", T.StringType(), True),
            T.StructField("resolution", T.StringType()),
            T.StructField(
                "retargetingParameters", T.ArrayType(T.StringType(), True), True
            ),
            T.StructField("scrollDepth", T.LongType(), True),
            T.StructField("sessionBounce", T.BooleanType(), True),
            T.StructField("sessionStart", T.BooleanType(), True),
            T.StructField("sessionStop", T.BooleanType(), True),
            T.StructField("site", T.StringType(), True),
            T.StructField("rp_pageurl", T.StringType(), True),
            T.StructField("userAgent", T.StringType(), True),
            T.StructField("userCorrelationId", T.StringType(), True),
            T.StructField("DEVICE", T.StringType(), True),
            T.StructField(
                "userParameters",
                T.ArrayType(
                    T.StructType(
                        [
                            T.StructField("group", T.StringType(), True),
                            T.StructField("item", T.StringType(), True),
                        ]
                    ),
                    False,
                ),
                False,
            ),
            T.StructField("SOURCE_FILE", T.StringType(), True),
            T.StructField("OWNER_NAME", T.StringType(), True),
            T.StructField("OWNER_ID", T.StringType(), True),
            T.StructField("FLAG_ADVERTISER", T.StringType(), True),
            T.StructField("FLAG_PUBLISHER", T.StringType(), True),
            T.StructField("EVENT_TIME", T.TimestampType(), True),
            T.StructField("day", T.DateType(), True),
            T.StructField("hour", T.IntegerType(), True),
            T.StructField("rp_pagetitle", T.StringType(), True),
            T.StructField("rp_pagekeywords", T.StringType(), True),
            T.StructField("rp_pagedescription", T.StringType(), True),
            T.StructField("rp_c_p_pageclass", T.StringType(), True),
            T.StructField("rp_c_p_publishedDateTime", T.StringType(), True),
            T.StructField("AGE", T.StringType(), True),
            T.StructField("GENDER", T.StringType(), True),
        ]
    )

    # Define additional information
    info = {
        "primary_key": ["EVENT_TIME", "DEVICE"],
        "partition_by": ["day"],  # INSERT PARTITION KEY(s) HERE (OPTIONAL)
        "table_properties": {"delta.autoOptimize.optimizeWrite": "true"},
    }
    return schema, info
