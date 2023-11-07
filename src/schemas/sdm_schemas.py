from pyspark.sql import types as T


def get_schema_sdm_preprocessed():
    schema = T.StructType(
        [
            T.StructField("USER_ID", T.StringType(), True),
            T.StructField("SESSION_ID", T.LongType(), False),
            T.StructField("BROWSER_ID", T.LongType(), False),
            T.StructField("BROWSER_LANGUAGE", T.StringType(), True),
            T.StructField("VIEW_PORT_SIZE", T.StringType(), True),
            T.StructField("SCREEN_SIZE", T.StringType(), True),
            T.StructField("SESSION_NUMBER", T.StringType(), True),
            T.StructField("SESSION_START_TIME", T.TimestampType(), True),
            T.StructField("SESSION_END_TIME", T.TimestampType(), True),
            T.StructField("IS_BOUNCED", T.StringType(), True),
            T.StructField("NUMBER_OF_EVENTS", T.LongType(), False),
            T.StructField("NUMBER_OF_PAGE_VIEWS", T.LongType(), False),
            T.StructField("UNIQUE_PAGE_VIEWS", T.LongType(), False),
            T.StructField("TOTAL_TIME_IN_SESSION", T.DoubleType(), True),
            T.StructField("REFERRAL_PATH_FULL", T.StringType(), True),
            T.StructField("REFERRAL_PATH_NORMALIZED", T.StringType(), True),
            T.StructField("TRAFFIC_SOURCE_SOURCE", T.StringType(), True),
            T.StructField("TRAFFIC_SOURCE_MEDIUM", T.StringType(), True),
            T.StructField("TRAFFIC_SOURCE_KEYWORDS", T.StringType(), True),
            T.StructField("TRAFFIC_SOURCE_AD_CONTENT", T.StringType(), True),
            T.StructField("CHANNELING_GROUP", T.StringType(), True),
            T.StructField("LOCATION_ID", T.StringType(), True),
            T.StructField("IPv6", T.StringType(), True),
            T.StructField("IPv4", T.StringType(), True),
            T.StructField("DEVICE_ID", T.LongType(), False),
            T.StructField(
                "ZIPPED_DEVICE_DETAILS", T.ArrayType(T.StringType(), True), False
            ),
            T.StructField("OS_ID", T.LongType(), False),
            T.StructField("OS_LANGUAGE", T.StringType(), True),
            T.StructField("DATE", T.DateType(), True),
            T.StructField("TIMESTAMP", T.TimestampType(), True),
            T.StructField(
                "PAGES_TIMES",
                T.ArrayType(T.ArrayType(T.StringType(), True), False),
                False,
            ),
            T.StructField("DURATION", T.DoubleType(), True),
            T.StructField("PAGEVIEW_TYPE", T.StringType(), True),
        ]
    )

    # Define additional information
    info = {
        "primary_key": ["SESSION_ID"],
        "partition_by": ["DATE"],  # INSERT PARTITION KEY(s) HERE (OPTIONAL)
        "table_properties": {"delta.autoOptimize.optimizeWrite": "true"},
    }
    return schema, info


def get_schema_sdm_session():
    schema = T.StructType(
        [
            T.StructField("session_id", T.LongType(), True),
            T.StructField("session_start_datetime", T.TimestampType(), True),
            T.StructField("session_date", T.DateType(), True),
            T.StructField("session_end_datetime", T.TimestampType(), True),
            T.StructField("flag_active_session", T.BooleanType(), False),
            T.StructField("user_id", T.StringType(), True),
            T.StructField("browser_name", T.StringType(), True),
            T.StructField("device_category", T.StringType(), True),
            T.StructField("device_full_specification", T.StringType(), True),
            T.StructField("device_brand_name", T.StringType(), True),
            T.StructField("device_marketing_name", T.StringType(), True),
            T.StructField("device_model_name", T.StringType(), True),
            T.StructField("city", T.StringType(), True),
            T.StructField("continent", T.StringType(), True),
            T.StructField("subcontinent", T.StringType(), True),
            T.StructField("country", T.StringType(), True),
            T.StructField("region", T.StringType(), True),
            T.StructField("metro", T.StringType(), True),
            T.StructField("os_name", T.StringType(), True),
            T.StructField("os_version", T.StringType(), True),
            T.StructField("traffic_source_campaign", T.StringType(), True),
            T.StructField("traffic_source_medium", T.StringType(), True),
            T.StructField("traffic_source_source", T.StringType(), True),
            T.StructField("traffic_source_id", T.LongType(), True),
            T.StructField("os_id", T.LongType(), True),
            T.StructField("geo_id", T.LongType(), True),
            T.StructField("device_id", T.LongType(), True),
            T.StructField("browser_id", T.LongType(), True),
        ]
    )
    info = {
        "primary_key": ["session_id"],
        "partition_by": ["session_date"],  # INSERT PARTITION KEY(s) HERE (OPTIONAL)
        "table_properties": {"delta.autoOptimize.optimizeWrite": "true"},
    }
    return schema, info


def get_schema_sdm_device():
    schema = T.StructType(
        [
            T.StructField("DEVICE_ID", T.LongType()),
            T.StructField("DEVICE_CATEGORY", T.StringType()),
            T.StructField("DEVICE_FULL_SPECIFICATION", T.StringType()),
            T.StructField("DEVICE_BRAND_NAME", T.StringType()),
            T.StructField("DEVICE_MARKETING_NAME", T.StringType()),
        ]
    )

    info = {
        "primary_key": ["DEVICE_ID"],
        "table_properties": {"delta.autoOptimize.optimizeWrite": "true"},
    }
    return schema, info


def get_schema_sdm_browser():
    schema = T.StructType(
        [
            T.StructField("BROWSER_ID", T.LongType()),
            T.StructField("BROWSER_NAME", T.StringType()),
        ]
    )

    info = {
        "primary_key": ["BROWSER_ID"],
        "table_properties": {"delta.autoOptimize.optimizeWrite": "true"},
    }
    return schema, info


def get_schema_sdm_os():
    schema = T.StructType(
        [
            T.StructField("OS_ID", T.LongType()),
            T.StructField("OS_NAME", T.StringType()),
            T.StructField("OS_VERSION", T.StringType()),
        ]
    )

    info = {
        "primary_key": ["OS_ID"],
        "table_properties": {"delta.autoOptimize.optimizeWrite": "true"},
    }
    return schema, info


def get_schema_sdm_pageview():
    schema = T.StructType(
        [
            T.StructField("page_screen_view_id", T.StringType(), False),
            T.StructField("page_screen_title", T.StringType(), True),
            T.StructField("page_screen_view_timestamp", T.TimestampType(), True),
            T.StructField("page_screen_view_date", T.DateType(), True),
            T.StructField("URL_NORMALIZED", T.StringType(), True),
            T.StructField("full_url", T.StringType(), True),
            T.StructField("hostname", T.StringType(), True),
            T.StructField("page_path", T.StringType(), True),
            T.StructField("page_path_level_1", T.StringType(), True),
            T.StructField("page_path_level_2", T.StringType(), True),
            T.StructField("page_path_level_3", T.StringType(), True),
            T.StructField("page_path_level_4", T.StringType(), True),
            T.StructField("search_engine", T.StringType(), True),
            T.StructField("flag_advertiser", T.BooleanType(), True),
            T.StructField("flag_publisher", T.BooleanType(), True),
            T.StructField("session_id", T.LongType(), True),
            T.StructField("user_id", T.StringType(), True),
            T.StructField("owner_name", T.StringType(), True),
        ]
    )

    info = {
        "primary_key": ["page_screen_view_id"],
        "partition_by": [
            "page_screen_view_date"
        ],  # INSERT PARTITION KEY(s) HERE (OPTIONAL)
        "table_properties": {"delta.autoOptimize.optimizeWrite": "true"},
    }
    return schema, info


def get_schema_sdm_url():
    schema = T.StructType(
        [
            T.StructField("URL_TITLE", T.StringType(), True),
            T.StructField("URL_KEYWORDS", T.StringType(), True),
            T.StructField("URL_DESCRIPTIONS", T.StringType(), True),
            T.StructField("PAGE_TYPE", T.StringType(), True),
            T.StructField("PUBLISHING_DATE", T.StringType(), True),
            T.StructField("URL_DOMAIN_FULL", T.StringType(), True),
            T.StructField("URL_DOMAIN_1_LEVEL", T.StringType(), False),
            T.StructField("URL_DOMAIN_2_LEVEL", T.StringType(), False),
            T.StructField("URL_NORMALIZED", T.StringType(), True),
            T.StructField("URL_ADFORM_FORMAT", T.StringType(), True),
            T.StructField(
                "URL_TITLE_CLEANED", T.ArrayType(T.StringType(), True), False
            ),
            T.StructField(
                "URL_TITLE_STEMMED", T.ArrayType(T.StringType(), True), False
            ),
            T.StructField(
                "URL_KEYWORDS_CLEANED", T.ArrayType(T.StringType(), True), False
            ),
            T.StructField(
                "URL_KEYWORDS_STEMMED", T.ArrayType(T.StringType(), True), False
            ),
            T.StructField(
                "URL_DESCRIPTIONS_CLEANED", T.ArrayType(T.StringType(), True), False
            ),
            T.StructField(
                "URL_DESCRIPTIONS_STEMMED", T.ArrayType(T.StringType(), True), False
            ),
            T.StructField(
                "URL_NORMALIZED_KEYWORDS_CLEANED",
                T.ArrayType(T.StringType(), True),
                False,
            ),
            T.StructField(
                "URL_NORMALIZED_KEYWORDS_STEMMED",
                T.ArrayType(T.StringType(), True),
                False,
            ),
            T.StructField(
                "URL_TOKENS_ALL_CLEANED", T.ArrayType(T.StringType(), True), False
            ),
            T.StructField(
                "URL_TOKENS_ALL_STEMMED", T.ArrayType(T.StringType(), True), False
            ),
            T.StructField(
                "URL_TOKENS_ALL_CLEANED_UNIQUE",
                T.ArrayType(T.StringType(), True),
                False,
            ),
            T.StructField(
                "URL_TOKENS_ALL_STEMMED_UNIQUE",
                T.ArrayType(T.StringType(), True),
                False,
            ),
            T.StructField(
                "URL_TITLE_CLEANED_BIGRAMS", T.ArrayType(T.StringType(), True), True
            ),
            T.StructField(
                "URL_TITLE_STEMMED_BIGRAMS", T.ArrayType(T.StringType(), True), True
            ),
            T.StructField(
                "URL_KEYWORDS_CLEANED_BIGRAMS", T.ArrayType(T.StringType(), True), True
            ),
            T.StructField(
                "URL_KEYWORDS_STEMMED_BIGRAMS", T.ArrayType(T.StringType(), True), True
            ),
            T.StructField(
                "URL_DESCRIPTIONS_CLEANED_BIGRAMS",
                T.ArrayType(T.StringType(), True),
                True,
            ),
            T.StructField(
                "URL_DESCRIPTIONS_STEMMED_BIGRAMS",
                T.ArrayType(T.StringType(), True),
                True,
            ),
            T.StructField(
                "URL_NORMALIZED_KEYWORDS_CLEANED_BIGRAMS",
                T.ArrayType(T.StringType(), True),
                True,
            ),
            T.StructField(
                "URL_NORMALIZED_KEYWORDS_STEMMED_BIGRAMS",
                T.ArrayType(T.StringType(), True),
                True,
            ),
            T.StructField(
                "URL_TOKENS_ALL_CLEANED_BIGRAMS",
                T.ArrayType(T.StringType(), True),
                True,
            ),
            T.StructField(
                "URL_TOKENS_ALL_STEMMED_BIGRAMS",
                T.ArrayType(T.StringType(), True),
                True,
            ),
            T.StructField(
                "URL_TOKENS_ALL_CLEANED_UNIQUE_BIGRAMS",
                T.ArrayType(T.StringType(), True),
                True,
            ),
            T.StructField(
                "URL_TOKENS_ALL_STEMMED_UNIQUE_BIGRAMS",
                T.ArrayType(T.StringType(), True),
                True,
            ),
            T.StructField("TIME_ADDED", T.TimestampType(), True),
            T.StructField("DATE_ADDED", T.DateType(), True),
        ]
    )
    info = {
        "primary_key": ["URL_NORMALIZED"],
        "partition_by": ["DATE_ADDED"],  # INSERT PARTITION KEY(s) HERE (OPTIONAL)
        "table_properties": {"delta.autoOptimize.optimizeWrite": "true"},
    }
    return schema, info


def get_schema_sdm_tokenized_domains():
    schema = T.StructType(
        [
            T.StructField("USER_ID", T.StringType(), True),
            T.StructField("TOKENS", T.ArrayType(T.StringType()), True),
            T.StructField("URL_NORMALIZED", T.StringType(), True),
            T.StructField("DATE", T.DateType(), True),
            T.StructField("TOKEN", T.StringType(), True),
        ]
    )

    info = {
        "primary_key": [],
        "partition_by": ["DATE"],  # INSERT PARTITION KEY(s) HERE (OPTIONAL)
        "table_properties": {"delta.autoOptimize.optimizeWrite": "true"},
    }
    return schema, info
