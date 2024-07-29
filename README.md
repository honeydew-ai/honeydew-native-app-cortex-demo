# Streamlit + Honeydew Native App + Snowflake Cortex LLM

This demo shows a simple integration between Cortex and Honeydew in order to generate queries from English questions on semantics.

It uses the `tpch_demo` workspace based on [TPC-H Snowflake sample data](https://docs.snowflake.com/en/user-guide/sample-data-tpch), modeled in [Honeydew](https://honeydew.ai/).

## Example output

For the question "promotion revenue by year and region":

The LLM uses the [promo_revenue](https://github.com/honeydew-ai/tpch-demo/blob/main/tpch_demo/schema/lineitem/metrics/promo_revenue.yml) metric, as well as region name attribute in the [region table](https://github.com/honeydew-ai/tpch-demo/blob/main/tpch_demo/schema/region/datasets/region.yml).

![Output for "promotion revenue by year and region"](images/output.png)

## Best Practices

1. Use a [domain](https://honeydew.ai/docs/domains) (like `llm`) to control what the LLM sees.
2. Attach a date spine with descriptions (the default one in Honeydew is good enough). Avoid putting any other date fields in the domain - resolving how a metric is connecting to the date spine is a semantic layer task, not an LLM task. 
3. Avoid similar-sounding attribute / metric names that can be confusing. Rename if possible, add descriptions if not.
4. Add sample values in descriptions of fields that you expect the LLM to filter on. For example, if you have a field city, add a description such as "Sample values: Seattle, Boston, Tel Aviv" so it will know the city format. 
5. Start with a small domain and expand gradually adding user questions and corresponding metrics.

## Usage

Put the code in `src/streamlit_app.py` in a Snowflake Streamlit application. 

Make sure it runs with a user that has access to Cortex ([required privileges](https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions#required-privileges)) and to the Native App ([installation guide steps 6-7](https://honeydew.ai/docs/integration/snowflake-native-app#installation))



