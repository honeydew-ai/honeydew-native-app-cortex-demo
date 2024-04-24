# Streamlit + Honeydew Native App + Snowflake Cortex LLM

This demo shows a simple integration between Cortex and Honeydew in order to generate queries from English questions on semantics.

It uses the `tpch_demo` workspace based on [TPC-H Snowflake sample data](https://docs.snowflake.com/en/user-guide/sample-data-tpch), modeled in [Honeydew](https://honeydew.ai/).

## Example output

For the question "promotion revenue by year and region":

The LLM uses the [promo_revenue](https://github.com/honeydew-ai/tpch-demo/blob/main/tpch_demo/schema/lineitem/metrics/promo_revenue.yml) metric, as well as region name attribute in the [region table](https://github.com/honeydew-ai/tpch-demo/blob/main/tpch_demo/schema/region/datasets/region.yml).

![Output for "promotion revenue by year and region"](images/output.png)

## Usage

Put the code in `src/streamlit_app.py` in a Snowflake Streamlit application. 

Make sure it runs with a user that has access to Cortex ([required privileges](https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions#required-privileges)) and to the Native App ([installation guide steps 6-7](https://honeydew.ai/docs/integration/snowflake-native-app#installation))



