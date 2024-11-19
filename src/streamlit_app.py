import json
import typing
import uuid

import altair as alt
import pandas as pd
import streamlit as st
from snowflake.snowpark.context import get_active_session

# Constants

#
# Honeydew related settings
HD_APP = "SEMANTIC_LAYER_ENTERPRISE_EDITION"  # Name of installed Native Application
HD_WORKSPACE = "tpch_demo"  # Honeydew model
HD_BRANCH = "prod"  # Honeydew branch
HD_DOMAIN = "llm"  # Honeydew domain

#
# Behavior & UI  settings
HISTORY_ENABLED = True  # Display chat history
SHOW_SQL_QUERY = 1  # Display SQL tab
SHOW_EXPLAIN_QUERY = 1  # Display Explain tab

# Set to '1' to see debug messages
_DEBUG = 1

# Cortex LLM
CORTEX_LLM = "llama3.1-405b"
# CORTEX_LLM = "mistral-large2"

# Results limit
RESULTS_LIMIT = 10000


class HistoryItemTypes:  # pylint: disable=too-few-public-methods
    USER = "user"
    QUERY_RESULT = "data"
    ASSISTANT_INIT = "assistant_init"
    MARKDOWN = "markdown"


class Roles:  # pylint: disable=too-few-public-methods
    USER = "user"
    ASSISTANT = "assistant"


HONEYDEW_ICON_URL = "https://honeydew.ai/wp-content/uploads/2022/12/Logo_Icon@2x.svg"

# Application

session = get_active_session()


def supress_failures(
    func: typing.Callable[..., typing.Any],
) -> typing.Callable[..., typing.Any]:
    def func_without_errors(*args: typing.Any, **kw: typing.Any) -> typing.Any:
        try:
            return func(*args, **kw)
        except Exception as exp:  # pylint: disable=broad-exception-caught
            if _DEBUG:
                st.code(f"Failed {func.__name__}:\n{exp}")
            return None

    return func_without_errors


def render_chart(df: pd.DataFrame) -> bool:
    # pylint: disable=too-many-branches,too-many-statements

    def render_grouped_bar_chart(
        df: pd.DataFrame,
        str_columns: typing.List[str],
        numeric_columns: typing.List[str],
    ) -> bool:
        if len(str_columns) > 2:
            return False

        params = {}
        if len(str_columns) == 2:

            params["x"] = alt.X(f"{str_columns[0]}:N", title=None)
            params["color"] = (
                "Series:N" if len(numeric_columns) > 1 else f"{str_columns[1]}:N"
            )

            if len(numeric_columns) > 1:
                params["column"] = f"{str_columns[1]}:N"

        else:
            params["x"] = alt.X(f"{str_columns[0]}:N", title=None)
            params["color"] = (
                "Series:N" if len(numeric_columns) > 1 else f"{str_columns[0]}:N"
            )

        if len(numeric_columns) == 1:
            params["y"] = alt.Y("Value:Q", title=numeric_columns[0])

        else:
            params["y"] = alt.Y("Value:Q")
            params["xOffset"] = alt.XOffset("Series:N")

        # Melt the DataFrame to convert wide format into long format for grouped bars
        df = df.melt(
            id_vars=str_columns,
            value_vars=numeric_columns,
            var_name="Series",
            value_name="Value",
        )

        # Create the Altair grouped bar chart using the index as the x-axis
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(**params)
            .resolve_scale(y="shared")  # Ensure that y-axes across columns are shared
        )

        st.altair_chart(chart, use_container_width=True)

        return True

    def get_possible_date_columns(df: pd.DataFrame) -> typing.List[str]:
        date_columns = []
        for col in df.select_dtypes(include="object").columns:
            try:
                # Try converting the column to datetime
                df[col] = pd.to_datetime(
                    df[col],
                    errors="raise",
                    infer_datetime_format=True,
                )
                # If successful, record the column as a date column
                date_columns.append(col)
            except (ValueError, TypeError):
                # If conversion fails, the column is not a date
                pass
        return date_columns

    # Bug in streamlit with dots in column names - update column names
    updated_columns = {col: col.replace(".", "_") for col in df.columns}
    df = df.rename(columns=updated_columns)
    numeric_columns = list(df.select_dtypes(include=["number"]).columns)

    if len(numeric_columns) == 0:
        st.markdown("Chart currently not available for non-numeric data")
        return False

    # Bug in snowflake connector - does not set date types correctly in pandas, so manually detecting
    date_columns = get_possible_date_columns(df)
    str_columns = df.select_dtypes(include=["object"]).columns
    str_columns = [col for col in str_columns if col not in date_columns]
    df_to_show = df[numeric_columns + str_columns]

    for col in date_columns[:1]:
        df_to_show[col] = pd.to_datetime(df[col], errors="coerce")

    # date
    if len(date_columns) > 0:
        df_to_show = df_to_show.rename(columns={date_columns[0]: "index"}).set_index(
            "index",
        )

        if len(str_columns) == 0:
            st.line_chart(df_to_show)
        elif len(str_columns) == 1 and len(numeric_columns) == 1:
            st.line_chart(df_to_show, y=numeric_columns[0], color=str_columns[0])

    # text
    elif len(str_columns) >= 1:

        if len(df_to_show) > 50:
            metric_column = numeric_columns[0]  # First numeric column
            st.markdown(f"**Showing top 50 values sorted by {metric_column}**")
            df_to_show = df_to_show.sort_values(by=metric_column, ascending=False).head(
                50,
            )

        if len(str_columns) == 1:
            chart = (
                alt.Chart(df_to_show)
                .mark_bar()
                .encode(
                    y=alt.X(
                        str_columns[0],
                        sort=alt.EncodingSortField(
                            field=metric_column,
                            order="descending",
                        ),
                    ),
                    x=alt.Y(numeric_columns[0]),
                    color=alt.Color(str_columns[0], legend=None),  # Hide legend
                )
            )

            st.altair_chart(chart, use_container_width=True)

        else:
            return render_grouped_bar_chart(df_to_show, str_columns, numeric_columns)

    # multiple numeric
    elif len(numeric_columns) > 1:
        return render_grouped_bar_chart(
            df_to_show,
            [numeric_columns[0]],
            numeric_columns[1:],
        )

    # single numeric
    elif len(numeric_columns) == 1:
        if len(df_to_show) == 1:
            value = f"{df[numeric_columns[0]].iloc[0]:,}"
            st.metric(label=numeric_columns[0], value=value)
        else:
            st.bar_chart(df_to_show)

    else:
        st.markdown("Chart currently not available")
        return False

    return True


@supress_failures
def render_content(
    parent: typing.Any,
    content: typing.Any,
) -> typing.Any:

    def find_item(json_data: typing.Any, search_str: str) -> typing.Any:
        # Split the search string into entity and name components
        entity, name = search_str.split(".")

        # Search in both attributes and metrics within components
        for section in ["attributes", "metrics"]:
            for item in json_data.get("components", {}).get(section, []):
                if item.get("entity") == entity and item.get("name") == name:
                    return item
        return None

    def get_item_caption(json_data: typing.Any, p: str, name: str) -> typing.Any:
        parts = [" * "]

        item = find_item(json_data, name)

        if item is not None and item["ui_url"] is not None and item["ui_url"] != "":
            parts.append(f" [{name}]({item['ui_url']})")
        elif p == "filters":
            parts.append(f"`{name}`")
        else:
            parts.append(name)

        if (
            item is not None
            and item["description"] is not None
            and item["description"] != ""
        ):
            parts.append(f"\n\n    {item['description']}")

        return "".join(parts)

    def get_panel_decription(q: typing.Dict[str, typing.Any], p: str, n: str) -> str:
        if p not in q or q[p] is None or len(q[p]) == 0:
            return ""

        r = f"\n\n**{n}:**\n\n"
        if isinstance(q[p], str):
            r += q[p]

        else:
            r += "\n\n".join(get_item_caption(q, p, val) for val in q[p])

        return r

    if str(content["type"]) == HistoryItemTypes.USER:
        parent = parent.chat_message(content["role"], avatar="🧑‍💻")

    if str(content["type"]) == HistoryItemTypes.ASSISTANT_INIT:
        parent = parent.chat_message(content["role"], avatar=HONEYDEW_ICON_URL)

    if content["text"] is not None:
        parent.markdown(content["text"])

    if str(content["type"]) == HistoryItemTypes.QUERY_RESULT:
        df = content["data"]
        df = df.round(2)
        json_query = content["json_query"]
        sql_query = content["sql_query"]

        tab_names = ["Chart", "Data"]
        if SHOW_EXPLAIN_QUERY:
            tab_names.append("Explain")

        if SHOW_SQL_QUERY:
            tab_names.append("SQL")

        chart_tab, data_tab, explain_tab, hd_sql_tab = parent.tabs(tab_names)

        with chart_tab:
            if not render_chart(df):
                st.dataframe(df)

        with data_tab:
            st.dataframe(df)
            csv = typing.cast(bytes, df.to_csv().encode("utf-8"))
            st.download_button(
                label="Download data as CSV",
                data=csv,
                key=f"download_btn_{str(uuid.uuid4())}",
                file_name="data.csv",
                mime="text/csv",
            )

        if SHOW_EXPLAIN_QUERY:
            with explain_tab:
                st.markdown(
                    "".join(
                        [
                            get_panel_decription(
                                json_query,
                                "attributes",
                                "Attributes",
                            ),
                            get_panel_decription(json_query, "metrics", "Metrics"),
                            get_panel_decription(json_query, "filters", "Filters"),
                            #   get_panel_decription(json_query, "transform_sql", "Order"),
                        ],
                    ),
                )

        if SHOW_SQL_QUERY:
            with hd_sql_tab:
                st.markdown(f"```sql{sql_query}```")

    return parent


def ask(
    question: str,
) -> typing.Dict[str, typing.Any]:
    try:
        sql = f"""select {HD_APP}.API.ASK_QUESTION(
                    workspace => '{HD_WORKSPACE}',
                    branch => '{HD_BRANCH}',
                    question => '{question}',
                    domain => '{HD_DOMAIN}',
                    cortex_llm_name => '{CORTEX_LLM}',
                    default_results_limit => {RESULTS_LIMIT}) as response"""

        data = session.sql(sql)

        return typing.cast(
            typing.Dict[str, typing.Any],
            json.loads(data.collect()[0].as_dict()["RESPONSE"]),
        )

    except Exception as exp:  # pylint: disable=broad-except
        st.error(f"Failed:\n{exp}")

    return {}


@supress_failures
def run_query_flow(user_question: str) -> None:

    def render_history_item(  # pylint: disable=too-many-arguments
        *,
        parent: typing.Any,
        history_item_type: str,
        role: str,
        text: typing.Optional[str] = None,
        data: typing.Optional[typing.Any] = None,
        json_query: typing.Optional[typing.Any] = None,
        sql_query: typing.Optional[str] = None,
    ) -> typing.Any:

        item = {
            "type": history_item_type,
            "role": role,
            "text": text,
        }

        if data is not None:
            item["data"] = data

        if json_query is not None:
            item["json_query"] = json_query

        if sql_query is not None:
            item["sql_query"] = sql_query

        st.session_state.history.append(item)
        return render_content(parent, item)

    render_history_item(
        parent=st,
        history_item_type=HistoryItemTypes.USER,
        role=Roles.USER,
        text=user_question,
    )

    parent = render_history_item(
        parent=st,
        history_item_type=HistoryItemTypes.ASSISTANT_INIT,
        role=Roles.ASSISTANT,
    )
    container = parent.container()
    stat_parent = parent.empty()
    stat = stat_parent.status(label="Running..", state="running")

    try:
        # executing Honeydew
        hdresponse = ask(user_question)

        # showing unexepcted error
        if "error" in hdresponse and hdresponse["error"] is not None:
            render_history_item(
                parent=container,
                history_item_type=HistoryItemTypes.MARKDOWN,
                role=Roles.ASSISTANT,
                text=str(hdresponse["error"]),
            )
            return

        # showing the LLM response
        if "llm_response" in hdresponse:
            render_history_item(
                parent=container,
                history_item_type=HistoryItemTypes.MARKDOWN,
                role=Roles.ASSISTANT,
                text=str(hdresponse["llm_response"]),
            )

        # executing query
        if "sql" in hdresponse and hdresponse["sql"] is not None:
            stat.update(label="Runninq Query..", state="running")
            df = session.sql(str(hdresponse["sql"])).to_pandas()
            render_history_item(
                parent=container,
                history_item_type=HistoryItemTypes.QUERY_RESULT,
                role=Roles.ASSISTANT,
                text="",
                data=df,
                json_query=hdresponse["perspective"],
                sql_query=str(hdresponse["sql"]),
            )

    finally:
        stat.update(label="Completed", state="complete")


#
# Main flow
if "history" not in st.session_state:
    st.session_state.history = []

st.title("Honeydew Analyst")
st.markdown(f"Semantic Model: `{HD_WORKSPACE}` on branch `{HD_BRANCH}`")

parent_st = st

if HISTORY_ENABLED:
    for history_item in st.session_state.history:
        if (
            history_item["type"] == HistoryItemTypes.ASSISTANT_INIT
            or history_item["type"] == HistoryItemTypes.USER
        ):
            parent_st = render_content(parent=st, content=history_item)

        else:
            render_content(parent=parent_st, content=history_item)

if user_question_input := st.chat_input("Ask me.."):
    run_query_flow(user_question_input)
