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
HD_APP = "HONEYDEW_APP"  # Name of installed Native Application
HD_WORKSPACE = "tasty_bytes"  # Honeydew model
HD_BRANCH = "prod"  # Honeydew branch
HD_DOMAIN = "llm"  # Honeydew domain

#
# Behavior & UI  settings
HISTORY_ENABLED = True  # Display chat history
SHOW_SQL_QUERY = 1  # Display SQL tab
SHOW_EXPLAIN_QUERY = 1  # Display Explain tab

# Set to '1' to see debug messages
_DEBUG = 0


# Results limit
RESULTS_LIMIT = 10000


class TYPES:
    # pylint: disable=too-few-public-methods
    QUERY_RESULT = "data"
    INIT = "init"
    SQL = "sql"
    MARKDOWN = "markdown"


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


@supress_failures
def make_chart(df: pd.DataFrame) -> bool:
    # pylint: disable=too-many-branches,too-many-statements

    def create_grouped_bar_chart(
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
            return create_grouped_bar_chart(df_to_show, str_columns, numeric_columns)

    # multiple numeric
    elif len(numeric_columns) > 1:
        return create_grouped_bar_chart(
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


def ask(
    question: str,
) -> typing.Dict[str, typing.Any]:
    try:
        sql = f"""select {HD_APP}.API.ASK_QUESTION(
                    workspace => '{HD_WORKSPACE}',
                    branch => '{HD_BRANCH}',
                    question => '{question}',
                    domain => '{HD_DOMAIN}',
                    default_results_limit => {RESULTS_LIMIT}) as response"""

        data = session.sql(sql)

        return typing.cast(
            typing.Dict[str, typing.Any],
            json.loads(data.collect()[0].as_dict()["RESPONSE"]),
        )

    except Exception as exp:  # pylint: disable=broad-except
        st.error(str(exp))

    return {}


def append_content(
    parent: typing.Any,
    content: typing.Any,
    arr: typing.Optional[typing.List[typing.Any]],
) -> typing.Any:

    def get_csv(df: pd.DataFrame) -> bytes:
        return typing.cast(bytes, df.to_csv().encode("utf-8"))

    if arr is not None:
        arr.append(content)

    if content["type"] == TYPES.INIT:
        if str(content["role"]) == "assistant":
            avatar = HONEYDEW_ICON_URL
        elif str(content["role"]) == "user":
            avatar = "🧑‍💻"
        else:
            avatar = None
        parent = parent.chat_message(content["role"], avatar=avatar)

    if content["text"] is not None:
        parent.markdown(content["text"])

    if str(content["type"]) == TYPES.QUERY_RESULT:

        df = content["data"]
        json_query = content["json_query"]
        sql = content["hd_sql"]

        tab_names = ["Chart", "Data"]
        if SHOW_EXPLAIN_QUERY:
            tab_names.append("Explain")

        if SHOW_SQL_QUERY:
            tab_names.append("SQL")

        chart_tab, data_tab, explain_tab, hd_sql_tab = parent.tabs(tab_names)

        with chart_tab:
            if not make_chart(df):
                st.dataframe(df)

        with data_tab:
            st.dataframe(df)
            csv = get_csv(df)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                key=f"download_btn_{str(uuid.uuid4())}",
                file_name="data.csv",
                mime="text/csv",
            )

        if SHOW_EXPLAIN_QUERY:
            with explain_tab:
                st.code(json.dumps(json_query, indent=4))

        if SHOW_SQL_QUERY:
            with hd_sql_tab:
                st.markdown(f"```sql\n{sql}\n```")

    return parent


def run_query_flow(user_question: str) -> None:

    append_content(
        parent=st,
        arr=st.session_state.content,
        content={
            "type": TYPES.INIT,
            "role": "user",
            "text": user_question,
        },
    )

    parent = append_content(
        parent=st,
        arr=st.session_state.content,
        content={
            "type": TYPES.INIT,
            "role": "assistant",
            "text": None,
        },
    )

    container = parent.container()
    stat_parent = parent.empty()
    stat = stat_parent.status(label="Running..", state="running")

    # executing Honeydew
    hdresponse = ask(user_question)

    # showing unexepcted error
    if hdresponse["error"] is not None:
        append_content(
            parent=container,
            arr=st.session_state.content,
            content={
                "type": TYPES.MARKDOWN,
                "role": "assistant",
                "text": str(hdresponse["error"]),
            },
        )
        return

    # showing the LLM response
    if hdresponse["llm_response"] is not None:
        append_content(
            parent=container,
            arr=st.session_state.content,
            content={
                "type": TYPES.MARKDOWN,
                "role": "assistant",
                "text": str(hdresponse["llm_response"]),
            },
        )

    # executing query
    if hdresponse["sql"] is not None:
        stat.update(label="Runninq Query..", state="running")
        df = session.sql(str(hdresponse["sql"])).to_pandas()

        append_content(
            parent=container,
            content={
                "type": TYPES.QUERY_RESULT,
                "text": "",
                "data": df,
                "json_query": dict(hdresponse["perspective"]),
                "hd_sql": str(hdresponse["sql"]),
                "role": "assistant",
            },
            arr=st.session_state.content,
        )

    stat.update(label="Completed", state="complete")


#
# Main flow
if "content" not in st.session_state:
    st.session_state.content = []

st.title("Honeydew Analyst")
st.markdown(f"Semantic Model: `{HD_WORKSPACE}` on branch `{HD_BRANCH}`")

parent_st = st

if HISTORY_ENABLED:
    for content_item in st.session_state.content:
        if content_item["type"] == TYPES.INIT:
            parent_st = append_content(parent=st, content=content_item, arr=None)

        else:
            append_content(parent=parent_st, content=content_item, arr=None)

if user_question_input := st.chat_input("Ask me.."):
    run_query_flow(user_question_input)
