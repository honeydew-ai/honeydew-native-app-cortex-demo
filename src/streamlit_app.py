####################################################################################
# MIT License
#
# Copyright (c) Honeydew Data Inc.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
####################################################################################

import json
import os
import re
import typing
import uuid
from typing import Any, Callable, Dict, Optional

import altair as alt
import dotenv as env
import pandas as pd
import snowflake.connector
import streamlit as st

# Constants


#
# Behavior & UI  settings
HISTORY_ENABLED = True  # Display chat history
SHOW_SQL_QUERY = 1  # Display SQL tab
SHOW_EXPLAIN_QUERY = 1  # Display Explain tab

HD_WORKSPACE = "tasty_bytes"  # Honeydew model
HD_APP = "HONEYDEW_APP"  # Name of installed Native Application
HD_BRANCH = "prod"  # Honeydew branch
HD_DOMAIN = "llm"  # Honeydew domain

# Set to '1' to see debug messages
_DEBUG = 1

HONEYDEW_ICON_URL = "https://honeydew.ai/wp-content/uploads/2022/12/Logo_Icon@2x.svg"


# Load environment variables from .env file
env.load_dotenv()


def supress_failures(
    func: Callable[..., Any],
) -> Callable[..., Any]:
    def func_without_errors(*args: Any, **kw: Any) -> Any:
        try:
            return func(*args, **kw)
        except Exception as exp:
            if _DEBUG:
                raise exp

            return None

    return func_without_errors


#
# HD API


def init() -> None:

    if (
        "connection_initialized" in st.session_state
        and st.session_state.connection_initialized is True
    ):
        return

    try:
        from snowflake.snowpark.context import get_active_session

        st.session_state.SESSION = get_active_session()

    except Exception:
        st.session_state.CONECTION = snowflake.connector.connect(
            user=os.getenv("SF_USER"),
            password=os.getenv("SF_PASSWORD"),
            account=os.getenv("SF_ACCOUNT"),
            host=os.getenv("SF_HOST"),
            port=443,
            warehouse=os.getenv("SF_WAREHOUSE"),
            role=os.getenv("SF_ROLE"),
        )
        st.session_state.CONECTION.cursor().execute(
            f"USE WAREHOUSE {os.getenv('SF_WAREHOUSE')}",
        )

    st.session_state.connection_initialized = True


def encode(str_val: str) -> str:
    str_val = re.sub(r"'+", "''", str_val, flags=re.S)
    return str_val


def execute_sql(sql: str) -> typing.Any:
    if (sql is None) or (sql == ""):
        raise Exception("SQL is empty")

    r = None
    if "SESSION" in st.session_state:
        r = st.session_state.SESSION.sql(sql).collect()

    elif "CONECTION" in st.session_state:
        r = st.session_state.CONECTION.cursor().execute(sql).fetchall()
    else:
        raise Exception("No session / connection")

    return r


def execute_sql_table(sql: str) -> typing.Any:
    if (sql is None) or (sql == ""):
        raise Exception("SQL is empty")

    r = None
    if "SESSION" in st.session_state:
        r = st.session_state.SESSION.sql(sql).to_pandas()

    elif "CONECTION" in st.session_state:
        r = st.session_state.CONECTION.cursor().execute(sql).fetch_pandas_all()
    else:
        raise Exception("No session / connection")

    return r


def execute_llm(
    messages: typing.List[str],
    temperature: float = 0.2,
    max_tokens: float = 8096,
) -> typing.Any:

    def extract_message(t: str) -> str:
        markers = ["JSON Query:", "```json", "```"]
        first_marker_pos = len(t)
        for marker in markers:
            pos = t.find(marker)
            if pos != -1 and pos < first_marker_pos:
                first_marker_pos = pos

        return t[:first_marker_pos].rstrip()

    def extract_json_query(t: str) -> typing.Any:
        for m in re.findall(r"```json(.*?)```", t, flags=re.DOTALL):
            return json.loads(m)

    query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE('{os.getenv("CORTEX_LLM")}',
        [
            {', '.join(messages)}
        ],
        {{'temperature': {temperature}, 'max_tokens': {max_tokens}}}) as RESPONSE"""

    cortex_response = json.loads(execute_sql(query)[0][0])["choices"][0]["messages"]

    return extract_message(cortex_response), extract_json_query(cortex_response)


def execute_hd_ask_question(
    question: str,
    h: typing.List[str],
) -> Any:

    sql = f"""select {HD_APP}.API.ASK_QUESTION(
                workspace => '{HD_WORKSPACE}',
                branch => '{HD_BRANCH}',
                question => '{question}',
                history => { "[" + ",".join(h) + "]"},
                domain => '{HD_DOMAIN}') as response"""

    r = json.loads(execute_sql(sql)[0][0])
    return r


init()


#
# Chat History

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"
TYPE_QUERY_RESULT = "assistant_query_result"
TYPE_ASSISTANT_INIT = "assistant_init"
TYPE_USER_MESSAGE = "user"
TYPE_ASSISTANT_MESSAGE = "assistant_message"
TYPE_LLM_RESPONSE = "assistant_llm_response"
TYPE_ERROR = "assistant_error"


class History:
    def __init__(self, name: Optional[str] = None, system_prompt: Optional[str] = None):
        self.messages: typing.List[Any] = []
        self.ui_model: typing.List[Dict[str, Any]] = []
        self.name = name
        if system_prompt is not None and system_prompt != "":
            self.push_system_message(encode(system_prompt))

    def push_system_message(self, m: str) -> None:
        if m is not None and m != "":
            self.messages.append(
                f"""{{'role':'{ROLE_SYSTEM}', 'content': '{m}'}}""",
            )

    def to_llm_message(self, m: Dict[str, Any]) -> str:
        return f"""{{'role':'{m["role"]}', 'content': '{encode(m['text'])}'}}"""

    def push_assistant_init(self) -> Dict[str, Any]:
        m = {
            "type": TYPE_ASSISTANT_INIT,
            "role": ROLE_ASSISTANT,
            "text": "Got it, please wait a sec.",
        }

        self.ui_model.append(m)
        return m

    def push_assistant_error(
        self,
        error: str,
    ) -> Dict[str, Any]:

        m = {
            "type": TYPE_ERROR,
            "role": ROLE_ASSISTANT,
            "text": f"\n{error}\n",
        }

        # self.messages.append(self.to_llm_message(m))
        self.ui_model.append(m)

        return m

    def push_assistant_llm_response(
        self,
        message: str,
        json_query: Optional[typing.Any] = None,
        append_json_to_ui: bool = True,
    ) -> typing.Dict[str, Any]:
        m = {
            "type": TYPE_LLM_RESPONSE,
            "role": ROLE_ASSISTANT,
            "json_query": json_query,
            "text": f"""\n{message}\n```json\n{json.dumps(json_query)}\n```""",
        }

        self.messages.append(self.to_llm_message(m))

        if not append_json_to_ui:
            m["text"] = f"\n{message}\n"

        self.ui_model.append(m)

        return m

    def push_assistant_query_result(self, df: Any) -> Dict[str, Any]:
        m = {
            "type": TYPE_QUERY_RESULT,
            "role": ROLE_ASSISTANT,
            "data": df,
            "text": "",
        }

        self.ui_model.append(m)

        return m

    def push_user_message(self, message: str) -> typing.Dict[str, Any]:
        m = {
            "type": TYPE_USER_MESSAGE,
            "role": ROLE_USER,
            "text": message,
        }

        self.messages.append(self.to_llm_message(m))
        self.ui_model.append(m)

        return m


# # Application

if "history" not in st.session_state:
    st.session_state.history = History()

history: History = st.session_state.history


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


def render_chart(df: pd.DataFrame) -> bool:

    if len(df) == 0:
        return False

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
        metric_column = numeric_columns[0]  # First numeric column

        if len(df_to_show) > 50:
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


def render_dataframe(df: pd.DataFrame) -> None:
    if len(df) == 0:
        st.markdown("No data available to display. Please refine your question.")
    st.dataframe(df)
    if len(df) > 0:
        csv = typing.cast(bytes, df.to_csv().encode("utf-8"))
        st.download_button(
            label="Download data as CSV",
            data=csv,
            key=f"download_btn_{str(uuid.uuid4())}",
            file_name="data.csv",
            mime="text/csv",
        )


@supress_failures
def render_message(
    message: typing.Dict[str, typing.Any],
    parent: typing.Any,
) -> Optional[Any]:

    def find_item(json_data: typing.Any, search_str: str) -> Optional[typing.Any]:
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

    if message["type"] == TYPE_ASSISTANT_INIT:
        parent = parent.chat_message(ROLE_ASSISTANT, avatar=HONEYDEW_ICON_URL)

    if message["type"] == TYPE_USER_MESSAGE:
        parent = parent.chat_message(ROLE_USER, avatar="🧑‍💻")

    if message["text"] is not None:
        parent.markdown(message["text"])

    if message["type"] == TYPE_QUERY_RESULT:
        df = message["data"]
        json_query = message["json_query"]
        sql_query = message["sql_query"]

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
            render_dataframe(df)

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


@supress_failures
def process_user_question(user_question: str) -> None:

    history_clone = history.messages[:]
    stat = None
    try:
        render_message(history.push_user_message(user_question), st)
        parent = render_message(history.push_assistant_init(), st)

        container = parent.container()
        stat_parent = parent.empty()
        stat = stat_parent.status(label="Running..", state="running")

        # executing Honeydew
        hdresponse = execute_hd_ask_question(
            user_question,
            h=history_clone,
        )
        if "error" in hdresponse and hdresponse["error"] is not None:
            render_message(history.push_assistant_error(hdresponse["error"]), container)
            return

        render_message(
            history.push_assistant_llm_response(
                message=hdresponse["llm_response"],
                json_query=(
                    json.loads(hdresponse["llm_response_json"])
                    if "llm_response_json" in hdresponse
                    else None
                ),
                append_json_to_ui=False,
            ),
            container,
        )

        if "sql" in hdresponse and hdresponse["sql"] is not None:
            df = execute_sql_table(hdresponse["sql"])
            df = df.round(2)
            m = history.push_assistant_query_result(df)
            m["json_query"] = hdresponse["perspective"]
            m["sql_query"] = hdresponse["sql"]
            render_message(m, container)

            parent.divider()

    finally:
        if stat is not None:
            stat.update(label="Completed", state="complete")
            stat_parent.empty()


#
# Main flow
st.title("Honeydew Analyst")
st.markdown(
    f"Semantic Model: `{HD_WORKSPACE}` on branch `{HD_BRANCH}`",
)

parent_st = st

for history_item in history.ui_model:
    if (
        history_item["type"] == TYPE_ASSISTANT_INIT
        or history_item["type"] == TYPE_USER_MESSAGE
    ):
        parent_st = render_message(history_item, st)
    else:
        render_message(history_item, parent_st)

if user_question_input := st.chat_input(placeholder="Ask me.."):
    process_user_question(user_question_input)
