# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""The embedding atlas widget for notebooks"""

import pathlib
from typing import Any, Unpack

import duckdb

from .options import EmbeddingAtlasOptions, make_embedding_atlas_props
from .utils import arrow_to_bytes

try:
    import anywidget
    import traitlets
except ImportError:
    print(
        "⚠️ The widget depends on anywidget. Please run `pip install anywidget`, then try again."
    )
    raise


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _compute_trajectories_from_spec(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    x_col: str | None,
    y_col: str | None,
    spec: dict,
) -> list[dict]:
    """Aggregate the data table into a list of trajectory dicts for the embedding view.

    Each resulting dict has: ``points`` (list of {x, y}), ``id`` (group value),
    and optionally ``color``, ``width``, ``opacity``.
    """
    group_by = spec.get("group_by")
    order_by = spec.get("order_by")
    if group_by is None or order_by is None:
        raise ValueError(
            "trajectories spec requires both 'group_by' and 'order_by' columns"
        )
    if x_col is None or y_col is None:
        raise ValueError(
            "trajectories require x and y projection columns to be specified"
        )

    max_groups = int(spec.get("max_groups", 50))
    width = spec.get("width")
    opacity = spec.get("opacity")
    color_by = spec.get("color_by")
    colors = spec.get("colors") or {}

    select_color = (
        f", any_value({_quote_ident(color_by)}) AS color_val" if color_by else ""
    )

    sql = f"""
        SELECT
            {_quote_ident(group_by)} AS group_id,
            list({_quote_ident(x_col)} ORDER BY {_quote_ident(order_by)}) AS xs,
            list({_quote_ident(y_col)} ORDER BY {_quote_ident(order_by)}) AS ys,
            count(*) AS n
            {select_color}
        FROM {table}
        GROUP BY {_quote_ident(group_by)}
        HAVING n >= 2
        ORDER BY n DESC
        LIMIT {max_groups}
    """
    rows = connection.sql(sql).fetchall()

    cols = [desc[0] for desc in connection.sql(sql).description]
    group_idx = cols.index("group_id")
    xs_idx = cols.index("xs")
    ys_idx = cols.index("ys")
    color_idx = cols.index("color_val") if color_by else None

    result: list[dict] = []
    for row in rows:
        group_id = row[group_idx]
        xs = row[xs_idx]
        ys = row[ys_idx]
        if xs is None or ys is None or len(xs) < 2:
            continue
        traj: dict = {
            "id": group_id,
            "points": [
                {"x": float(xv), "y": float(yv)}
                for xv, yv in zip(xs, ys)
                if xv is not None and yv is not None
            ],
        }
        if width is not None:
            traj["width"] = float(width)
        if opacity is not None:
            traj["opacity"] = float(opacity)
        if color_idx is not None:
            color_value = row[color_idx]
            mapped = colors.get(color_value) if color_value is not None else None
            if mapped is not None:
                traj["color"] = str(mapped)
        result.append(traj)
    return result


class EmbeddingAtlasWidget(anywidget.AnyWidget):
    """An Embedding Atlas widget in notebooks"""

    _esm = pathlib.Path(__file__).parent / "widget_static" / "anywidget" / "index.js"

    # The props to the embedding atlas component, internal use only
    _props = traitlets.Dict({}).tag(sync=True)

    # The state of the embedding atlas component, internal use only
    _state = traitlets.Any(None).tag(sync=True)
    _predicate = traitlets.Any(None).tag(sync=True)

    def __init__(
        self,
        data_frame: Any,
        *,
        connection: duckdb.DuckDBPyConnection | None = None,
        **options: Unpack[EmbeddingAtlasOptions],
    ):
        """
        Create an Embedding Atlas widget.

        Args:
            data_frame:
                A DataFrame/Arrow object to "register" with DuckDB.

            row_id:
                The column name for row id (if not specified, a row id column will be added).

            x:
                The column name for X axis in the embedding.

            y:
                The column name for Y axis in the embedding.

            text:
                The column name for the textual data.

            neighbors:
                The column name containing precomputed K-nearest neighbors for each point.
                Each value in the column should be a dictionary with the format:
                ``{ "ids": [id1, id2, ...], "distances": [distance1, distance2, ...] }``.

                - ``"ids"`` should be an array of row ids of the neighbors
                  (if ``row_id`` is specified, match the value in row_id, otherwise use zero-based row index),
                  sorted by distance.
                - ``"distances"`` should contain the corresponding distances to each neighbor.

            labels:
                Labels for the embedding view. Set to string ``"automatic"`` to generate labels automatically, or ``"disabled"`` to disable auto labels.
                Automatic labels are generated by clustering the 2D density distribution and selecting
                representative keywords using TF-IDF ranking.
                You can also pass in a list of labels. Each label must contain ``x`` and ``y`` coordinates
                and ``text`` for the label content. Optionally, you may specify an integer ``level`` to roughly
                control the zoom level where the label appears, and `priority` for the label's priority.
                Higher priority labels have a better chance to appear when multiple labels overlap.

            stop_words:
                Stop words for automatic label generation.

            point_size:
                Override the default point size for the embedding view.

            show_table:
                Whether to display the data table when the widget opens.

            show_charts:
                Whether to display charts when the widget opens.

            color:
                The column name to use for coloring points in the embedding view.

            default_charts_include:
                If provided, only these columns will appear as auto-generated charts.

            default_charts_exclude:
                Columns to exclude from auto-generated charts.

            show_embedding:
                Whether to display the embedding view when the widget opens.

            connection (DuckDBPyConnection, optional):
                A DuckDB connection. Defaults to duckdb.connect().
        """

        _ = data_frame  # used by DuckDB

        table_name = "embedding_atlas"
        row_id_column = options.get("row_id", "__row_index__")

        if connection is None:
            connection = duckdb.connect()

        connection.sql(
            f"CREATE TEMPORARY TABLE {table_name} AS SELECT * FROM data_frame"
        )

        if options.get("row_id") is None:
            # Create the row_id_column if it does not exist.
            connection.sql(
                f"""
                CREATE TEMPORARY SEQUENCE row_id_sequence MINVALUE 0 START 0;
                ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {row_id_column} INTEGER DEFAULT nextval('row_id_sequence');
                """
            )

        trajectories = options.get("trajectories")
        if isinstance(trajectories, dict):
            resolved = _compute_trajectories_from_spec(
                connection,
                table_name,
                x_col=options.get("x"),
                y_col=options.get("y"),
                spec=trajectories,
            )
            options = {**options, "trajectories": resolved}

        props = make_embedding_atlas_props(
            **(options | {"table": table_name, "row_id": row_id_column}),
        )

        super().__init__()

        self._props = props

        self._connection: duckdb.DuckDBPyConnection = connection
        self._table_name = table_name
        self.on_msg(self._handle_custom_msg)

    def selection(self, format: str = "dataframe") -> Any:
        """
        Returns the current selection in the widget.

        Args:
            format: the format of the returned selection, 'dataframe', 'arrow', or 'predicate'
        """
        if self._predicate is not None:
            self._connection.execute(
                f"SELECT * FROM {self._table_name} WHERE {self._predicate}"
            )
        else:
            self._connection.execute(f"SELECT * FROM {self._table_name}")
        if format == "dataframe":
            return self._connection.fetch_df()
        elif format == "arrow":
            return self._connection.fetch_arrow_table()
        else:
            raise ValueError(
                "invalid format, supported options are 'dataframe', 'arrow', and 'predicate'"
            )

    def _handle_custom_msg(self, content: dict, buffers: list):
        uuid = content["uuid"]
        sql = content["sql"]
        command = content["type"]

        try:
            if command == "arrow":
                result = self._connection.query(sql).arrow()
                buf = arrow_to_bytes(result)
                self.send({"type": "arrow", "uuid": uuid}, buffers=[buf])
            elif command == "exec":
                self._connection.execute(sql)
                self.send({"type": "exec", "uuid": uuid})
            elif command == "json":
                result = self._connection.query(sql).df()
                json = result.to_dict(orient="records")
                self.send({"type": "json", "uuid": uuid, "result": json})
            else:
                raise ValueError(f"Unknown command {command}")
        except Exception as e:
            self.send({"error": str(e), "uuid": uuid})
