// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Coordinator } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";

import { boundingRect, type Point, type Rectangle } from "../utils.js";
import type { DataField, DataPoint, DataPointID, Trajectory, TrajectorySpec } from "./types.js";

export function predicateForDataPoints(
  source: { x: string; y: string; identifier?: string | null; category?: string | null },
  points: DataPoint[],
) {
  if (points.length == 0) {
    return SQL.literal(false);
  }
  if (source.identifier != null) {
    let identifier = source.identifier;
    return SQL.or(...points.map((p) => SQL.eq(SQL.column(identifier), SQL.literal(p.identifier))));
  } else {
    let x = source.x;
    let y = source.y;
    let category = source.category;
    if (category != null) {
      return SQL.or(
        ...points.map((p) =>
          SQL.and(
            SQL.eq(SQL.cast(SQL.column(x), "DOUBLE"), SQL.literal(p.x)),
            SQL.eq(SQL.cast(SQL.column(y), "DOUBLE"), SQL.literal(p.y)),
            SQL.eq(SQL.cast(SQL.column(category), "INTEGER"), SQL.literal(p.category)),
          ),
        ),
      );
    } else {
      return SQL.or(
        ...points.map((p) =>
          SQL.and(
            SQL.eq(SQL.cast(SQL.column(x), "DOUBLE"), SQL.literal(p.x)),
            SQL.eq(SQL.cast(SQL.column(y), "DOUBLE"), SQL.literal(p.y)),
          ),
        ),
      );
    }
  }
}

function pointInPolygonPredicate(x: SQL.ExprNode, y: SQL.ExprNode, polygon: Point[]) {
  // Equavilant algorithm:
  // let counter = 0;
  // for each edge (x1, y1) to (x2, y2)) {
  //   if ((y1 <= y) && (y < y2) || (y2 <= y) && (y < y1)) { // -> pred1
  //     if ((x2 - x1) * (y - y1) / (y2 - y1) + x1 < x) {    // -> pred2
  //       counter += 1;
  //     }
  //   }
  // }
  // return counter % 2 == 1;

  let parts: SQL.ExprNode[] = [];
  for (let i = 0; i < polygon.length; i++) {
    let j = (i + 1) % polygon.length;
    let { x: x1, y: y1 } = polygon[i];
    let { x: x2, y: y2 } = polygon[j];
    let pred1 =
      y1 < y2
        ? SQL.and(SQL.lte(SQL.literal(y1), y), SQL.lt(y, SQL.literal(y2)))
        : SQL.and(SQL.lte(SQL.literal(y2), y), SQL.lt(y, SQL.literal(y1)));
    let pred2 = (y1 < y2 ? SQL.lt : SQL.gt)(
      SQL.sub(SQL.mul(SQL.literal(x2 - x1), y), SQL.mul(SQL.literal(y2 - y1), x)),
      SQL.literal((x2 - x1) * y1 - (y2 - y1) * x1),
    );
    parts.push(SQL.cast(SQL.and(pred1, pred2), "INT"));
  }
  let sum = parts.reduce((a, b) => SQL.add(a, b));
  return SQL.eq(SQL.mod(sum, SQL.literal(2)), SQL.literal(1));
}

export function predicateForRangeSelection(source: { x: string; y: string }, range: Rectangle | Point[]) {
  if (range instanceof Array) {
    if (range.length < 3) {
      // Degenerate case.
      return SQL.literal(false);
    }
    let bounds = boundingRect(range);
    return SQL.and(
      SQL.isBetween(SQL.column(source.x), [bounds.xMin, bounds.xMax]),
      SQL.isBetween(SQL.column(source.y), [bounds.yMin, bounds.yMax]),
      pointInPolygonPredicate(SQL.column(source.x), SQL.column(source.y), range),
    );
  } else {
    return SQL.and(
      SQL.isBetween(SQL.column(source.x), [range.xMin, range.xMax]),
      SQL.isBetween(SQL.column(source.y), [range.yMin, range.yMax]),
    );
  }
}

export async function queryApproximateDensity(
  coordinator: Coordinator,
  source: { table: string; x: string; y: string; category?: string | null },
): Promise<{
  centerX: number;
  centerY: number;
  scaler: number;
  totalCount: number;
  categoryCount: number;
  maxDensity: number;
}> {
  let { x, y, table } = source;
  // Find the view transform that fits all data points in a square view.
  let r = await coordinator.query(
    SQL.Query.from(table).select({
      centerX: SQL.sql`MEDIAN(${SQL.column(x)})`,
      centerY: SQL.sql`MEDIAN(${SQL.column(y)})`,
      stdX: SQL.sql`STDDEV(${SQL.column(x)})`,
      stdY: SQL.sql`STDDEV(${SQL.column(y)})`,
      ...(source.category != null
        ? {
            maxCategory: SQL.sql`MAX(${SQL.column(source.category)}::UTINYINT)`,
          }
        : {}),
    }),
  );
  let { centerX, centerY, stdX, stdY, maxCategory } = r.get(0);
  let scaler = 1.0 / (Math.max(stdX, stdY, 1e-3) * 3);

  // Estimate maximum density.
  // This is the approximate max number of points per square unit in data dimensions.
  let binWidth = 0.1 / scaler;
  let xBinClause = SQL.sql`FLOOR((${SQL.column(x)} - ${centerX}) / ${binWidth})`;
  let yBinClause = SQL.sql`FLOOR((${SQL.column(y)} - ${centerY}) / ${binWidth})`;
  let categoryClause = source.category != null ? SQL.column(source.category) : null;
  let groupby = categoryClause != null ? [xBinClause, yBinClause, categoryClause] : [xBinClause, yBinClause];
  let q = SQL.Query.from(
    SQL.Query.from(table)
      .select({ count: SQL.sql`COUNT(*)` })
      .groupby(...groupby),
  ).select({
    totalCount: SQL.sql`SUM(count)::INT`,
    maxCount: SQL.sql`MAX(count)::INT`,
  });

  r = await coordinator.query(q);
  let { maxCount, totalCount } = r.get(0);
  let maxDensity = maxCount / (binWidth * binWidth);

  return {
    centerX,
    centerY,
    scaler,
    totalCount,
    categoryCount: (maxCategory ?? 0) + 1,
    maxDensity,
  };
}

export interface DataPointQuerySource {
  table: string;
  x: string;
  y: string;
  category?: string | null;
  text?: string | null;
  identifier?: string | null;
  additionalFields?: Record<string, DataField> | null;
}

export class DataPointQuery {
  coordinator: Coordinator;
  source: DataPointQuerySource;
  lastDistance: number;
  selectParams: Record<string, SQL.ExprNode>;

  constructor(coordinator: Coordinator, source: DataPointQuerySource) {
    this.coordinator = coordinator;
    this.source = source;
    this.lastDistance = 0;

    let { x, y, category, text, identifier } = this.source;
    let fieldExpressions: Record<string, SQL.ExprNode> = {};
    let fields = source.additionalFields ?? {};
    for (let key in fields) {
      let spec = fields[key];
      if (typeof spec == "string") {
        fieldExpressions["field_" + key] = SQL.column(spec);
      } else {
        fieldExpressions["field_" + key] = SQL.sql`${spec.sql}`;
      }
    }
    this.selectParams = {
      x: SQL.sql`${SQL.column(x)}::DOUBLE`,
      y: SQL.sql`${SQL.column(y)}::DOUBLE`,
      ...(category != null ? { category: SQL.sql`${SQL.column(category)}::INT` } : {}),
      ...(text != null ? { text: SQL.sql`${SQL.column(text)}` } : {}),
      ...(identifier != null ? { identifier: SQL.sql`${SQL.column(identifier)}` } : {}),
      ...fieldExpressions,
    };
  }

  _convertToDataPoint(row: any): DataPoint {
    let fields: Record<string, any> = {};
    for (let key in row) {
      if (key.startsWith("field_")) {
        fields[key.slice("field_".length)] = row[key];
      }
    }
    return {
      x: row.x,
      y: row.y,
      category: row.category,
      text: row.text,
      identifier: row.identifier,
      fields: fields,
    };
  }

  async queryClosestPoint(
    predicate: any | null,
    px: number,
    py: number,
    unitDistance: number,
  ): Promise<DataPoint | null> {
    let rMax = unitDistance * 12;
    let { x, y } = this.source;

    for (let r of [this.lastDistance, rMax]) {
      if (r == 0 || r > rMax) {
        continue;
      }
      let q = SQL.Query.from(this.source.table).select(this.selectParams);
      q = q.where(SQL.sql`${SQL.column(x)} BETWEEN ${px - r} AND ${px + r}`);
      q = q.where(SQL.sql`${SQL.column(y)} BETWEEN ${py - r} AND ${py + r}`);
      if (predicate) {
        q = q.where(predicate);
      }
      q = q.orderby(SQL.sql`(x - (${px}))**2 + (y - (${py}))**2`).limit(1);
      let result = await this.coordinator.query(q);
      let point = result.get(0);
      if (point) {
        this.lastDistance = Math.max(Math.abs(point.x - px), Math.abs(point.y - py)) * 4;
        return this._convertToDataPoint(point);
      }
    }
    return null;
  }

  async queryPoints(identifiers: DataPointID[]): Promise<DataPoint[]> {
    let { table, identifier } = this.source;
    if (identifier == null) {
      return [];
    }
    let q = SQL.Query.from(table).select(this.selectParams);
    q = q.where(
      SQL.isIn(
        SQL.column(identifier),
        identifiers.map((x) => SQL.literal(x)),
      ),
    );
    let result = Array.from(await this.coordinator.query(q));
    return result.map((row) => this._convertToDataPoint(row));
  }
}

export interface TrajectoryQuerySource {
  table: string;
  x: string;
  y: string;
}

/** Build the SQL that aggregates the data table into trajectory rows under the
 *  given Mosaic predicate. A window function assigns each row a per-group step
 *  index over the *unfiltered* ordering; the predicate is then applied. The
 *  caller compares consecutive step indices in the result to detect rows that
 *  were filtered out of the middle of a trajectory and break the polyline at
 *  those gaps (see `parseTrajectoryResult`). */
export function buildTrajectoryQuery(source: TrajectoryQuerySource, spec: TrajectorySpec, predicate: any | null) {
  let maxGroups = spec.max_groups ?? 50;
  let groupCol = SQL.column(spec.group_by);
  let orderCol = SQL.column(spec.order_by);
  let xCol = SQL.column(source.x);
  let yCol = SQL.column(source.y);

  let inner = SQL.Query.from(source.table).select("*", {
    __traj_step: SQL.sql`row_number() OVER (PARTITION BY ${groupCol} ORDER BY ${orderCol})`,
  });

  let outerSelect: Record<string, any> = {
    group_id: groupCol,
    xs: SQL.sql`list(${xCol}::DOUBLE ORDER BY "__traj_step")`,
    ys: SQL.sql`list(${yCol}::DOUBLE ORDER BY "__traj_step")`,
    steps: SQL.sql`list("__traj_step" ORDER BY "__traj_step")`,
    n: SQL.sql`count(*)::INT`,
  };
  if (spec.color_by) {
    outerSelect.color_val = SQL.sql`any_value(${SQL.column(spec.color_by)})`;
  }

  let q = SQL.Query.from(inner).select(outerSelect).groupby(groupCol);
  if (predicate) {
    q = q.where(predicate);
  }
  q = q
    .having(SQL.sql`count(*) >= 2`)
    .orderby(SQL.sql`count(*) DESC`)
    .limit(maxGroups);
  return q;
}

/** Materialize the Arrow result from `buildTrajectoryQuery` into a
 *  `Trajectory[]`. Inserts a `{x: NaN, y: NaN}` break-point between any two
 *  consecutive points whose step indices are not adjacent in the original
 *  ordering, so the renderer draws disconnected segments at filtered-out gaps
 *  (see `buildTrajectorySegmentData` — it skips segments with non-finite
 *  endpoints). */
export function parseTrajectoryResult(data: any, spec: TrajectorySpec): Trajectory[] {
  let groupCol = data.getChild("group_id");
  let xsCol = data.getChild("xs");
  let ysCol = data.getChild("ys");
  let stepsCol = data.getChild("steps");
  let colorCol = spec.color_by ? data.getChild("color_val") : null;
  let colors = spec.colors ?? {};
  let result: Trajectory[] = [];
  let rowCount = data.numRows;
  for (let i = 0; i < rowCount; i++) {
    let xs = xsCol?.get(i);
    let ys = ysCol?.get(i);
    let steps = stepsCol?.get(i);
    if (xs == null || ys == null || steps == null) {
      continue;
    }
    let len = Math.min(xs.length, ys.length, steps.length);
    if (len < 2) {
      continue;
    }
    let points: { x: number; y: number }[] = [];
    let prevStep: number | null = null;
    for (let j = 0; j < len; j++) {
      let xv = xs[j];
      let yv = ys[j];
      let step = Number(steps[j]);
      if (xv == null || yv == null || !Number.isFinite(Number(xv)) || !Number.isFinite(Number(yv))) {
        continue;
      }
      if (prevStep != null && step !== prevStep + 1) {
        points.push({ x: NaN, y: NaN });
      }
      points.push({ x: Number(xv), y: Number(yv) });
      prevStep = step;
    }
    if (points.length < 2) {
      continue;
    }
    let traj: Trajectory = {
      id: groupCol?.get(i) ?? undefined,
      points,
    };
    if (spec.width != null) {
      traj.width = spec.width;
    }
    if (spec.opacity != null) {
      traj.opacity = spec.opacity;
    }
    if (colorCol != null) {
      let value = colorCol.get(i);
      let mapped = value != null ? colors[String(value)] : undefined;
      if (mapped != null) {
        traj.color = mapped;
      }
    }
    result.push(traj);
  }
  return result;
}
