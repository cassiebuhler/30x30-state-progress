import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

def checkerboard_geom(
    gdf,
    type_col,
    state_col="state_id",
    tile_m=50_000,          # ~50 km squares
    work_crs="EPSG:5070",   # planar meters CRS for US
):
    gdf = gdf.copy()
    geom_col = gdf.geometry.name
    src_crs = gdf.crs

    gdf[type_col] = gdf[type_col].astype("string")

    # states with exactly 2 unique (non-null) types
    types = (gdf.dropna(subset=[type_col])
               .groupby(state_col)[type_col]
               .unique()
               .apply(lambda a: sorted(map(str, a))))
    target = types[types.apply(len) == 2]
    if target.empty:
        return gdf

    # work in meters
    gdf_m = gdf.to_crs(work_crs)

    out = gdf_m[~gdf_m[state_col].isin(target.index)].copy()
    pieces = []

    for st, (t1, t2) in target.items():
        state_rows = gdf_m[gdf_m[state_col] == st]
        poly = state_rows[geom_col].unary_union

        minx, miny, maxx, maxy = poly.bounds

        # build square grid with fixed tile size
        xs = np.arange(minx, maxx + tile_m, tile_m)
        ys = np.arange(miny, maxy + tile_m, tile_m)

        grid_geoms = []
        parity = []
        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                grid_geoms.append(box(xs[i], ys[j], xs[i+1], ys[j+1]))
                parity.append((i + j) % 2)

        grid = gpd.GeoDataFrame({"parity": parity}, geometry=grid_geoms, crs=work_crs)
        state_gdf = gpd.GeoDataFrame({state_col: [st]}, geometry=[poly], crs=work_crs)

        clipped = gpd.overlay(grid, state_gdf, how="intersection")
        g_even = clipped.loc[clipped["parity"] == 0].unary_union
        g_odd  = clipped.loc[clipped["parity"] == 1].unary_union

        base = state_rows.iloc[0].copy()

        r1 = base.copy(); r1[type_col] = t1; r1[geom_col] = g_even
        r2 = base.copy(); r2[type_col] = t2; r2[geom_col] = g_odd
        pieces.extend([r1, r2])

    out = pd.concat([out, gpd.GeoDataFrame(pieces, geometry=geom_col, crs=work_crs)], ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry=geom_col, crs=work_crs)

    # back to original CRS
    if src_crs is not None:
        out = out.to_crs(src_crs)
    return out


