# CutCellMethods.jl

Cut-cell geometry on Cartesian grids: per-cell volume fractions, face apertures, centroids and
interface facets, plus the reconstructed surface. The bodies being cut are
[SDFLibrary.jl](https://github.com/tidalflight/SDFLibrary.jl) geometries, or any callable
`x -> phi` that is negative inside.

## The API is a cache

Allocate a cache once for a grid, update it in place as the body moves, and read it:

```julia
using SDFLibrary, CutCellMethods, CartesianMeshes, StaticArrays

geo  = SDFCircle(center=SVector(0.0, 0.0), radius=0.5)
grid = CartesianIsoGrid(mincorner=(-1, -1), maxcorner=(1, 1), sz=0.01)

cache = allocate_cache(grid, MarchingSquaresCutCell(); backend=CPU())   # or CUDABackend()
update_cache!(cache, geo, grid)          # grid may move between updates; its `n` may not

cache.cells.volume_fraction              # one field, as a plain array
cache.cells[ci]                          # one cell's whole CutCellData
surface = generate_mesh(cache, grid)     # the contour, a Mesh{2} of Line elements
```

`method` is one of three stateless tags, all `<: AbstractCutCellMethod`:

| tag | dimension | reconstruction | `cache.cells` holds |
|---|---|---|---|
| `MarchingSquaresCutCell()` | 2D | nodal, from the cell's 4 corner values | `CutCellData{2,T,4,1}` |
| `MarchingCubesCutCell()` | 3D | nodal, from the cell's 8 corner values | `CutCellData{3,T,6,2}` |
| `PLICCutCell()` | 2D and 3D | one plane per cell, fitted at its centroid | `CutCellData{D,T,2D,D-1}` |

`geo` is an `AbstractSDFGeometry` or a callable `x -> phi`; PLIC needs an `AbstractSDFGeometry`,
since it reads the normal from `get_sdf`. The domain is a `CartesianGrid`, or for
`MarchingSquaresCutCell` also an `AdaptiveMesh{2}`, where the cache holds one entry per leaf and
`generate_mesh(cache, mesh)` draws the contour over the tree.

A tree cache is sized for the tree's leaves as they are: after refining or coarsening, allocate
it again. A device tree cache is updated against the tree adapted to that device
(`adapt(CuArray, mesh)`).

## What a cache holds

Every cache holds a `CutCellData` for every cell as a `StructArray` in `cells`. A new cache is
seeded as no body: every cell outside, all fluid, every face open. `update_cache!` synchronizes
before it returns. Beyond `cells`:

| method | extra field | holds |
|---|---|---|
| `MarchingSquaresCutCell`, `MarchingCubesCutCell` | `phi` | the field at every node; `generate_mesh(cache, grid)` marches it, drawing the same reconstruction |
| `PLICCutCell` | `normals`, `intercepts`, `is_valid` | every cell's fitted plane and its degenerate-fit flag, as plain arrays sized `grid.n`. This is the only place the plane is kept. `generate_mesh(cache, grid)` clips these |

The PLIC cache's `cells.face_fraction` holds the **resolved** face fractions, the mean of each
face's two one-sided candidates. A centroid plane fit has no centroids to report, so the PLIC
cells' `centroid`, `face_centroid_local` and `interface_centroid` are `NaN`.

The full struct costs about 200 bytes a cell in 3D at `Float64`.

## Exports are deliberately narrow

Only the method tags, `CutCellData`, `allocate_cache`/`update_cache!` and `generate_mesh`
(re-exported from MeshLibrary.jl) are exported. Everything else is qualified:

| call | meaning |
|---|---|
| `CutCellMethods.face_fractions(cache)`, `volume_fractions`, `kinds` | the three fields every cache holds, as its own storage |
| `CutCellMethods.face_centroids_local(cache)` | each cell's open-face centroids (in-plane offsets), as its own storage; `NaN` under PLIC |
| `CutCellMethods.face_area_of(cell, dir, cellsize)` | open measure, raw units (length in 2D, area in 3D) |
| `CutCellMethods.full_face_area(cellsize, dir)` | a fully open face's measure, the scale a `face_fraction` multiplies |
| `CutCellMethods.is_cell_open(cell)`, `is_cell_open(cache, ci)` | connectivity, **not** `kind`; the cache form reads `face_fractions` alone, so it sees in-place edits |
| `CutCellMethods.is_cut`, `is_inside`, `is_outside`, `is_ambiguous` | classification |
| `CutCellMethods.face_centroid(cell, dir, nodes)`, `face_centroid(cell, dir, mesh, i)` | an open face's centroid in global coordinates, from the stored in-plane offsets |
| `CutCellMethods.interface_area(cell, cellsize)`, `interface_normal`, `interface_normal_area` | the interface facet, formed from the face fractions |
| `CutCellMethods.closure_residual(m, cellsize)`, `cut_cell_report(cache.cells)` | verification |
| `CutCellMethods.calc_volume(geo, grid)` | the enclosed volume, from the PLIC fit |
| `CutCellMethods.cell_indices(mesh, grid)` | which cell a marched mesh's triangle came from |
| `CutCellMethods.CELL_INSIDE`, `CELL_OUTSIDE`, `CELL_CUT` | the `kind` encoding |

The `face_*` readers take one `CutCellData`, whichever method or cache it came from.
`cellsize` is the cell's own extent: `grid.d`, or `get_elem_size(mesh, cell)` on a tree.

## The traps

These go wrong silently.

- **PLIC face fractions are resolved, not a cell's own.** Each plane is fitted from its own cell,
  so the two cells sharing a face have different candidates for it wherever their fits disagree.
  The cache stores their mean, which is bitwise single-valued because `+` is commutative. Its
  `centroid`, `face_centroid_local` and `interface_centroid` are `NaN`. The nodal methods need no
  resolution: both cells interpolate the same corner values, so a shared face agrees bitwise by
  construction.
- **`volume_fraction` is the fluid fraction** — the part of the cell outside the body. The solid
  fraction `calc_volume` integrates is its complement. `kind` follows the fluid reading:
  `CELL_INSIDE` at `0`, `CELL_OUTSIDE` at `1`, `CELL_CUT` strictly between.
- **Two samplings of one field agree only to roundoff.** A nodal cache samples its `phi` at
  `CartesianMeshes.get_node`, while `SDFLibrary.sample_sdf(geo, grid)` builds node coordinates from
  `grid_lines`; the two differ by about an ulp per node. Each is watertight within itself, so do
  not compare one against the other bitwise.
- **Bitwise claims hold within one route and one backend.** `closure_residual` is exactly zero and
  shared faces agree bitwise on one backend. Across CPU and GPU, compare integers exactly and floats
  with a tight `isapprox`.
- **PLIC requires an isotropic grid.** The PLIC cache's `allocate_cache` and `update_cache!` throw
  on an anisotropic one.
- **Faces are indexed by direction:** `1 = -x`, `2 = +x`, `3 = -y`, `4 = +y`, `5 = -z`, `6 = +z`.
  `CartesianMeshes.direction_axis` and `direction_sign` decode them.
- **The two normal senses are opposite, on purpose.** Open-face normals point out of the fluid
  region; `interface_normal` points out of the body. The divergence theorem over the fluid region
  is `sum_k A_k n_k - A_int n_int == 0`, which is what `closure_residual` returns. A flux balance
  negates the interface normal at the point of use.
- **`ambiguous` cells** (a saddle, or an ambiguous cube interior) have a lumped `interface_normal`:
  right for conservation, wrong for a surface flux. Count them and refine them away.

## Which reconstruction to use

- **A cut-cell flux scheme:** a nodal method. Shared faces are single-valued, the closure identity
  is exact per cell, and the volume, apertures and interface come from one construction.
- **A VOF solver's interface:** `PLICCutCell`. It is what `calc_volume` integrates, and its surface
  is a field of disconnected per-cell shards, which is what a VOF solver believes.
- **A picture of the body:** `generate_mesh` with a nodal method. It is watertight and wound
  outwards, so it can be fed back to `SDFMesh` as a geometry.

Nodal reconstructions chamfer sharp features to within one cell. Either resolve the feature or give
the geometry a `smoothing` comparable to the cell size.

## Reference

The invariants above are asserted in `test/`, bitwise where the text says bitwise.
`dev/dev_mc_moments.jl` checks the marching-cubes moments against the whole-grid march.
