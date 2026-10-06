using SDFLibrary
using CutCellMethods
using CutCellMethods: calc_volume
using MeshLibrary
using CartesianMeshes

outdir = joinpath(@__DIR__, "outputs")

# A fuselage STL and the z = 0 plane. `SDFPlane` is the **half-space** behind the plane, not a
# zero-thickness sheet -- a sheet has no inside, so there would be nothing for a boolean to keep or
# remove. With `normal = +z`, everything at `z < 0` is inside it.
mesh = load_mesh(joinpath(@__DIR__, "inputs", "AMRA_Subscale_v2_fuselage.stl"))
geo = SDFMesh(mesh, cache=TriGrid, target_tris_per_cell=4)

plane = SDFPlane(point=SVector(0.0, 0.0, 0.0), normal=SVector(0.0, 0.0, 1.0))

# `SDFSubtract(target, tool)` keeps `target` and removes everything inside `tool`, and the two roles
# are not interchangeable -- unlike `SDFUnion`, swapping the arguments gives a completely different
# solid. Here the half-space is the body and the fuselage is the cutter, so the result is the mould:
# solid below z = 0, with a fuselage-shaped cavity pressed into it.
#
# `smoothing` is a blend width in units of length, as in `SDFUnion`, but blended with a soft
# *maximum* instead: it rounds the rim where the cavity breaks the surface into a groove rather than
# a crease. 0.01 against a body 7.6 long is well under one cell here, which is where you want it --
# the blend is the one part of the field that isn't a true distance. `smoothing = 0` gives the exact
# (creased) difference back.
geo_mould = SDFSubtract(plane, geo; smoothing=0.0)

# The cutter mesh itself, for reference in ParaView -- open it inside the mould to see that the
# cavity is its negative.
write_vtk(mesh, joinpath(outdir, "subtract_tool"))

# The grid, sized from the *fuselage's* bounding box rather than the subtraction's. `SDFSubtract`'s
# box is its target's, and this target is the plane, whose box is unbounded in every direction --
# `bounding_grid` rightly refuses it, so the extent has to come from the one bounded thing in the
# tree. 10% of each axis's extent is added at both ends so the cavity has clear space around it.
#
# `CartesianIsoGrid` rather than `bounding_grid` because PLIC needs cubic cells: one cell size is
# what converts a distance into unit-cell coordinates. `sz` asks for ~200 cells along the longest
# axis, and the grid rounds every axis up to a power of two at that size, so it covers rather more
# than the padded box. One grid for both reconstructions below, so they are directly comparable.
lo, hi = get_bounding_box(geo)
pad = 0.1 .* (hi .- lo)
lo, hi = lo .- pad, hi .+ pad
grid = CartesianIsoGrid(mincorner=Tuple(lo), maxcorner=Tuple(hi), sz=maximum(hi .- lo) / 200)

println("grid: ", grid.n, " cells of ", grid.d, " over ", Tuple(lo), " to ", Tuple(hi))

# Marching cubes: samples the field at every grid node and reconstructs `sdf == 0` as an
# outward-facing triangle mesh. It needs nothing from the geometry but `get_sdf`, which is the only
# way to mesh a CSG tree -- neither child has a surface that survives the boolean intact.
#
# The half-space fills the lower half of the grid, so this surface runs off the edges and is left
# open where the grid clips it. That is expected here, not a sign of anything wrong: an unbounded
# target has no grid that would close it. The cavity itself, which is the part worth looking at,
# closes perfectly well inside.
#
# Marching cubes draws from a cache: update one from the geometry, then march the nodal samples it
# kept. It is the same cache a cut-cell solver reads its volume fractions and apertures from.
method_mc = MarchingCubesCutCell()
cache_mc = update_cache!(allocate_cache(grid, method_mc), geo_mould, grid)
mesh_mc = generate_mesh(cache_mc, grid)
write_vtk(mesh_mc, joinpath(outdir, "subtract_fuselage_mould"))
println("marching cubes: $(length(mesh_mc.elements)) triangles (open at the grid boundary, by construction)")

# The same body as PLIC sees it: one plane fitted per cell from a single `get_sdf` at that cell's
# centroid, then the cell clipped against it. That is the reconstruction `calc_volume` integrates
# and the one a VOF solver's interface is made of, so it comes out as a field of per-cell shards
# rather than a joined-up surface -- open it next to the marching cubes mesh to see what the nodal
# construction buys, especially along the rim where the cavity meets the flat top of the mould.
#
# Like marching cubes, PLIC draws from a cache: its update fits every cell's plane, and
# `generate_mesh` clips those fits in the cells the interface actually crosses.
method = PLICCutCell()
cache_plic = update_cache!(allocate_cache(grid, method), geo_mould, grid)
mesh_plic = generate_mesh(cache_plic, grid)
write_vtk(mesh_plic, joinpath(outdir, "subtract_fuselage_mould_plic"))
println("plic: $(length(mesh_plic.elements)) facets")

# The same fit read the other way: the cache's `cells` hold each cell's fraction, face fractions
# and classification, as plain arrays per field.
#
# `volume_fraction` is the **fluid** fraction, the part of the cell the body does *not* fill, so the
# VOF field the mould occupies is its complement -- and summing that reproduces `calc_volume` to
# rounding, since it is the same per-cell fit integrated. Here it is the grid's lower half less the
# part of the fuselage that was pressed into it.
vof = 1 .- cache_plic.cells.volume_fraction
CartesianMeshes.write_vtk(joinpath(outdir, "subtract_fuselage_mould_vof"), grid;
                          cell_data=Dict("vof" => vof))
@show sum(vof) * prod(grid.d) calc_volume(geo_mould, grid)
