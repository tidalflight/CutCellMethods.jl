using SDFLibrary
using CutCellMethods
using Printf

outdir = joinpath(@__DIR__, "outputs")

# A large sphere with a square-section bar driven straight through the middle of it. The bar is
# 3.0 long against the sphere's diameter of 2.0, so it passes entirely through and sticks out both
# ends. `SDFRectangle` is the 2D quadrilateral -- the 3D box is `SDFRectPrism`, and `SDFUnion` insists
# on the match: every geometry in a union must agree on `Base.ndims`.
sphere = SDFSphere(center=SVector(0.0, 0.0, 0.0), radius=1.0)
bar    = SDFRectPrism(center=SVector(0.0, 0.0, 0.0), dims=SVector(3.0, 0.5, 0.5))

# `smoothing` is a blend width in units of length. It rounds the seams where the bodies meet into
# fillets, using a soft minimum in place of the exact union's `min`; `smoothing = 0` gives that
# exact union back, creases and all. Any number of geometries can go in, and `SDFUnion` is itself
# an `AbstractSDFGeometry`, so unions nest.
geo = SDFUnion(sphere, bar; smoothing=0.1)

println("SDFUnion of $(length(geo.geoms)) geometries, $(ndims(geo))D, smoothing = $(geo.smoothing)\n")

# It evaluates like every other geometry here: `(signed_distance, normal)`, distance negative
# inside, normal pointing outwards. The normal is the gradient of the *blended* field, so it turns
# continuously across a fillet rather than jumping between the two bodies.
@printf("%26s  %10s  %s\n", "point", "sdf", "normal")
for x in (SVector(0.5, 0.1, 0.0),     # inside both bodies
          SVector(1.4, 0.0, 0.0),     # inside the bar, past the sphere
          SVector(1.0, 0.4, 0.0),     # just outside, near a seam
          SVector(0.0, 2.0, 0.0))     # well clear of everything
    d, n = get_sdf(geo, x)
    @printf("%26s  %10.5f  %s\n", string(Tuple(x)), d, string(round.(Tuple(n), digits=4)))
end

# The union's bounding box merges the children's and pads them, because a fillet bulges into the
# gap between two bodies and can sit outside both.
lo, hi = get_bounding_box(geo)
println("\nbounding box: ", Tuple(lo), " to ", Tuple(hi))

# `generate_mesh(geo)` on a union only concatenates the children's meshes, which keeps the internal
# surfaces the union removes and misses the fillets `smoothing` adds. Instead reconstruct it: a
# marching-cubes cache samples the field at every node of a grid, and `generate_mesh(cache, grid)`
# marches the `sdf == 0` surface out of those samples, which needs nothing from the geometry but
# `get_sdf`. `bounding_grid` builds the grid from the body's own bounding box, so the only choice to
# make is `N`, the number of cells per axis.
#
# The cache is the method's whole-grid state: its update also builds every cell's moments, about
# 200 bytes a cell (1.6 GB at `N = 200`). One mesh per smoothing value, so the seams can be compared.
println()
mc = MarchingCubesCutCell()
for (k, tag) in ((0.0, "hard"), (0.1, "smooth"))
    geo_k = SDFUnion(sphere, bar; smoothing=k)
    grid_k = bounding_grid(geo_k; N=200)
    cache_k = update_cache!(allocate_cache(grid_k, mc), geo_k, grid_k)
    mesh_k = generate_mesh(cache_k, grid_k)
    fname = joinpath(outdir, "union_sphere_bar_$tag")
    write_vtk(mesh_k, fname)
    println("wrote $fname.vtu  (smoothing = $k, $(length(mesh_k.elements)) triangles)")
end
println("\nIn ParaView: open both and compare the seams -- creases at smoothing = 0, fillets at 0.1.")