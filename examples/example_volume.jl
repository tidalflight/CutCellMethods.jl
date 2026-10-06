using SDFLibrary
using CutCellMethods: calc_volume
using CartesianMeshes
using Printf

# Sphere of known volume, deliberately off-centre so its surface doesn't sit flush with any
# grid line -- an exactly grid-aligned body would flatter the cut-cell treatment.
radius = 1.0
sphere = SDFSphere(center=SVector(0.013, -0.021, 0.007), radius=radius)
exact = 4/3 * pi * radius^3

# calc_volume needs an isotropic (cubic-cell) grid; CartesianIsoGrid guarantees that. The
# grid also has to contain the body -- only the part inside its bounds gets counted -- so
# size it off the sphere's own bounding box with some slack.
mincorner, maxcorner = get_bounding_box(sphere)
pad = 0.1 * (maxcorner .- mincorner)
lo, hi = Tuple(mincorner .- pad), Tuple(maxcorner .+ pad)

println("volume of a unit sphere, exact = $exact\n")
@printf("%8s  %10s  %14s  %10s\n", "cells/ax", "dx", "volume", "rel. err")
for dx in (0.2, 0.1, 0.05, 0.025)
    grid = CartesianIsoGrid(; mincorner=lo, maxcorner=hi, sz=dx)
    vol = calc_volume(sphere, grid)
    @printf("%8d  %10.4f  %14.8f  %10.2e\n", size(grid)[1], grid.d[1], vol, abs(vol - exact)/exact)
end
