# =====================================
# Polyline clipping, P2-: the cache
#
# P2 asserts what the host decides and the classification kernels write: the crossings, the node
# and cell states and the cut list, against references that share no code with them -- brute force
# over every grid line, `BigFloat` rays with a numerical perturbation, and exact rational
# segment-box intersection.

using Test
using StaticArrays
using CartesianMeshes
using MeshLibrary
using Random
using KernelAbstractions
using Logging
using CutCellMethods: PolylineClippingCutCellCache, PolylineLattice, PL_FLUID, PL_SOLID, PL_CUT,
                      PL_FLAG_STATUS_CONFLICT, CELL_INSIDE, CELL_OUTSIDE, CELL_CUT, _pl_dims,
                      _pl_xedge_lin, _pl_yedge_lin, _pl_node_lin, _pl_cell_lin, _pl_nxedges,
                      _crossed_lines, _xcross_above, _ycross_right, _cross_offset, _below_on_line,
                      _line_frame, reset_topology!

isdefined(@__MODULE__, :PV) || include("polyline_meshes.jl")
isdefined(@__MODULE__, :ref_xcross_above) || include("test_poly_crossings.jl")

# A dyadic grid, so that shifting it by whole cells moves every line value exactly.
const PCG = CartesianGrid(SVector(-1.0, -1.0), (64, 64), SVector(1 / 32, 1 / 32))
const PCM = PolylineClippingCutCell()

const PC_BODIES = [
    "square on nodes" => [rect_pts((-0.5, -0.25), (0.25, 0.5))],
    "rotated square" => [square_pts((0.013, -0.021), 0.9; θ=0.3)],
    "64-gon" => [ngon_pts((0.01, 0.02), 0.7, 64; θ0=0.05)],
    "airfoil" => [naca4_pts(; m=0.02, p=0.4, t=0.12, n=100, chord=1.4, le=(-0.7, 0.01), α=0.1)],
    "three elements" => [[p + PV(-0.55, 0.0) for p in pts] for pts in three_element_pts()],
    "thin plate" => [plate_pts((0.0, 0.1), 1.3, 1e-3; θ=0.21)],
    "islands" => [square_pts((0.3, 0.3), 1 / 32), square_pts((-0.4016, 0.3016), 0.004),
                  square_pts((-0.3934, 0.3094), 0.003)],
]

# ---- references ----------------------------------------------------------------------------------

# Every crossing by brute force: every element against every grid line by the side rule, the node
# interval by counting the nodes it passes.
function brute_crossings(X, lines, L::PolylineLattice)
    out = Tuple{Int,Int,Int,Int}[]      # (elem, axis, line, interval)
    for (e, c) in enumerate(lines), axis in 1:2
        a, b = X[c[1]], X[c[2]]
        ls = axis == 1 ? L.xs : L.ys
        nodes = axis == 1 ? L.ys : L.xs
        for k in eachindex(ls)
            (a[axis] > ls[k]) != (b[axis] > ls[k]) || continue
            past(v) = axis == 1 ? _xcross_above(a, b, ls[k], v) : _ycross_right(a, b, v, ls[k])
            push!(out, (e, axis, k, count(past, nodes)))
        end
    end
    return sort!(out)
end

# The cache's crossings, decoded from its edge CSR.
function cache_crossings(cache)
    C = cache.work.cross
    b = cache.work.block
    d = _pl_dims(b)
    out = Tuple{Int,Int,Int,Int}[]
    nxe = _pl_nxedges(b)
    for m in 1:(length(C.edge_start) - 1), s in C.edge_start[m]:(C.edge_start[m + 1] - 1)
        if m <= nxe
            il, jl = mod1(m, d[1] + 1), (m - 1) ÷ (d[1] + 1) + 1
            push!(out, (C.elem[s], 1, il + b.lo[1] - 1, jl + b.lo[2] - 1))
        else
            q = m - nxe
            il, jl = mod1(q, d[1]), (q - 1) ÷ d[1] + 1
            push!(out, (C.elem[s], 2, jl + b.lo[2] - 1, il + b.lo[1] - 1))
        end
    end
    return sort!(out)
end

# Whether the perturbed point `(X + ε, Y + ε²)` is in the solid, by `BigFloat` rays: up the x-line
# (`dir = 1`) or along the y-line (`dir = 2`), counting the crossings beyond it.
function ref_node_solid(X, lines, Xn, Yn, dir)
    n = 0
    for c in lines
        a, b = X[c[1]], X[c[2]]
        if dir == 1
            (a[1] > Xn) != (b[1] > Xn) && ref_xcross_above(a, b, Xn, Yn) && (n += 1)
        else
            (a[2] > Yn) != (b[2] > Yn) && ref_ycross_right(a, b, Xn, Yn) && (n += 1)
        end
    end
    return isodd(n)
end

# Whether an element meets a cell box, exactly, in rationals: `open` for its interior, else the
# closed box.
function seg_meets_box(a, b, lo, hi; open::Bool)
    # the segment's box against the cell's, exact in floats: most pairs stop here
    (max(a[1], b[1]) < lo[1] || min(a[1], b[1]) > hi[1] ||
     max(a[2], b[2]) < lo[2] || min(a[2], b[2]) > hi[2]) && return false
    R = Rational{BigInt}
    t0, t1 = R(0), R(1)
    for ax in 1:2
        p, q = R(a[ax]), R(b[ax])
        l, h = R(lo[ax]), R(hi[ax])
        Δ = q - p
        if Δ == 0
            (open ? (l < p < h) : (l <= p <= h)) || return false
        else
            u, v = (l - p) / Δ, (h - p) / Δ
            u > v && ((u, v) = (v, u))
            t0, t1 = max(t0, u), min(t1, v)
        end
    end
    return open ? t0 < t1 : t0 <= t1
end

# Whether a point on no element is in the solid: a ray along +x, in rationals.
function ref_point_solid(X, lines, p)
    R = Rational{BigInt}
    n = 0
    for c in lines
        a, b = X[c[1]], X[c[2]]
        (a[2] > p[2]) != (b[2] > p[2]) || continue
        x = R(a[1]) + (R(p[2]) - R(a[2])) * (R(b[1]) - R(a[1])) / (R(b[2]) - R(a[2]))
        x > R(p[1]) && (n += 1)
    end
    return isodd(n)
end

mesh_X(mesh) = [SVector{2,Float64}(c) for c in mesh.nodes.coord]
mesh_lines(mesh) = [SVector{2,Int32}(e.con) for e in mesh.elements]

# ---- tests ----------------------------------------------------------------------------------------

@testset verbose = true "polyline clipping: cache" begin

    @testset "a fresh cache reads as no body" begin
        cache = allocate_cache(PCG, PCM)
        @test cache isa PolylineClippingCutCellCache
        @test size(cache.cells) == Tuple(PCG.n)
        @test size(cache.edges.ax) == (65, 64) && size(cache.edges.ay) == (64, 65)
        @test all(==(CELL_OUTSIDE), cache.cells.kind)
        @test all(isone, cache.cells.volume_fraction)
        @test all(f -> all(isone, f), cache.cells.face_fraction)
        @test all(==(PL_FLUID), cache.info.status)
        @test all(isone, cache.edges.ax.fraction) && all(isone, cache.edges.ay.fraction)
        @test CutCellMethods.face_fractions(cache) === cache.cells.face_fraction
        @test CutCellMethods.volume_fractions(cache) === cache.cells.volume_fraction
        @test CutCellMethods.kinds(cache) === cache.cells.kind
        @test CutCellMethods.face_centroids_local(cache) === cache.cells.face_centroid_local
    end

    @testset "$name" for (name, loops) in PC_BODIES
        mesh = poly_mesh(loops)
        cache = allocate_cache(PCG, PCM)
        @test update_cache!(cache, SDFMesh(mesh), PCG) === cache
        X, lines = mesh_X(mesh), mesh_lines(mesh)
        L = cache.work.lattice
        C = cache.work.cross
        b = cache.work.block
        d = _pl_dims(b)

        # Crossings: the edge CSR holds exactly the brute-force set, each edge sorted along itself.
        @test cache_crossings(cache) == brute_crossings(X, lines, L)
        nxe = _pl_nxedges(b)
        sorted = true
        for m in 1:(length(C.edge_start) - 1)
            r = C.edge_start[m]:(C.edge_start[m + 1] - 1)
            axis = m <= nxe ? 1 : 2
            fr(s) = _line_frame(X[lines[C.elem[s]][1]], X[lines[C.elem[s]][2]], axis)
            for s in r[1:(end - 1)]
                sorted &= _below_on_line(fr(s)..., fr(s + 1)...) && C.offset[s] <= C.offset[s + 1]
            end
        end
        @test sorted

        # Node states: the walks agree with each other and with both BigFloat rays, everywhere.
        @test C.nconflict == 0
        @test !any(f -> f & PL_FLAG_STATUS_CONFLICT != 0, cache.info.flags)
        nbad = 0
        for jl in 1:(d[2] + 1), il in 1:(d[1] + 1)
            Xn, Yn = L.xs[il + b.lo[1] - 1], L.ys[jl + b.lo[2] - 1]
            s = C.node_solid[_pl_node_lin(d, il, jl)]
            nbad += (s != ref_node_solid(X, lines, Xn, Yn, 1)) + (s != ref_node_solid(X, lines, Xn, Yn, 2))
        end
        @test nbad == 0

        # Cells: cut exactly where an element meets the interior, never where it misses the closed
        # box, and an uncut cell's kind is where its centre is. (The host's classification: the cut
        # pass may then split a cut cell, or find it exactly solid or fluid once slivers are gone.)
        nbad_cut = nbad_uncut = 0
        hstatus(ci) = all(b.lo .<= Tuple(ci) .<= b.hi) ?
            C.status[_pl_cell_lin(d, ci[1] - b.lo[1] + 1, ci[2] - b.lo[2] + 1)] : PL_FLUID
        for ci in CartesianIndices(Tuple(PCG.n))
            lo, hi = get_node(PCG, ci), get_node(PCG, ci + CartesianIndex(1, 1))
            st = hstatus(ci)
            k = cache.cells.kind[ci]
            if st == PL_CUT
                nbad_cut += !any(c -> seg_meets_box(X[c[1]], X[c[2]], lo, hi; open=false), lines)
            else
                nbad_cut += any(c -> seg_meets_box(X[c[1]], X[c[2]], lo, hi; open=true), lines)
                solid = ref_point_solid(X, lines, (lo + hi) / 2)
                nbad_uncut += (st == PL_SOLID) != solid
                # (a fluid cell a neighbour's sliver handed a wall to is cut, with all its area)
                snapped = cache.info.flags[ci] & CutCellMethods.PL_FLAG_SNAPPED != 0
                nbad_uncut += k != (solid ? CELL_INSIDE : CELL_OUTSIDE) && !(snapped && k == CELL_CUT)
                nbad_uncut += cache.cells.volume_fraction[ci] != (solid ? 0 : 1)
            end
        end
        @test nbad_cut == 0
        @test nbad_uncut == 0
        @test length(C.cut_list) == count(==(PL_CUT), C.status)
        @test all(s -> cache.info.slot[C.cut_list[s]] == s, eachindex(C.cut_list))

        # Uncrossed edges are open or closed as their nodes -- unless a sliver beside one was closed
        # onto it, which only happens next to a cut cell. Nothing outside the block is touched.
        nbad_e = 0
        cutcell(ci) = hstatus(ci) == PL_CUT
        for jl in 1:d[2], il in 1:(d[1] + 1)
            m = _pl_xedge_lin(d, il, jl)
            C.edge_start[m + 1] == C.edge_start[m] || continue
            ei = CartesianIndex(il + b.lo[1] - 1, jl + b.lo[2] - 1)
            f = cache.edges.ax.fraction[ei]
            f == (C.node_solid[_pl_node_lin(d, il, jl)] ? 0 : 1) && continue
            nbad_e += !(f == 0 && (cutcell(ei) || cutcell(ei - CartesianIndex(1, 0))))
        end
        for jl in 1:(d[2] + 1), il in 1:d[1]
            m = _pl_yedge_lin(d, il, jl)
            C.edge_start[m + 1] == C.edge_start[m] || continue
            ei = CartesianIndex(il + b.lo[1] - 1, jl + b.lo[2] - 1)
            f = cache.edges.ay.fraction[ei]
            f == (C.node_solid[_pl_node_lin(d, il, jl)] ? 0 : 1) && continue
            nbad_e += !(f == 0 && (cutcell(ei) || cutcell(ei - CartesianIndex(0, 1))))
        end
        @test nbad_e == 0
        outside = [ci for ci in CartesianIndices(Tuple(PCG.n)) if !all(b.lo .<= Tuple(ci) .<= b.hi)]
        @test all(ci -> cache.info.status[ci] == PL_FLUID && cache.cells.kind[ci] == CELL_OUTSIDE, outside)
    end

    @testset "a whole-cell shift moves every state exactly" begin
        mesh = poly_mesh(PC_BODIES[4].second)
        c1 = update_cache!(allocate_cache(PCG, PCM), SDFMesh(mesh), PCG)
        sh = (3, -2)
        g2 = CartesianGrid(PCG.x0 - SVector(sh) .* PCG.d, Tuple(PCG.n), PCG.d)
        c2 = update_cache!(allocate_cache(g2, PCM), SDFMesh(mesh), g2)
        o = CartesianIndex(sh)
        idx = [ci for ci in CartesianIndices(Tuple(PCG.n)) if checkbounds(Bool, c2.info.status, ci + o)]
        @test all(ci -> c2.info.status[ci + o] == c1.info.status[ci], idx)
        @test all(ci -> c2.cells.kind[ci + o] == c1.cells.kind[ci], idx)
        @test count(==(PL_CUT), c1.info.status) == count(==(PL_CUT), c2.info.status)
    end

    @testset "moving, and the checks" begin
        airfoil = poly_mesh(PC_BODIES[4].second)
        cache = update_cache!(allocate_cache(PCG, PCM), SDFMesh(airfoil), PCG)
        ncut = count(==(PL_CUT), cache.info.status)
        # The body moves: whatever it covered before and does not now reads as fluid again.
        moved = poly_mesh([[p + PV(0.1, 0.35) for p in PC_BODIES[4].second[1]]])
        update_cache!(cache, SDFMesh(moved), PCG)
        ref = update_cache!(allocate_cache(PCG, PCM), SDFMesh(moved), PCG)
        @test cache.info.status == ref.info.status
        @test cache.cells.kind == ref.cells.kind
        @test cache.edges.ax.fraction == ref.edges.ax.fraction && cache.edges.ay.fraction == ref.edges.ay.fraction
        # The grid moves by a fraction of a cell: every cell is rewritten for the new origin.
        g2 = CartesianGrid(PCG.x0 + SVector(0.0071, -0.0043), Tuple(PCG.n), PCG.d)
        update_cache!(cache, SDFMesh(moved), g2)
        ref2 = update_cache!(allocate_cache(g2, PCM), SDFMesh(moved), g2)
        @test cache.info.status == ref2.info.status
        @test cache.cells.centroid == ref2.cells.centroid
        # (No empty-mesh case: an `SDFMesh` cannot be built from an empty mesh.)
        update_cache!(cache, SDFMesh(airfoil), PCG)
        @test count(==(PL_CUT), cache.info.status) == ncut

        @test_throws DimensionMismatch update_cache!(
            cache, SDFMesh(airfoil), CartesianGrid(SVector(-1.0, -1.0), (32, 32), SVector(1 / 16, 1 / 16)))
        @test_throws ArgumentError update_cache!(
            cache, SDFMesh(airfoil), CartesianGrid(SVector(-1.0, -1.0), (64, 64), SVector(1 / 32, 1 / 31)))
        @test_throws ArgumentError allocate_cache(CartesianGrid(SVector(0.0, 0.0, 0.0), (4, 4, 4),
                                                                SVector(0.25, 0.25, 0.25)), PCM)
        # The body must stay a cell clear of the grid's edge.
        @test_throws ArgumentError update_cache!(cache, SDFMesh(poly_mesh(square_pts((0.0, 0.0), 1.95))), PCG)
        # A mesh that fails validation is rejected, and not remembered as valid.
        bad = poly_mesh(reverse(square_pts((0.0, 0.0), 0.5)))
        @test_throws ArgumentError update_cache!(cache, SDFMesh(bad), PCG)
        @test_throws ArgumentError update_cache!(cache, SDFMesh(bad), PCG)
        update_cache!(cache, SDFMesh(airfoil), PCG)
        @test count(==(PL_CUT), cache.info.status) == ncut
        # Rigid motion keeps the topology; `validate = :always` rechecks the geometry anyway.
        topo = cache.work.topo
        update_cache!(cache, SDFMesh(moved), PCG)
        @test cache.work.topo === topo
        always = allocate_cache(PCG, PolylineClippingCutCell(validate=:always))
        update_cache!(always, SDFMesh(airfoil), PCG)
        crossed = poly_mesh([square_pts((0.0, 0.0), 0.5), square_pts((0.3, 0.1), 0.5)])
        # (same connectivity as two separate squares, so only the geometric check can catch it)
        apart = poly_mesh([square_pts((-0.4, 0.0), 0.5), square_pts((0.4, 0.0), 0.5)])
        update_cache!(always, SDFMesh(apart), PCG)
        @test_throws ArgumentError update_cache!(always, SDFMesh(crossed), PCG)
        update_cache!(cache, SDFMesh(apart), PCG)
        # :topology trusts the cached loops (the cut of an invalid body is then garbage, but it is
        # not a validation error) ...
        @test try
            with_logger(NullLogger()) do     # it warns that it could not walk the crossed cells
                update_cache!(cache, SDFMesh(crossed), PCG)
            end
            true
        catch e
            !(e isa ArgumentError && occursin("loops 1 and 2", e.msg))
        end
        reset_topology!(cache)
        @test_throws ArgumentError update_cache!(cache, SDFMesh(crossed), PCG)   # ... until told not to
    end

    @testset "element type" begin
        # A Float32 grid on the same (dyadic) lattice makes the same decisions.
        g32 = CartesianGrid(SVector(-1.0f0, -1.0f0), (64, 64), SVector(1.0f0 / 32, 1.0f0 / 32))
        mesh = poly_mesh(PC_BODIES[5].second)
        c32 = update_cache!(allocate_cache(g32, PCM), SDFMesh(mesh), g32)
        c64 = update_cache!(allocate_cache(PCG, PCM), SDFMesh(mesh), PCG)
        @test eltype(c32.cells) === CutCellData{2,Float32,4,1}
        @test c32.info.status == c64.info.status
        @test c32.cells.kind == c64.cells.kind
    end

    if isdefined(Main, :HAS_GPU) && Main.HAS_GPU
        @testset "GPU agrees with the host" begin
            g32 = CartesianGrid(SVector(-1.0f0, -1.0f0), (64, 64), SVector(1.0f0 / 32, 1.0f0 / 32))
            mesh = poly_mesh(PC_BODIES[5].second)
            host = update_cache!(allocate_cache(g32, PCM), SDFMesh(mesh), g32)
            dev = update_cache!(allocate_cache(g32, PCM; backend=Main.CUDABackend()), SDFMesh(mesh), g32)
            @test Array(dev.info.status) == host.info.status
            @test Array(dev.cells.kind) == host.cells.kind
            @test Array(dev.edges.ax.fraction) == host.edges.ax.fraction
        end
    end
end
