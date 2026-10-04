# =====================================
# The caches: `allocate_cache` / `update_cache!` for each method.
#
# What is asserted, per method:
#
#   * a fresh cache reads as no body -- every cell outside, all fluid, every face open
#   * what the cache stores is what it sampled: a nodal cache's `phi` is the field at its nodes, a
#     PLIC cache's planes the distance and normal at each centroid
#   * a shared face is single-valued (for PLIC, after the face pass resolves it), and exact for PLIC
#     where the boundary is straight
#   * the grid may move between updates but not change size, and an update on a moved grid is
#     bitwise a fresh cache's; PLIC refuses an anisotropic grid
#   * the element type follows the grid, and the device path agrees with the host

using CutCellMethods: MarchingSquaresCutCellCache, MarchingCubesCutCellCache, PLICCutCellCache,
                      CELL_OUTSIDE, CELL_CUT, closure_residual, cell_nodes
using CartesianMeshes: get_elem_centroid
using KernelAbstractions

const CACHE_G2 = CartesianGrid(SVector(-1.3, -1.25), (48, 44), SVector(0.055, 0.055))
const CACHE_G3 = CartesianGrid(SVector(-1.5, -1.5, -1.5), (24, 24, 24), SVector(0.125, 0.125, 0.125))
const CACHE_CIRCLE = SDFCircle(center=SVector(0.013, -0.021), radius=0.83)
const CACHE_SPHERE = SDFSphere(SVector(0.01, 0.02, -0.03), 1.0)

"""How many interior faces of a field of per-cell `SVector{2D}` face values disagree between the two
cells that share them -- cell `ci`'s slot `2c` against cell `ci + e_c`'s slot `2c-1`."""
function shared_face_mismatches(ff::AbstractArray{<:Any,D}) where {D}
    bad = 0
    for c in 1:D
        e = CartesianIndex(ntuple(a -> a == c ? 1 : 0, D))
        for ci in CartesianIndices(ntuple(a -> a == c ? size(ff, a) - 1 : size(ff, a), D))
            ff[ci][2c] === ff[ci + e][2c - 1] || (bad += 1)
        end
    end
    return bad
end

"""Whether a nodal cache's `phi` is `geo` sampled at every node of `g` by `get_node` -- the
expression `cell_nodes` places a corner with."""
sampled_at_nodes(cache, geo, g) =
    all(ni -> cache.phi[ni] === eltype(cache.phi)(SDFLibrary.sdf_value(geo, get_node(g, ni))),
        CartesianIndices(cache.phi))

"""Whether two caches over the same grid hold bitwise the same cells."""
same_cells(a, b) = all(ci -> a.cells[ci] === b.cells[ci], CartesianIndices(a.cells))

"""The exact open fraction of the edge `p0 -> p1` against the straight boundary `f(x) = 0`, fluid
where `f >= 0`: `f` is linear along the edge, so this is its root, with nothing reconstructed."""
function edge_open(f, p0, p1)
    a, b = f(p0), f(p1)
    a >= 0 && b >= 0 && return 1.0
    a < 0 && b < 0 && return 0.0
    return a >= 0 ? a / (a - b) : b / (b - a)
end

"""Cell `ci`'s four exact face fractions against the straight boundary `f(x) = 0`, by direction."""
function exact_faces(f, g, ci)
    lo, hi = get_node(g, ci), get_node(g, ci + CartesianIndex(1, 1))
    return SVector(edge_open(f, lo, SVector(lo[1], hi[2])), edge_open(f, SVector(hi[1], lo[2]), hi),
                   edge_open(f, lo, SVector(hi[1], lo[2])), edge_open(f, SVector(lo[1], hi[2]), hi))
end

@testset verbose = true "caches" begin

    @testset "marching squares" begin
        MS = MarchingSquaresCutCell()
        g = CACHE_G2
        idx = CartesianIndices(Tuple(g.n))
        cache = allocate_cache(g, MS)
        @test cache isa MarchingSquaresCutCellCache
        @test size(cache.cells) == Tuple(g.n)
        @test size(cache.phi) == Tuple(g.n) .+ 1

        # Seeded as no body, not as whatever memory the allocation returned.
        @test all(==(CELL_OUTSIDE), cache.cells.kind)
        @test all(isone, cache.cells.volume_fraction)
        @test all(f -> all(isone, f), cache.cells.face_fraction)

        @test update_cache!(cache, CACHE_CIRCLE, g) === cache
        @test count(==(CELL_CUT), cache.cells.kind) > 100

        # `phi` is sampled at `get_node`, the expression `cell_nodes` places a corner with.
        @test sampled_at_nodes(cache, CACHE_CIRCLE, g)

        @test shared_face_mismatches(cache.cells.face_fraction) == 0
        @test all(ci -> closure_residual(cache.cells[ci], g.d) == zero(SVector{2,Float64}), idx)

        # The contour of the very same reconstruction, straight off the stored samples: one segment
        # per cut cell, each vertex a crossing of the circle.
        s = generate_mesh(cache, g)
        @test length(s.elements) == count(==(CELL_CUT), cache.cells.kind)
        @test all(x -> abs(norm(x - CACHE_CIRCLE.center) - CACHE_CIRCLE.radius) < 2e-3, s.nodes.coord)
        # The cache knows its `n`, so the wrong grid is caught.
        @test_throws DimensionMismatch generate_mesh(cache, CartesianGrid(g.x0, Tuple(g.n) .+ 1, g.d))

        # A moving body is a moving grid: same `n`, new origin -- and nothing of the old position
        # survives, bitwise.
        moved = CartesianGrid(g.x0 .+ SVector(0.013, -0.007), Tuple(g.n), g.d)
        update_cache!(cache, CACHE_CIRCLE, moved)
        @test sampled_at_nodes(cache, CACHE_CIRCLE, moved)
        @test same_cells(cache, update_cache!(allocate_cache(moved, MS), CACHE_CIRCLE, moved))
        @test_throws DimensionMismatch update_cache!(
            cache, CACHE_CIRCLE, CartesianGrid(SVector(0.0, 0.0), (10, 10), SVector(0.1, 0.1)))

        # A callable level set is a field too.
        update_cache!(cache, x -> x[2] - 0.3, g)
        @test count(==(CELL_CUT), cache.cells.kind) == g.n[1]

        c32 = allocate_cache(CartesianMeshes.adapt_type(Float32, g), MS)
        @test eltype(c32.cells) === CutCellData{2,Float32,4,1}
        @test eltype(c32.phi) === Float32
    end

    @testset "marching squares on a tree" begin
        MS = MarchingSquaresCutCell()
        # Refined in a band round the circle, so the cut leaves come in several sizes.
        base = CartesianGrid(mincorner=(-1.5, -1.5), maxcorner=(1.5, 1.5), n=(8, 8))
        tree = AdaptiveMesh(base; max_level=3, initial_level=1)
        near(c) = abs(first(get_sdf(CACHE_CIRCLE, CartesianMeshes.get_elem_centroid(tree, c)))) <
                  1.5 * maximum(CartesianMeshes.get_elem_size(tree, c))
        for _ in 1:2
            CartesianMeshes.refine_where!(near, tree)
        end
        n = nleaves(tree)
        @test length(unique(leaf_level(tree, i) for i in 1:n)) > 1

        cache = allocate_cache(tree, MS)
        @test cache isa MarchingSquaresCutCellCache
        @test length(cache.cells) == n && length(cache.phi) == n
        @test eltype(cache.phi) === SVector{4,Float64}
        # Seeded as no body.
        @test all(==(CELL_OUTSIDE), cache.cells.kind)
        @test all(isone, cache.cells.volume_fraction)

        @test update_cache!(cache, CACHE_CIRCLE, tree) === cache
        # Each leaf's four corner values, sampled at its lattice corners, in `MS_NODE_BITS` order.
        @test all(i -> cache.phi[i] === map(x -> SDFLibrary.sdf_value(CACHE_CIRCLE, x), cell_nodes(tree, i)), 1:n)
        @test count(==(CELL_CUT), cache.cells.kind) > 20
        @test all(i -> closure_residual(cache.cells[i], CartesianMeshes.get_elem_size(tree, leaf(tree, i))) ==
                       zero(SVector{2,Float64}), 1:n)
        @test CutCellMethods.volume_fractions(cache) === cache.cells.volume_fraction
        @test CutCellMethods.face_fractions(cache) === cache.cells.face_fraction
        @test CutCellMethods.kinds(cache) === cache.cells.kind
        @test CutCellMethods.face_centroids_local(cache) === cache.cells.face_centroid_local

        # The contour off the stored corners: one segment per cut leaf.
        @test !any(cache.cells.ambiguous)
        @test length(generate_mesh(cache, tree).elements) == count(==(CELL_CUT), cache.cells.kind) > 0

        # Another tree has other leaves, which is caught on update and on the contour.
        other = AdaptiveMesh(base; max_level=3, initial_level=2)
        @test_throws DimensionMismatch update_cache!(cache, CACHE_CIRCLE, other)
        @test_throws DimensionMismatch generate_mesh(cache, other)
        # A tree cache is not a grid cache, nor the other way round.
        @test_throws MethodError update_cache!(cache, CACHE_CIRCLE, CACHE_G2)
        @test_throws MethodError update_cache!(allocate_cache(CACHE_G2, MS), CACHE_CIRCLE, tree)

        # The element type follows the tree.
        t32 = AdaptiveMesh(CartesianMeshes.adapt_type(Float32, base); max_level=2, initial_level=1)
        c32 = update_cache!(allocate_cache(t32, MS), CACHE_CIRCLE, t32)
        @test eltype(c32.cells) === CutCellData{2,Float32,4,1}
        @test eltype(c32.phi) === SVector{4,Float32}
    end

    @testset "marching cubes" begin
        MC = MarchingCubesCutCell()
        g = CACHE_G3
        idx = CartesianIndices(Tuple(g.n))
        cache = allocate_cache(g, MC)
        @test cache isa MarchingCubesCutCellCache
        @test all(==(CELL_OUTSIDE), cache.cells.kind)
        @test all(isone, cache.cells.volume_fraction)

        update_cache!(cache, CACHE_SPHERE, g)
        @test count(==(CELL_CUT), cache.cells.kind) > 1000
        @test sampled_at_nodes(cache, CACHE_SPHERE, g)
        @test shared_face_mismatches(cache.cells.face_fraction) == 0
        @test all(ci -> closure_residual(cache.cells[ci], g.d) == zero(SVector{3,Float64}), idx)
        # The cache's own field, marched: the same surface a second, independent cache produces.
        s = generate_mesh(cache, g; warn=false)
        @test !isempty(s.elements)
        other_cache = update_cache!(allocate_cache(g, MC), CACHE_SPHERE, g)
        ref = generate_mesh(other_cache, g; warn=false)
        @test [e.con for e in s.elements] == [e.con for e in ref.elements]
        @test s.nodes.coord == ref.nodes.coord
        # The cache knows its `n`, so the wrong grid is caught.
        @test_throws DimensionMismatch generate_mesh(cache, CartesianGrid(g.x0, Tuple(g.n) .+ 1, g.d))
        # A moved grid rewrites every cell: bitwise a fresh cache's.
        moved = CartesianGrid(g.x0 .+ SVector(0.013, -0.007, 0.021), Tuple(g.n), g.d)
        update_cache!(cache, CACHE_SPHERE, moved)
        @test same_cells(cache, update_cache!(allocate_cache(moved, MC), CACHE_SPHERE, moved))
    end

    @testset "PLIC" begin
        PL = PLICCutCell()
        g = CACHE_G2
        idx = CartesianIndices(Tuple(g.n))
        cache = allocate_cache(g, PL)
        @test cache isa PLICCutCellCache
        # The same per-cell struct every other cache holds; what a plane fit has no value for is NaN.
        @test eltype(cache.cells) === CutCellData{2,Float64,4,1}
        @test all(==(CELL_OUTSIDE), cache.cells.kind)
        @test all(f -> all(isone, f), cache.cells.face_fraction)

        update_cache!(cache, CACHE_CIRCLE, g)
        @test all(c -> all(isnan, c), cache.cells.centroid)
        @test all(c -> all(p -> all(isnan, p), c), cache.cells.face_centroid_local)
        @test all(c -> all(isnan, c), cache.cells.interface_centroid)
        @test !any(cache.cells.ambiguous)
        # The fit pass is one `get_sdf` at each centroid: its normal stored as it came, and the plane
        # through the cell it implies, `n . xi = sum(n)/2 - dist/dx` in unit-cell coordinates, whose
        # clip's complement is the fluid fraction. (The circle has no degenerate centroid here.)
        @test all(cache.is_valid)
        @test all(idx) do ci
            d, nrm = get_sdf(CACHE_CIRCLE, get_elem_centroid(g, ci))
            n = SVector{2,Float64}(nrm)
            a = sum(n) / 2 - d / g.d[1]
            cache.normals[ci] === n && cache.intercepts[ci] ≈ a &&
                isapprox(cache.cells.volume_fraction[ci], 1 - CartesianMeshes.get_volume_fraction(n, a); atol=1e-14)
        end

        # The face pass: every interior face single-valued, every fraction a fraction, and the
        # interface formed from those fractions, so the closure holds exactly.
        ff = cache.cells.face_fraction
        @test shared_face_mismatches(ff) == 0
        @test all(f -> all(x -> 0 <= x <= 1, f), ff)
        @test all(ci -> closure_residual(cache.cells[ci], g.d) == zero(SVector{2,Float64}), idx)

        # Where the boundary is straight, every cell's plane is the boundary itself, so each face's
        # resolved fraction is exactly that line's open fraction of it -- boundary faces included.
        nrm = normalize(SVector(1.0, 0.37))
        line(x) = dot(nrm, x - CACHE_CIRCLE.center)
        update_cache!(cache, SDFPlane(point=CACHE_CIRCLE.center, normal=nrm), g)
        @test count(==(CELL_CUT), cache.cells.kind) > 40
        @test maximum(ci -> maximum(abs, cache.cells.face_fraction[ci] - exact_faces(line, g, ci)), idx) < 1e-13

        aniso = CartesianGrid(SVector(-1.5, -1.5), (30, 15), SVector(0.1, 0.2))
        @test_throws ArgumentError allocate_cache(aniso, PL)

        # Dimension-generic: the 3D cache is the same two passes.
        c3 = allocate_cache(CACHE_G3, PL)
        update_cache!(c3, CACHE_SPHERE, CACHE_G3)
        @test eltype(c3.cells) === CutCellData{3,Float64,6,2}
        @test count(==(CELL_CUT), c3.cells.kind) > 1000
        @test shared_face_mismatches(c3.cells.face_fraction) == 0
        @test all(f -> all(x -> 0 <= x <= 1, f), c3.cells.face_fraction)
        idx3 = CartesianIndices(Tuple(CACHE_G3.n))
        @test all(ci -> closure_residual(c3.cells[ci], CACHE_G3.d) == zero(SVector{3,Float64}), idx3)
    end

    @testset "the uniform readers" begin
        # Every cache answers the same three questions, whatever its layout, and hands back its own
        # storage -- so a write through a reader is a write to the cache.
        ms = update_cache!(allocate_cache(CACHE_G2, MarchingSquaresCutCell()), CACHE_CIRCLE, CACHE_G2)
        mc = update_cache!(allocate_cache(CACHE_G3, MarchingCubesCutCell()), CACHE_SPHERE, CACHE_G3)
        pl = update_cache!(allocate_cache(CACHE_G2, PLICCutCell()), CACHE_CIRCLE, CACHE_G2)
        for c in (ms, mc, pl)
            @test CutCellMethods.face_fractions(c) === c.cells.face_fraction
            @test CutCellMethods.volume_fractions(c) === c.cells.volume_fraction
            @test CutCellMethods.kinds(c) === c.cells.kind
            @test CutCellMethods.face_centroids_local(c) === c.cells.face_centroid_local
        end
        @test all(f -> all(x -> all(isnan, x), f), CutCellMethods.face_centroids_local(pl)) # PLIC has none to report
        # The polyline clipper takes a line mesh, as an `SDFMesh`, rather than a field: a 64-gon of
        # the same circle.
        circ = [CACHE_CIRCLE.center + CACHE_CIRCLE.radius * SVector(cos(2π * k / 64), sin(2π * k / 64))
                for k in 0:63]
        poly = Mesh([Point(p) for p in circ], [Line(Int32(k), Int32(mod1(k + 1, 64))) for k in 1:64])
        pc = update_cache!(allocate_cache(CACHE_G2, PolylineClippingCutCell()), SDFMesh(poly), CACHE_G2)
        @test CutCellMethods.face_fractions(pc) === pc.cells.face_fraction
        @test CutCellMethods.volume_fractions(pc) === pc.cells.volume_fraction
        @test CutCellMethods.kinds(pc) === pc.cells.kind
        @test CutCellMethods.face_centroids_local(pc) === pc.cells.face_centroid_local
        @test count(==(CELL_CUT), pc.cells.kind) > 100
        @test shared_face_mismatches(pc.cells.face_fraction) == 0

        # `is_cell_open` on a cache agrees with the per-cell reading everywhere, whatever the method.
        for c in (ms, mc, pl, pc)
            @test all(ci -> CutCellMethods.is_cell_open(c, ci) == CutCellMethods.is_cell_open(c.cells[ci]),
                      CartesianIndices(size(c.cells)))
            @test count(ci -> !CutCellMethods.is_cell_open(c, ci), CartesianIndices(size(c.cells))) > 0
        end

        ci = findfirst(==(CELL_CUT), ms.cells.kind)
        @test CutCellMethods.is_cell_open(ms, ci)
        CutCellMethods.face_fractions(ms)[ci] = zero(SVector{4,Float64})
        CutCellMethods.kinds(ms)[ci] = CutCellMethods.CELL_INSIDE
        @test ms.cells[ci].face_fraction == zero(SVector{4,Float64})
        @test ms.cells[ci].kind == CutCellMethods.CELL_INSIDE
        @test !CutCellMethods.is_cell_open(ms, ci) # it reads the edited fractions
    end

    if HAS_GPU
        @testset "GPU agrees with the host" begin
            for (method, g, geo) in ((MarchingSquaresCutCell(), CACHE_G2, CACHE_CIRCLE),
                                     (MarchingCubesCutCell(), CACHE_G3, CACHE_SPHERE),
                                     (PLICCutCell(), CACHE_G2, CACHE_CIRCLE),
                                     (PLICCutCell(), CACHE_G3, CACHE_SPHERE))
                g32 = CartesianMeshes.adapt_type(Float32, g)
                geo32 = SDFLibrary.adapt_type(Float32, geo)
                host = update_cache!(allocate_cache(g32, method), geo32, g32)
                dev = update_cache!(allocate_cache(g32, method; backend=CUDABackend()), geo32, g32)
                @test Array(dev.cells.kind) == host.cells.kind
                @test maximum(abs.(Array(dev.cells.volume_fraction) .- host.cells.volume_fraction)) <
                      100 * eps(Float32)
                # Looser than the volume: a face nearly parallel to its plane amplifies the two
                # backends' last-bit difference in the fit. Each backend's own faces are exact.
                @test maximum(f -> maximum(abs, f), Array(dev.cells.face_fraction) .- host.cells.face_fraction) <
                      1e-4
                @test shared_face_mismatches(Array(dev.cells.face_fraction)) == 0
                # The surface marched from a device cache's field, copied to the host.
                if method isa Union{MarchingCubesCutCell,MarchingSquaresCutCell}
                    @test length(generate_mesh(dev, g32; warn=false).elements) ==
                          length(generate_mesh(host, g32; warn=false).elements) > 0
                end
            end
        end
        @testset "a tree cache on the device" begin
            MS = MarchingSquaresCutCell()
            base = CartesianGrid(mincorner=(-1.5, -1.5), maxcorner=(1.5, 1.5), n=(8, 8))
            tree = AdaptiveMesh(base; max_level=2, initial_level=1)
            CartesianMeshes.refine_where!(c -> CartesianMeshes.get_elem_centroid(tree, c)[1] > 0, tree)
            dtree = adapt(CuArray, tree)
            host = update_cache!(allocate_cache(tree, MS), CACHE_CIRCLE, tree)
            dev = update_cache!(allocate_cache(dtree, MS), adapt(CuArray, CACHE_CIRCLE), dtree)
            # The cache follows the tree's backend, and a tree on another one is refused.
            @test KernelAbstractions.get_backend(dev) isa CUDABackend
            @test_throws ArgumentError update_cache!(allocate_cache(tree, MS), CACHE_CIRCLE, dtree)
            @test Array(dev.cells.kind) == host.cells.kind
            @test maximum(abs.(Array(dev.cells.volume_fraction) .- host.cells.volume_fraction)) < 1e-13
            # The report reads a device cache's cells as it reads a host one's.
            rd, rh = CutCellMethods.cut_cell_report(dev.cells), CutCellMethods.cut_cell_report(host.cells)
            @test (rd.n, rd.cut, rd.inside, rd.outside, rd.ambiguous) == (rh.n, rh.cut, rh.inside, rh.outside, rh.ambiguous) &&
                  rd.cut > 0
            # The contour is marched on the host, over the host tree, from the device's corners.
            @test length(generate_mesh(dev, tree).elements) == length(generate_mesh(host, tree).elements) > 0
        end
    end
end
