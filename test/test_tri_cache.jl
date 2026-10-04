# =====================================
# Tri clipping, phase 3: allocating the cache, binning, and classifying cells along grid rows.
#
# What is asserted:
#
#   * a fresh cache reads as no body, and an update against an empty mesh returns it there
#   * the bins an update leaves in `cache.work` are exactly the documented test, conservative
#     against a real clip, and deterministic
#   * every cell centre is classified as the analytic inside test says: an axis-aligned box whose
#     faces lie on cell-centre planes and on cell-face planes, rows through mesh vertices and
#     edges, a rotated box, the prism hull, the L-block, a sphere, and the GPPH hull against its
#     generalized winding number
#   * a grid shifted by whole cells shifts the classification exactly; a moved body leaves nothing
#     behind; the cell size and the grid's `n` are fixed; a body touching the grid edge is refused
#   * the uniform readers, Float32, and the GPU path, including a kernel handed the cache itself

using KernelAbstractions
using CutCellMethods: TriClippingCutCellCache, TriClippingCutCellView, CELL_INSIDE, CELL_OUTSIDE,
                      TriScratch, pg_tri_box!, pg_area_moment, reset_topology!

isdefined(@__MODULE__, :tm_box) || include("tri_meshes.jl")
isdefined(@__MODULE__, :shared_face_mismatches) || function shared_face_mismatches(ff::AbstractArray{<:Any,D}) where {D}
    bad = 0
    for c in 1:D
        e = CartesianIndex(ntuple(a -> a == c ? 1 : 0, D))
        for ci in CartesianIndices(ntuple(a -> a == c ? size(ff, a) - 1 : size(ff, a), D))
            ff[ci][2c] === ff[ci + e][2c - 1] || (bad += 1)
        end
    end
    return bad
end

const TCC = TriClippingCutCell()

tcc_centre(g, ci) = get_node(g, ci) + 0.5 * g.d

# A consumer's struct holding a cache, adapted field by field, as a solver's geometry cache is.
struct TccHolder{C}
    cut_cells::C
end
Adapt.@adapt_structure TccHolder
tcc_cut_cells(c) = c
tcc_cut_cells(h::TccHolder) = h.cut_cells

# The uniform readers from inside a kernel, handed the cache (or its holder) as an argument.
@kernel function tcc_read_kernel!(vf, ff, kind, c)
    ci = @index(Global, Cartesian)
    cc = tcc_cut_cells(c)
    @inbounds vf[ci] = CutCellMethods.volume_fractions(cc)[ci]
    @inbounds ff[ci] = CutCellMethods.face_fractions(cc)[ci]
    @inbounds kind[ci] = CutCellMethods.kinds(cc)[ci]
end

# The uncut cells whose kind differs from `inside(centre)`, skipping centres `skip(centre)` marks as
# too close to the surface to call. A cut cell's kind is `CELL_CUT` by definition; what the parity
# sweep decided for it was overwritten by the cut pass.
function tcc_misclassified(cache, g, inside; skip=x -> false)
    bad = CartesianIndex{3}[]
    for ci in CartesianIndices(Tuple(g.n))
        x = tcc_centre(g, ci)
        (skip(x) || cache.cells.kind[ci] == CutCellMethods.CELL_CUT) && continue
        want = inside(x) ? CELL_INSIDE : CELL_OUTSIDE
        cache.cells.kind[ci] == want || push!(bad, ci)
    end
    return bad
end

# A convex polyhedron's inside test and distance to its surface, from its triangles' planes.
function tcc_convex(mesh)
    X = tm_coords(mesh)
    planes = map(tm_tris(mesh)) do c
        n = normalize(cross(X[c[2]] - X[c[1]], X[c[3]] - X[c[1]]))
        (n, dot(n, X[c[1]]))
    end
    sd(x) = maximum(p -> dot(p[1], x) - p[2], planes)
    return (x -> sd(x) < 0), (x -> abs(sd(x)) < 1e-9)
end

# The generalized winding number of a closed triangle mesh at `x`: 1 inside, 0 outside, by the
# Van Oosterom–Strackee solid angle of every triangle.
function tcc_winding(X, tris, x)
    w = 0.0
    for c in tris
        a, b, d = X[c[1]] - x, X[c[2]] - x, X[c[3]] - x
        la, lb, ld = norm(a), norm(b), norm(d)
        num = dot(a, cross(b, d))
        den = la * lb * ld + dot(a, b) * ld + dot(b, d) * la + dot(d, a) * lb
        w += 2 * atan(num, den)
    end
    return w / 4π
end

@testset verbose = true "tri clipping: cache and classification" begin
    # Dyadic cell size and origin, so cell-centre and cell-face planes are exact numbers.
    g = CartesianGrid(SVector(-1.0, -1.0, -1.0), (16, 16, 16), SVector(0.125, 0.125, 0.125))
    idx = CartesianIndices(Tuple(g.n))

    @testset "a fresh cache is no body" begin
        cache = allocate_cache(g, TCC)
        @test cache isa TriClippingCutCellCache
        @test size(cache.cells) == Tuple(g.n) && size(cache.info) == Tuple(g.n)
        @test all(==(CELL_OUTSIDE), cache.cells.kind)
        @test all(isone, cache.cells.volume_fraction)
        @test all(f -> all(isone, f), cache.cells.face_fraction)
        @test shared_face_mismatches(cache.cells.face_centroid_local) == 0
        @test all(ci -> cache.cells.centroid[ci] == tcc_centre(g, ci), idx)
        # The uniform readers hand back the cache's own storage.
        @test CutCellMethods.face_fractions(cache) === cache.cells.face_fraction
        @test CutCellMethods.volume_fractions(cache) === cache.cells.volume_fraction
        @test CutCellMethods.kinds(cache) === cache.cells.kind
        @test CutCellMethods.face_centroids_local(cache) === cache.cells.face_centroid_local
        # Adapting gives the results alone, which read the same way.
        view = Adapt.adapt(Array, cache)
        @test view isa TriClippingCutCellView
        @test CutCellMethods.face_fractions(view) == cache.cells.face_fraction
        @test CutCellMethods.volume_fractions(view) == cache.cells.volume_fraction
        @test CutCellMethods.kinds(view) == cache.cells.kind
        @test CutCellMethods.face_centroids_local(view) == cache.cells.face_centroid_local
        @test_throws ArgumentError allocate_cache(CartesianGrid(SVector(0.0, 0.0), (4, 4), SVector(0.1, 0.1)), TCC)
    end

    @testset "an axis-aligned box: faces on centre planes and on face planes" begin
        # Centre planes are at -1 + (i - 1/2)/8; face planes at -1 + i/8. The box's own vertices
        # (n = 4 per face) land on centre planes too, so rows pass through its vertex fans.
        for (lo, hi) in ((SVector(-0.3125, -0.4375, -0.1875), SVector(0.4375, 0.3125, 0.5625)),
                         (SVector(-0.375, -0.5, -0.25), SVector(0.5, 0.25, 0.625)))
            box, _, _ = tm_box(lo, hi; n=4)
            cache = update_cache!(allocate_cache(g, TCC), SDFMesh(box), g)
            # The ray is along +x and the line is perturbed to (y + ε, z + ε²), so a centre on a
            # face is inside for x ∈ (lo, hi] and y, z ∈ [lo, hi).
            inside(x) = lo[1] < x[1] <= hi[1] && lo[2] <= x[2] < hi[2] && lo[3] <= x[3] < hi[3]
            @test isempty(tcc_misclassified(cache, g, inside))
            # Nothing but the crease flag: no overflow, no unsupported cell, no split.
            @test all(f -> f & ~CutCellMethods.FLAG_MULTI_PATCH == 0x00, cache.info.flags)
            # (Shared face *fractions* agree only once the cut pass resolves the faces the surface
            # crosses, in P5; the placeholder centroids agree already.)
            @test shared_face_mismatches(cache.cells.face_centroid_local) == 0
        end
    end

    @testset "rows through edges and vertices of a rotated box" begin
        R = SMatrix{3,3,Float64}(cosd(45), sind(45), 0, -sind(45), cosd(45), 0, 0, 0, 1)
        # Rotated 45° about z with its corners on the lattice: rows run along its edges.
        box, _, _ = tm_box(SVector(-0.5, -0.25, -0.375), SVector(0.25, 0.5, 0.375); n=2, R=R)
        cache = update_cache!(allocate_cache(g, TCC), SDFMesh(box), g)
        inside, near = tcc_convex(box)
        @test isempty(tcc_misclassified(cache, g, inside; skip=near))
        # And tilted about two axes, off the lattice.
        R2 = SMatrix{3,3,Float64}(cosd(30), sind(30), 0, -sind(30), cosd(30), 0, 0, 0, 1) *
             SMatrix{3,3,Float64}(1, 0, 0, 0, cosd(17), sind(17), 0, -sind(17), cosd(17))
        box2, _, _ = tm_box(SVector(-0.45, -0.3, -0.25), SVector(0.4, 0.35, 0.3); n=3, R=R2,
                            t=SVector(0.013, -0.021, 0.007))
        cache2 = update_cache!(allocate_cache(g, TCC), SDFMesh(box2), g)
        inside2, near2 = tcc_convex(box2)
        @test isempty(tcc_misclassified(cache2, g, inside2; skip=near2))
    end

    @testset "the prism hull, the L-block and a sphere" begin
        hull, _, _ = tm_prism(L=1.4, beam=0.9, deadrise=20.0, depth=0.6, nx=5,
                              t=SVector(-0.7, 0.0, -0.3))
        cache = update_cache!(allocate_cache(g, TCC), SDFMesh(hull), g)
        inside, near = tcc_convex(hull)
        @test isempty(tcc_misclassified(cache, g, inside; skip=near))

        lb, _ = tm_lblock(a=0.4, hz=0.9, t=SVector(-0.41, -0.39, -0.43))
        cache = update_cache!(allocate_cache(g, TCC), SDFMesh(lb), g)
        o = SVector(-0.41, -0.39, -0.43)
        inl(x) = (y = x - o; (0 < y[3] < 0.9) && ((0 < y[1] < 0.8 && 0 < y[2] < 0.4) || (0 < y[1] < 0.4 && 0 < y[2] < 0.8)))
        nearl(x) = (y = x - o; any(abs.(y) .< 1e-9) || any(abs.(y .- 0.4) .< 1e-9) ||
                    any(abs.(y .- 0.8) .< 1e-9) || abs(y[3] - 0.9) < 1e-9)
        @test isempty(tcc_misclassified(cache, g, inl; skip=nearl))

        sph, _, _ = tm_icosphere(r=0.77, c=SVector(0.031, -0.017, 0.022), level=3)
        cache = update_cache!(allocate_cache(g, TCC), SDFMesh(sph), g)
        inside, near = tcc_convex(sph)
        @test isempty(tcc_misclassified(cache, g, inside; skip=near))
        # The solid, (4/3)π 0.77^3, to within the faceting of a level-3 sphere.
        @test sum(1 .- cache.cells.volume_fraction) * prod(g.d) ≈ 4π * 0.77^3 / 3 rtol = 0.01
    end

    @testset "bins: the documented test, conservative, deterministic" begin
        sph, _, _ = tm_icosphere(r=0.6, c=SVector(0.05, -0.03, 0.02), level=2)
        X = tm_coords(sph)
        tris = tm_tris(sph)
        cache = update_cache!(allocate_cache(g, TCC), SDFMesh(sph), g)
        w = cache.work
        block = w.block
        s1, l1 = copy(w.bin_start), copy(w.bin_tri)
        r1, rl1 = copy(w.row_start), copy(w.row_tri)
        # Deterministic: the same body binned again, into the same grow-only buffers, bins the same.
        update_cache!(cache, SDFMesh(sph), g)
        @test w.bin_start == s1 && w.bin_tri == l1
        @test w.row_start == r1 && w.row_tri == rl1
        dims = block.hi - block.lo .+ 1
        δ = SVector{3,Float64}(cache.tols.margin)
        # The documented test: the triangle's box grown by `δ` overlaps the cell, and its plane
        # passes within the cell's half-extent grown by `δ` of the cell centre, measured as the
        # support along the normal -- with the same slack the binning takes, towards keeping it.
        reaches(n, p, c, r) = abs(dot(n, c - p)) <= sum(abs.(n) .* r) * (1 + 1e-12) + 1e-12 * maximum(r)
        scr = TriScratch(KernelAbstractions.CPU(), Float64, 1)
        mismatch = 0
        missed = 0
        for (l, ci) in enumerate(CartesianIndices(ntuple(a -> block.lo[a]:block.hi[a], 3)))
            bin = l1[s1[l]:(s1[l + 1] - 1)]
            @test issorted(bin)
            c = tcc_centre(g, ci)
            lo = c - g.d / 2 - δ
            hi = c + g.d / 2 + δ
            for (t, cc) in enumerate(tris)
                p1, p2, p3 = X[cc[1]], X[cc[2]], X[cc[3]]
                n = normalize(cross(p2 - p1, p3 - p1))
                boxhit = all(min.(p1, p2, p3) .<= hi) && all(max.(p1, p2, p3) .>= lo)
                want = boxhit && reaches(n, p1, c, g.d / 2 + δ)
                mismatch += want != (t in bin)
                # Conservative: any triangle with area in the grown cell is binned there.
                b, m = pg_tri_box!(scr, 1, p1, p2, p3, lo, hi, 1e-14)
                A, _ = pg_area_moment(scr, 1, b, m, n)
                missed += (A > 0) && !(t in bin)
            end
        end
        @test mismatch == 0
        @test missed == 0
        @test length(r1) == dims[2] * dims[3] + 1
    end

    @testset "a moving body: grid shifts, moved nodes" begin
        sph, _, _ = tm_icosphere(r=0.55, c=SVector(0.0123, -0.0217, 0.0311), level=3)
        cache = update_cache!(allocate_cache(g, TCC), SDFMesh(sph), g)
        kind0 = copy(cache.cells.kind)
        # The grid shifted by whole cells: cell i of the new grid is cell i + s of the old.
        s = (1, 2, -1)
        gs = CartesianGrid(g.x0 .+ g.d .* SVector(s), Tuple(g.n), g.d)
        update_cache!(cache, SDFMesh(sph), gs)
        shifted = [cache.cells.kind[ci] == kind0[ci + CartesianIndex(s)]
                   for ci in CartesianIndices(ntuple(a -> max(1, 1 - s[a]):min(g.n[a], g.n[a] - s[a]), 3))]
        @test all(shifted)
        # Every uncut cell's centroid is in the moved grid, including those the body never reached.
        @test all(ci -> cache.cells.kind[ci] == CutCellMethods.CELL_CUT ||
                        cache.cells.centroid[ci] == tcc_centre(gs, ci), idx)
        # A moved body leaves nothing behind: bitwise a fresh cache that only saw the new position.
        moved = tm_retri(sph, tm_tris(sph))
        moved.nodes.coord .+= Ref(SVector(0.21, -0.13, 0.08))
        update_cache!(cache, SDFMesh(moved), g)
        fresh = update_cache!(allocate_cache(g, TCC), SDFMesh(moved), g)
        @test cache.cells == fresh.cells && cache.info == fresh.info
        # (No empty-mesh case: an `SDFMesh` cannot be built from an empty mesh.)
        # The topology is kept across moves and rebuilt when forgotten.
        update_cache!(cache, SDFMesh(moved), g)
        topo = cache.work.topo
        update_cache!(cache, SDFMesh(sph), g)
        @test cache.work.topo === topo
        reset_topology!(cache)
        update_cache!(cache, SDFMesh(sph), g)
        @test cache.work.topo !== topo
    end

    @testset "fixed n, fixed cell size, a body inside the grid" begin
        sph, _, _ = tm_icosphere(r=0.5, level=2)
        cache = allocate_cache(g, TCC)
        @test_throws DimensionMismatch update_cache!(cache, SDFMesh(sph), CartesianGrid(g.x0, (8, 8, 8), g.d))
        @test_throws ArgumentError update_cache!(cache, SDFMesh(sph), CartesianGrid(g.x0, Tuple(g.n), 2 * g.d))
        big, _, _ = tm_icosphere(r=0.99, level=2)
        err = try
            update_cache!(cache, SDFMesh(big), g)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("edge of the grid", err.msg)
    end

    @testset "the GPPH hull against its winding number" begin
        h = tm_gpph()
        if h === nothing
            @info "gpph_clean.inp not generated (examples/geometry/run_generate_mesh.jl): skipped"
        else
            # A coarse grid: the reference sums a solid angle over every triangle for every cell.
            gh = CartesianGrid(SVector(-0.4, -1.5, -0.3), (44, 15, 10), SVector(0.2, 0.2, 0.2))
            c = @test_logs match_mode = :any update_cache!(allocate_cache(gh, TCC), SDFMesh(h), gh)
            X = tm_coords(h)
            tris = tm_tris(h)
            bad = 0
            ambiguous = 0
            for ci in CartesianIndices(Tuple(gh.n))
                c.cells.kind[ci] == CutCellMethods.CELL_CUT && continue
                w = tcc_winding(X, tris, tcc_centre(gh, ci))
                if abs(w - round(w)) > 1e-6
                    ambiguous += 1
                    continue
                end
                bad += (round(w) == 1) != (c.cells.kind[ci] == CELL_INSIDE)
            end
            @test bad == 0
            @test ambiguous == 0
            @test count(==(CELL_INSIDE), c.cells.kind) > 500
        end
    end

    @testset "Float32, and the GPU agrees with the host" begin
        sph, _, _ = tm_icosphere(r=0.6, c=SVector(0.0123, -0.0217, 0.0311), level=3)
        g32 = CartesianMeshes.adapt_type(Float32, g)
        c32 = update_cache!(allocate_cache(g32, TCC), SDFMesh(sph), g32)
        @test eltype(c32.cells) === CutCellData{3,Float32,6,2}
        c64 = update_cache!(allocate_cache(g, TCC), SDFMesh(sph), g)
        # Away from the surface, where single precision cannot move a centre across it.
        X = tm_coords(sph)
        planes = map(c -> (n = normalize(cross(X[c[2]] - X[c[1]], X[c[3]] - X[c[1]])); (n, dot(n, X[c[1]]))),
                     tm_tris(sph))
        clear(ci) = abs(maximum(p -> dot(p[1], tcc_centre(g, ci)) - p[2], planes)) > 1e-5
        @test count(ci -> clear(ci) && c32.cells.kind[ci] != c64.cells.kind[ci], idx) == 0
        if HAS_GPU
            dev = update_cache!(allocate_cache(g32, TCC; backend=CUDABackend()), SDFMesh(sph), g32)
            dk = Array(dev.cells.kind)
            @test count(ci -> clear(ci) && dk[ci] != c32.cells.kind[ci], idx) == 0
            @test maximum(abs, Array(dev.cells.volume_fraction) .- c32.cells.volume_fraction) <= 1
            # Weighted by the fluid fraction: a sliver's centroid is M/V with a tiny V, and in single
            # precision the two backends' rounding moves it the most where it carries least.
            @test maximum(norm.(Array(dev.cells.centroid) .- c32.cells.centroid) .* c32.cells.volume_fraction) < 1e-5
            # (Shared faces agree bitwise across all cells once P5 resolves the cut ones.)
            dk_uncut = [k != CutCellMethods.CELL_CUT for k in dk]
            @test Array(dev.cells.face_centroid_local)[dk_uncut] == c32.cells.face_centroid_local[dk_uncut]
            # A kernel handed the cache, alone or inside a consumer's struct: the launch adapts it to
            # its view, which is isbits on the device.
            @test isbitstype(typeof(CUDA.cudaconvert(dev)))
            be = CUDABackend()
            for arg in (dev, TccHolder(dev))
                vf = KernelAbstractions.zeros(be, Float32, size(dev.cells))
                ff = KernelAbstractions.allocate(be, SVector{6,Float32}, size(dev.cells))
                kind = KernelAbstractions.zeros(be, Int8, size(dev.cells))
                tcc_read_kernel!(be, 64)(vf, ff, kind, arg; ndrange=size(dev.cells))
                KernelAbstractions.synchronize(be)
                @test Array(vf) == Array(dev.cells.volume_fraction)
                @test Array(ff) == Array(dev.cells.face_fraction)
                @test Array(kind) == dk
            end
        end
    end
end
