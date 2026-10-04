# =====================================
# Polyline clipping, P4: determinism, performance, and what the cut looks like
#
#   * an update on a warm cache allocates only kernel-launch overhead, nothing that scales with the
#     grid or the mesh -- the GPU's requirement -- and it is inferred
#   * the output does not depend on the thread count, bitwise, and the GPU agrees with the host
#   * the walls are the polygon's edges split at the grid lines, and the region loops enclose
#     exactly the regions' areas
#   * `write_cache_vtk` writes its four files

using Test
using StaticArrays
using CartesianMeshes
using MeshLibrary
using Adapt
using CutCellMethods: PL_INVALID, PL_SPLIT, PL_FLAG_SNAPPED, region, region_loops, interface_mesh,
                      write_cache_vtk, PolylineClippingCutCellView
using KernelAbstractions

isdefined(@__MODULE__, :PV) || include("polyline_meshes.jl")

const PO_G = CartesianGrid(SVector(-1.0, -1.0), (64, 64), SVector(1 / 32, 1 / 32))
const PO_M = PolylineClippingCutCell()
po_cut(loops; g=PO_G) = update_cache!(allocate_cache(g, PO_M), SDFMesh(poly_mesh(loops)), g)

function po_second_update_bytes(n)
    g = CartesianGrid(SVector(-1.0, -1.0), (n, n), SVector(2 / n, 2 / n))
    # Built once, outside the measurement: the update's own allocations are what is measured.
    mesh = SDFMesh(poly_mesh([naca4_pts(; m=0.02, p=0.4, t=0.12, n=n, chord=1.2, le=(-0.6, 0.0))]))
    c = allocate_cache(g, PO_M)
    g2 = CartesianGrid(g.x0 .+ 0.37 .* g.d, Tuple(g.n), g.d)
    for _ in 1:3
        update_cache!(c, mesh, g)
        update_cache!(c, mesh, g2)
    end
    return (bytes=(@allocated update_cache!(c, mesh, g)), elements=length(mesh.elements))
end

# A consumer's kernel reading the cache through the uniform readers, as HYBIS's geometry pass does:
# handed the cache itself, or inside the consumer's own struct.
@kernel function po_read_kernel!(out, @Const(cc))
    ci = @index(Global, Cartesian)
    @inbounds out[ci] = CutCellMethods.volume_fractions(cc)[ci] +
                        2 * CutCellMethods.face_fractions(cc)[ci][4] + CutCellMethods.kinds(cc)[ci]
end
struct POConsumer{C}
    cut_cells::C
end
Adapt.@adapt_structure POConsumer
@kernel function po_consumer_kernel!(out, @Const(s))
    ci = @index(Global, Cartesian)
    @inbounds out[ci] = CutCellMethods.volume_fractions(s.cut_cells)[ci]
end

po_shoelace(p) =sum(i -> (p[i][1] * p[mod1(i + 1, length(p))][2] - p[i][2] * p[mod1(i + 1, length(p))][1]) / 2,
                     eachindex(p))

@testset verbose = true "polyline clipping: determinism and output" begin

    @testset "a second update allocates launch overhead and a copy of the mesh" begin
        # The launch overhead grows a little on small grids and then levels off (it is the same at
        # 1024^2 and 2048^2); past that, 16x the cells change nothing. The elements do: every update
        # copies the `SDFMesh` to a host `Mesh` and rebuilds its element sets from the labels
        # (`generate_mesh(geo)`), measured at ~20 B an element, so each is allowed 32 B on top.
        small, large = po_second_update_bytes(256), po_second_update_bytes(1024)
        @test large.bytes < 256 * 1024
        @test large.bytes < 1.15 * small.bytes + 32 * (large.elements - small.elements)
    end

    @testset "the update is inferred" begin
        mesh = poly_mesh([naca4_pts(; m=0.02, p=0.4, t=0.12, n=100, chord=1.4, le=(-0.7, 0.01), α=0.1)])
        c = allocate_cache(PO_G, PO_M)
        @test (@inferred update_cache!(c, SDFMesh(mesh), PO_G)) === c
    end

    @testset "adapting gives the results alone, which read the same way" begin
        c = po_cut([naca4_pts(; m=0.02, p=0.4, t=0.12, n=100, chord=1.4, le=(-0.7, 0.01), α=0.1)])
        view = Adapt.adapt(Array, c)
        @test view isa PolylineClippingCutCellView
        @test CutCellMethods.face_fractions(view) == c.cells.face_fraction
        @test CutCellMethods.volume_fractions(view) == c.cells.volume_fraction
        @test CutCellMethods.kinds(view) == c.cells.kind
        @test CutCellMethods.face_centroids_local(view) == c.cells.face_centroid_local
        @test view.info.status == c.info.status
        @test KernelAbstractions.get_backend(view) isa KernelAbstractions.CPU
        # and on the CPU a kernel takes the cache as it is
        out = zeros(Float64, size(c.cells))
        po_read_kernel!(KernelAbstractions.CPU(), 64)(out, c; ndrange=size(out))
        @test out == c.cells.volume_fraction .+ 2 .* getindex.(c.cells.face_fraction, 4) .+ c.cells.kind
    end

    @testset "the output does not depend on the thread count" begin
        # Three subprocesses; CUTCELL_SKIP_THREAD_TEST=1 skips them.
        if get(ENV, "CUTCELL_SKIP_THREAD_TEST", "0") != "1"
            script = joinpath(@__DIR__, "poly_thread_invariance.jl")
            proj = dirname(Base.active_project())
            digests = map((1, 4, 16)) do t
                out = read(`$(Base.julia_cmd()) --project=$proj -t $t $script`, String)
                mt = match(r"DIGEST (\d+) THREADS (\d+)", out)
                mt === nothing ? out : (mt[1], parse(Int, mt[2]))
            end
            @test all(x -> x isa Tuple, digests)
            @test [x[2] for x in digests] == [1, 4, 16]
            @test allequal(first.(digests))
        else
            @info "thread invariance skipped (CUTCELL_SKIP_THREAD_TEST=1)"
        end
    end

    if isdefined(Main, :HAS_GPU) && Main.HAS_GPU
        @testset "the GPU agrees with the host ($T)" for T in (Float32, Float64)
            g = CartesianGrid(SVector{2,T}(-1, -1), (64, 64), SVector{2,T}(1 // 32, 1 // 32))
            for loops in ([[p + PV(-0.55, 0.0) for p in pts] for pts in three_element_pts()],
                          [plate_pts((0.0, 0.1), 1.3, 1e-3; θ=0.21), square_pts((0.3, -0.5), 0.004)],
                          [rect_pts((-0.5, -0.25), (0.25, 0.5))])
                mesh = poly_mesh(loops)
                host = update_cache!(allocate_cache(g, PO_M), SDFMesh(mesh), g)
                dev = update_cache!(allocate_cache(g, PO_M; backend=Main.CUDABackend()), SDFMesh(mesh), g)
                dc = Adapt.adapt(Array, dev.cells)
                di = Adapt.adapt(Array, dev.info)
                @test di.status == host.info.status
                @test di.nregion == host.info.nregion
                @test dc.kind == host.cells.kind
                # the apertures are one expression on host-computed offsets: bitwise
                @test dc.face_fraction == host.cells.face_fraction
                @test Array(dev.arcs.region) == host.arcs.region
                @test Array(dev.arcs.nbr_region) == host.arcs.nbr_region
                @test Array(dev.bsegs.region) == host.bsegs.region
                # areas may differ by the device's fused multiply-adds
                @test maximum(abs.(dc.volume_fraction .- host.cells.volume_fraction)) < 64 * eps(T)
                @test length(dev.regions) == length(host.regions)
                # A device kernel takes the cache, directly or inside a consumer's struct: the launch
                # adapts it to its view.
                be = Main.CUDABackend()
                out = KernelAbstractions.zeros(be, T, size(dev.cells)...)
                po_read_kernel!(be, 64)(out, dev; ndrange=size(out))
                KernelAbstractions.synchronize(be)
                @test Array(out) == dc.volume_fraction .+ 2 .* getindex.(dc.face_fraction, 4) .+ dc.kind
                po_consumer_kernel!(be, 64)(out, POConsumer(dev); ndrange=size(out))
                KernelAbstractions.synchronize(be)
                @test Array(out) == dc.volume_fraction
                @test Adapt.adapt(Array, dev) isa PolylineClippingCutCellView
            end
        end
    end

    @testset "the walls are the polygon's edges" begin
        for loops in ([naca4_pts(; m=0.02, p=0.4, t=0.12, n=100, chord=1.4, le=(-0.7, 0.01), α=0.1)],
                      [[p + PV(-0.55, 0.0) for p in pts] for pts in three_element_pts()],
                      [plate_pts((0.0, 0.1), 1.3, 1e-3; θ=0.21)])
            mesh = poly_mesh(loops)
            c = update_cache!(allocate_cache(PO_G, PO_M), SDFMesh(mesh), PO_G)
            walls, cells, regs = interface_mesh(c, PO_G)
            perim = sum(pts -> sum(i -> sqrt(sum(abs2, pts[mod1(i + 1, length(pts))] - pts[i])), eachindex(pts)), loops)
            @test sum(get_length(walls, e) for e in walls.elements) ≈ perim rtol = 1e-13
            @test length(walls.elemset) == length(loops)
            @test all(>(0), regs)
            # every piece's normal is its element's, into the fluid
            @test all(b -> b.length == 0 ||
                           isapprox(b.normal, get_normal(mesh.nodes.coord[mesh.elements[b.elem].con[1]],
                                                         mesh.nodes.coord[mesh.elements[b.elem].con[2]]);
                                    atol=1e-14), c.bsegs)
            # and each drawn wall's (MeshLibrary's, right of the line) too -- where it is long enough
            # for two points to carry a direction at all
            nb = 0
            for (k, e) in enumerate(walls.elements)
                p = walls.nodes.coord[e.con[1]]; q = walls.nodes.coord[e.con[2]]
                sqrt(sum(abs2, q - p)) > 1e-12 || continue
                n = get_normal(p, q)
                nb += !any(el -> begin
                        a, b = mesh.nodes.coord[el.con[1]], mesh.nodes.coord[el.con[2]]
                        isapprox(get_normal(a, b), n; atol=1e-9) &&
                            abs((b - a)[1] * (p - a)[2] - (b - a)[2] * (p - a)[1]) < 1e-12
                    end, mesh.elements)
            end
            @test nb == 0
            # `generate_mesh` reads the same walls off the cache
            g2 = generate_mesh(c, PO_G)
            @test g2.nodes.coord == walls.nodes.coord
            @test [e.con for e in g2.elements] == [e.con for e in walls.elements]
            @test [(s.name, s.elems) for s in g2.elemset] == [(s.name, s.elems) for s in walls.elemset]
        end
        # The square on nodes: bottom and left walls from its own edges, top and right from the
        # closed slivers -- all four sides, each once.
        c = po_cut([rect_pts((-0.5, -0.25), (0.25, 0.5))])
        walls, cells, regs = interface_mesh(c, PO_G)
        @test sum(get_length(walls, e) for e in walls.elements) ≈ 4 * 0.75 rtol = 1e-15
        snapped = only(s for s in walls.elemset if s.name == "snapped")
        @test sum(get_length(walls, walls.elements[k]) for k in snapped.elems) ≈ 2 * 0.75 rtol = 1e-15
        @test all(ci -> c.info.flags[ci] & PL_FLAG_SNAPPED != 0 || c.info.nregion[ci] >= 1, cells)
    end

    @testset "the region loops enclose the regions" begin
        for loops in ([naca4_pts(; m=0.0, t=0.12, n=100, chord=1.4, le=(-0.7, 0.013))],
                      [plate_pts((0.0, 0.1), 1.3, 1e-3; θ=0.21), plate_pts((0.0, 0.112), 1.3, 1e-3; θ=0.21)],
                      [square_pts((0.0, 0.0), 0.9), square_pts((0.46, 0.01), 0.004),       # beside a wall
                       square_pts((-0.6016, 0.6016), 0.004), square_pts((-0.6084, 0.6094), 0.003)])  # two in one cell
            c = po_cut(loops)
            A = prod(PO_G.d)
            area = Dict{Tuple{CartesianIndex{2},Int},Float64}()
            for lp in region_loops(c, PO_G)
                # (a cell the finalize pass found exactly fluid still has its walk's loop)
                c.info.status[lp.cell] in (CutCellMethods.PL_CUT, PL_SPLIT) || continue
                k = (lp.cell, lp.region)
                area[k] = get(area, k, 0.0) + (lp.hole ? -1 : 1) * abs(po_shoelace(lp.points))
            end
            worst = 0.0
            nreg = 0
            for ci in CartesianIndices(Tuple(PO_G.n)), r in 1:c.info.nregion[ci]
                c.info.status[ci] in (CutCellMethods.PL_CUT, PL_SPLIT) || continue
                haskey(area, (ci, r)) || continue
                nreg += 1
                worst = max(worst, abs(area[(ci, r)] - region(c, ci, r).volume_fraction * A))
            end
            @test nreg == length(area)
            @test nreg == sum(ci -> c.info.status[ci] in (CutCellMethods.PL_CUT, PL_SPLIT) ?
                                    c.info.nregion[ci] : 0, CartesianIndices(Tuple(PO_G.n)))
            @test worst <= 1e-13 * A
        end
    end

    @testset "write_cache_vtk" begin
        c = po_cut([[p + PV(-0.55, 0.0) for p in pts] for pts in three_element_pts()])
        mktempdir() do dir
            files = write_cache_vtk(joinpath(dir, "foil"), c, PO_G)
            @test length(files) == 4
            @test all(f -> isfile(f) && filesize(f) > 0, files)
            @test any(endswith("_regions.vtu"), files) && any(endswith("_walls.vtu"), files)
        end
    end
end
