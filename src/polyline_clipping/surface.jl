# =====================================
# What the cut looks like: the walls, the regions' loops, and the VTK output
#
# All on the host, from what the last update produced. The walls are the boundary segments -- each
# element's piece in each cut cell -- plus any edge a sliver was closed onto; the region loops come
# from walking each cut cell again, on a private copy of its inputs, with a recorder that keeps the
# points the walk passes through (`_PLRecorder`, `walk.jl`). It is the same walk the kernel ran, on
# the same numbers, so the loops drawn are the loops cut.

# The kernels' inputs on the host, with private copies of everything a walk writes, so walking a
# cell again leaves the cache as it was.
function _pl_host_args(cache::PolylineClippingCutCellCache)
    m = Adapt.adapt(Array, _pl_kernel_args(cache))
    return merge(m, (arcs=copy(m.arcs), bsegs=copy(m.bsegs), racc=copy(m.racc), sres=copy(m.sres)))
end

function _pl_check_grid(cache::PolylineClippingCutCellCache{T}, grid::CartesianGrid{2}) where {T}
    g = convert(CartesianGrid{2,T}, grid)
    g.x0 == cache.work.x0 || throw(ArgumentError(
        "`grid` is not the grid the cache was last updated with (its origin differs)"))
    return g
end

"""
    region_loops(cache, grid) -> Vector{NamedTuple}

Every kept fluid region's boundary loop in every cut cell of `cache`'s last update, as
`(cell, region, hole, points)`: `points` in global coordinates, counter-clockwise, the first not
repeated at the end. `region` is the region's number in its cell (walk order). An island inside a
region is its own loop with `hole = true`, counter-clockwise round the island.

A cell the walk could not do (`PL_INVALID`) has no loops. Computed by walking the cells again on
the host, so it is for inspection -- a picture, or a check -- not for a solver's inner loop. `grid`
is the grid of that update.
"""
function region_loops(cache::PolylineClippingCutCellCache{T}, grid::CartesianGrid{2}) where {T}
    w = cache.work
    out = NamedTuple{(:cell, :region, :hole, :points),
                     Tuple{CartesianIndex{2},Int,Bool,Vector{SVector{2,Float64}}}}[]
    w.block.empty && return out
    g = _pl_check_grid(cache, grid)
    m = _pl_host_args(cache)
    blo = CartesianIndex(Tuple(w.block.lo))
    d = Tuple(_pl_dims(w.block))
    drop2 = 2 * cache.tols.drop_area
    R = _PLRecorder{T}()
    for s in eachindex(w.cross.cut_list)
        foreach(empty!, (R.region, R.hole, R.start, R.points))
        _pl_cut_cell!(m, g, blo, d, s, drop2, R)
        m.sres[s].ok || continue
        C = _pl_cell(m, g, blo, d, s)
        rs = Int(m.region_start[s])
        for k in eachindex(R.region)
            rank = Int(m.racc.rank[rs + R.region[k] - 1])
            rank > 0 || continue
            stop = k < length(R.region) ? R.start[k + 1] - 1 : length(R.points)
            pts = [SVector{2,Float64}(C.lo + R.points[i]) for i in R.start[k]:stop]
            # a region's walk ends where it began
            !R.hole[k] && length(pts) > 1 && pts[end] == pts[1] && pop!(pts)
            push!(out, (cell=w.cross.cut_list[s], region=rank, hole=R.hole[k], points=pts))
        end
    end
    return out
end

"""
    interface_mesh(cache, grid) -> (walls::Mesh{2}, cells, regions)

The walls `cache`'s last update cut, as a `Mesh{2}` of `Line`: each element's piece in each cut
cell, wound as the element is -- so MeshLibrary's `get_normal` points into the fluid -- and each
edge a sliver was closed onto, wound the same way. One element set per polyline, named as the
mesh's, and `"snapped"` for the closed edges. `cells` and `regions` give each line's cell and the
region whose wall it is. Pieces of zero length, and a dropped sliver's, are left out. Per-cell
pieces, not stitched. `grid` is the grid of that update.
"""
function interface_mesh(cache::PolylineClippingCutCellCache{T}, grid::CartesianGrid{2}) where {T}
    w = cache.work
    X = SVector{2,Float64}[]
    lines = Line{Int32}[]
    cells = CartesianIndex{2}[]
    regs = Int16[]
    loops = Int32[]
    (w.block.empty || w.topo === nothing) && return Mesh([Point(x) for x in X], lines), cells, regs
    g = _pl_check_grid(cache, grid)
    bsegs = Adapt.adapt(Array, cache.bsegs)
    arcs = Adapt.adapt(Array, cache.arcs)
    slot = Adapt.adapt(Array, cache.info.slot) # read per neighbour below; on a device cache, a scalar index
    C = w.cross
    function push_line!(a, b, ci, r, lp)
        push!(X, a, b)
        push!(lines, Line(Int32(length(X) - 1), Int32(length(X))))
        push!(cells, ci)
        push!(regs, r)
        push!(loops, lp)
    end
    for (s, ci) in enumerate(C.cut_list)
        for k in C.bseg_start[s]:(C.bseg_start[s + 1] - 1)
            b = bsegs[k]
            (b.region > 0 && b.length > 0) || continue
            # the piece's direction is its normal turned back a quarter
            f = b.length * SVector{2,Float64}(-b.normal[2], b.normal[1])
            mid = SVector{2,Float64}(b.midpoint)
            push_line!(mid - f / 2, mid + f / 2, ci, b.region, b.loop)
        end
        for k in C.arc_start[s]:(C.arc_start[s + 1] - 1)
            a = arcs[k]
            (a.closed && a.s1 > a.s0) || continue
            if a.region > 0
                # closed onto this cell's own region: its wall
                push_line!(_pl_wall_ends(g, ci, Int(a.dir), a.s0, a.s1)..., ci, a.region, Int32(0))
            elseif a.nbr_region > 0
                # this cell's sliver, closed onto a neighbour; drawn from here only when the
                # neighbour has no arcs of its own to draw it from (it was uncut)
                nb = ci + _pl_dir_offset(Int(a.dir))
                _pl_slot_of(slot, C.cut_list, nb) == 0 &&
                    push_line!(_pl_wall_ends(g, nb, _pl_opposite(Int(a.dir)), a.s0, a.s1)..., nb,
                               a.nbr_region, Int32(0))
            end
        end
    end
    names = w.topo.loop_name
    sets = MeshElementSet{Int,Vector{Int}}[]
    for lp in eachindex(names)
        el = findall(==(Int32(lp)), loops)
        isempty(el) || push!(sets, MeshElementSet(names[lp], el))
    end
    snapped = findall(==(Int32(0)), loops)
    isempty(snapped) || push!(sets, MeshElementSet("snapped", snapped))
    mesh = Mesh([Point(x) for x in X], lines; element_sets=sets)
    return mesh, cells, regs
end

@inline _pl_dir_offset(dir::Int) = dir == 1 ? CartesianIndex(-1, 0) : dir == 2 ? CartesianIndex(1, 0) :
                                  dir == 3 ? CartesianIndex(0, -1) : CartesianIndex(0, 1)
@inline _pl_opposite(dir::Int) = dir == 1 ? 2 : dir == 2 ? 1 : dir == 3 ? 4 : 3

# The ends of stretch `[s0, s1]` of cell `ci`'s face `dir`, in global coordinates, wound so the
# right-hand normal (MeshLibrary's `get_normal`) points into the cell.
function _pl_wall_ends(g::CartesianGrid{2}, ci::CartesianIndex{2}, dir::Int, s0, s1)
    lo = SVector{2,Float64}(get_node(g, ci))
    h = SVector{2,Float64}(get_node(g, ci + CartesianIndex(1, 1))) - lo
    P(t) = dir == 1 ? lo + SVector(0.0, t) : dir == 2 ? lo + SVector(h[1], t) :
           dir == 3 ? lo + SVector(t, 0.0) : lo + SVector(t, h[2])
    a, b = P(Float64(s0)), P(Float64(s1))
    return dir == 1 || dir == 4 ? (a, b) : (b, a)
end

"""
    generate_mesh(cache, grid) -> Mesh{2}

The walls `cache`'s last update cut, as a `Mesh{2}` of `Line` with one element set per polyline:
[`interface_mesh`](@ref)'s mesh, without each line's cell and region. For an exact polygon these
are its own edges, split at the grid lines. `grid` is the grid of that update.
"""
MeshLibrary.generate_mesh(cache::PolylineClippingCutCellCache, grid::CartesianGrid{2}) =
    first(interface_mesh(cache, grid))

# A simple polygon, counter-clockwise, as triangles by ear clipping: for drawing only. Repeated and
# collinear points are dropped; a polygon too degenerate to clip is fanned from what is left.
function _pl_triangulate(pts::Vector{SVector{2,Float64}})
    p = SVector{2,Float64}[]
    for q in pts
        (isempty(p) || q != p[end]) && push!(p, q)
    end
    length(p) > 1 && p[end] == p[1] && pop!(p)
    tris = NTuple{3,Int}[]
    length(p) < 3 && return p, tris
    cr(a, b, c) = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])
    idx = collect(eachindex(p))
    while length(idx) > 3
        n = length(idx)
        clipped = false
        for k in 1:n
            ia, ib, ic = idx[mod1(k - 1, n)], idx[k], idx[mod1(k + 1, n)]
            c = cr(p[ia], p[ib], p[ic])
            if c == 0
                deleteat!(idx, k)
                clipped = true
                break
            end
            c > 0 || continue
            any(j -> j != ia && j != ib && j != ic && cr(p[ia], p[ib], p[j]) >= 0 &&
                     cr(p[ib], p[ic], p[j]) >= 0 && cr(p[ic], p[ia], p[j]) >= 0, idx) && continue
            push!(tris, (ia, ib, ic))
            deleteat!(idx, k)
            clipped = true
            break
        end
        clipped || break
    end
    for k in 2:(length(idx) - 1)
        cr(p[idx[1]], p[idx[k]], p[idx[k + 1]]) > 0 && push!(tris, (idx[1], idx[k], idx[k + 1]))
    end
    return p, tris
end

"""
    write_cache_vtk(prefix, cache, grid; walls=true, regions=true, edges=true) -> Vector{String}

Write what `cache`'s last update produced, for ParaView:

- `prefix_cells.vti` -- the grid, with every cell's volume fraction, kind, status, region count,
  flags, ambiguity, closure residual and four face fractions;
- `prefix_walls.vtu` -- the walls ([`interface_mesh`](@ref)), with each line's region, cell,
  polyline and normal (into the fluid);
- `prefix_regions.vtu` -- every region's loop ([`region_loops`](@ref)) filled, with its region,
  cell, area, whether its cell is split, and whether it is an island;
- `prefix_edges.vtu` -- every edge of every cut cell, with its aperture and open-part centroid.

Returns the files written. `grid` is the grid of that update.
"""
function write_cache_vtk(prefix::AbstractString, cache::PolylineClippingCutCellCache,
                         grid::CartesianGrid{2}; walls::Bool=true, regions::Bool=true,
                         edges::Bool=true)
    cells = Adapt.adapt(Array, cache.cells)
    info = Adapt.adapt(Array, cache.info)
    data = Dict{String,Any}("volume_fraction" => cells.volume_fraction,
                            "kind" => Int32.(cells.kind),
                            "status" => Int32.(info.status),
                            "nregion" => Int32.(info.nregion),
                            "flags" => Int32.(info.flags),
                            "ambiguous" => Int32.(cells.ambiguous),
                            "residual" => map(r -> maximum(abs, r), info.residual))
    for (dir, name) in enumerate(("-x", "+x", "-y", "+y"))
        data["face_fraction_$name"] = map(f -> f[dir], cells.face_fraction)
    end
    files = String[]
    CartesianMeshes.write_vtk(prefix * "_cells", grid; cell_data=data)
    push!(files, first(filter(isfile, prefix * "_cells" .* (".vti", ".vtr"))))
    lin = LinearIndices(Tuple(grid.n))
    if walls
        wm, wc, wr = interface_mesh(cache, grid)
        nrm = zeros(3, length(wm.elements))
        loop = zeros(Int32, length(wm.elements))
        for (k, set) in enumerate(wm.elemset)
            loop[set.elems] .= set.name == "snapped" ? 0 : k
        end
        for (k, e) in enumerate(wm.elements)
            a, b = wm.nodes.coord[e.con[1]], wm.nodes.coord[e.con[2]]
            f = b - a
            L = sqrt(f[1]^2 + f[2]^2)
            nrm[1, k], nrm[2, k] = f[2] / L, -f[1] / L
        end
        MeshLibrary.write_vtk(wm, prefix * "_walls";
                              cell_data=Dict("region" => Int32.(wr), "cell" => Int64[lin[c] for c in wc],
                                             "polyline" => loop, "normal" => nrm))
        push!(files, prefix * "_walls.vtu")
    end
    if regions
        X = SVector{2,Float64}[]
        tris = Tri{Int32}[]
        rid = Int32[]
        rcell = Int64[]
        rarea = Float64[]
        rsplit = Int32[]
        rhole = Int32[]
        for lp in region_loops(cache, grid)
            p, tr = _pl_triangulate(lp.points)
            base = length(X)
            append!(X, p)
            a = 0.0
            for i in eachindex(p)
                q, r = p[i], p[mod1(i + 1, length(p))]
                a += (q[1] * r[2] - q[2] * r[1]) / 2
            end
            for t in tr
                push!(tris, Tri(Int32(base + t[1]), Int32(base + t[2]), Int32(base + t[3])))
                push!(rid, lp.region)
                push!(rcell, lin[lp.cell])
                push!(rarea, a)
                push!(rsplit, info.nregion[lp.cell] >= 2)
                push!(rhole, lp.hole)
            end
        end
        rm = Mesh([Point(x) for x in X], tris)
        MeshLibrary.write_vtk(rm, prefix * "_regions";
                              cell_data=Dict("region" => rid, "cell" => rcell, "loop_area" => rarea,
                                             "split" => rsplit, "hole" => rhole))
        push!(files, prefix * "_regions.vtu")
    end
    if edges
        ax = Adapt.adapt(Array, cache.edges.ax)
        ay = Adapt.adapt(Array, cache.edges.ay)
        seen = Set{Tuple{Int,CartesianIndex{2}}}()
        for ci in cache.work.cross.cut_list
            push!(seen, (1, ci), (1, ci + CartesianIndex(1, 0)), (2, ci), (2, ci + CartesianIndex(0, 1)))
        end
        X = SVector{2,Float64}[]
        el = Line{Int32}[]
        frac = Float64[]
        off = Float64[]
        for (axis, ei) in sort!(collect(seen))
            a = SVector{2,Float64}(get_node(grid, ei))
            b = SVector{2,Float64}(get_node(grid, ei + (axis == 1 ? CartesianIndex(0, 1) : CartesianIndex(1, 0))))
            push!(X, a, b)
            push!(el, Line(Int32(length(X) - 1), Int32(length(X))))
            e = axis == 1 ? ax[ei] : ay[ei]
            push!(frac, e.fraction)
            push!(off, e.offset)
        end
        MeshLibrary.write_vtk(Mesh([Point(x) for x in X], el), prefix * "_edges";
                              cell_data=Dict("aperture" => frac, "centroid_offset" => off))
        push!(files, prefix * "_edges.vtu")
    end
    return files
end
